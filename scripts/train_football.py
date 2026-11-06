"""Train football, optionally initializing ONLY the actor from a walking checkpoint.

The football critic has extra ball/state inputs, so ordinary cross-task resume
is incompatible. Actor weights and their normalizer transfer together; critic,
optimizer, iteration and task curricula start fresh. Run 64 envs / 5 iterations
before a full run, including when changing the source checkpoint.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--from-walking", type=Path)
    source_group.add_argument("--resume", type=Path, help="Resume a football checkpoint including critic, optimizer and curricula")
    source_group.add_argument("--from-policy", type=Path, help="Actor-only initialization from a compatible football checkpoint")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=0, help="Save and gate training on deterministic standing/approach evaluations every N iterations")
    parser.add_argument("--continue-on-regression", action="store_true",
                        help="Report evaluation regressions without stopping training")
    parser.add_argument("--run-name", default="football-warm-smoke")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0 or args.eval_every < 0:
        parser.error("num-envs and iterations must be positive")
    task = "Mjlab-Football-Flat-MicroDuck"
    cfg, agent = load_env_cfg(task), load_rl_cfg(task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed
    cfg.sim.nan_guard.enabled = args.iterations <= 5
    agent.logger = "tensorboard"
    agent.max_iterations = args.iterations
    agent.run_name = args.run_name
    if args.eval_every:
        agent.save_interval = args.eval_every
    source_checkpoint = args.from_walking or args.from_policy
    if source_checkpoint:
        source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        actor = source["actor_state_dict"]
        if actor["obs_normalizer._std"].shape != (1, 61):
            raise ValueError("Source must use the current 61D observation contract")
        if args.from_walking and actor["obs_normalizer._std"][0, 48:51].min() < 0.02:
            raise ValueError("Source twist normalizer looks like a stand expert, not a walking policy")
    if source_checkpoint or args.resume:
        # Source walking policy already experienced mature CoM DR. Restart only
        # football's spawn/push/smoothing schedules, not those inherited ranges.
        for name in ("com_range", "head_com_range"):
            if name in cfg.curriculum:
                stages = cfg.curriculum[name].params["range_stages"]
                cfg.curriculum[name].params["range_stages"] = [{**stages[-1], "step": 0}]
    log_dir = Path("logs/rsl_rl/football") / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    log_dir.mkdir(parents=True)
    dump_yaml(log_dir / "params/env.yaml", asdict(cfg))
    dump_yaml(log_dir / "params/agent.yaml", asdict(agent))
    (log_dir / "initialization.json").write_text(json.dumps({
        "walking_checkpoint": str(args.from_walking.resolve()) if args.from_walking else None,
        "policy_checkpoint": str(args.from_policy.resolve()) if args.from_policy else None,
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "actor_only": bool(source_checkpoint), "iterations": args.iterations,
        "recipe": "standing-rebuild-v2", "eval_every": args.eval_every,
        "continue_on_regression": args.continue_on_regression,
    }, indent=2) + "\n")
    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(task)(wrapped, asdict(agent), str(log_dir), device="cuda:0")
        if source_checkpoint:
            runner.load(str(source_checkpoint), load_cfg={"actor": True}, map_location="cuda:0")
            env.common_step_counter = 0
            runner.current_learning_iteration = 0
        elif args.resume:
            runner.load(str(args.resume), map_location="cuda:0")
            # Checkpoints store the last completed iteration, not the next one.
            runner.current_learning_iteration += 1
            print(f"Resumed iteration {runner.current_learning_iteration}, env step {env.common_step_counter}", flush=True)
        if args.eval_every:
            original_save = runner.save
            evaluated = set()
            def guarded_save(path, infos=None):
                original_save(path, infos)
                if path in evaluated:
                    return
                evaluated.add(path)
                checks, failures = {}, []
                for mode in ("idle", "approach"):
                    output = log_dir / f"{Path(path).stem}_{mode}.json"
                    command = [sys.executable, "scripts/eval_football.py", "--checkpoint-file", str(path),
                               "--num-envs", "32", "--episodes", "1", "--output", str(output)]
                    if mode == "idle":
                        command.append("--idle")
                    print(f"Checking {mode} for {path}", flush=True)
                    with output.with_suffix(".log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    result = json.loads(output.read_text())
                    checks[mode] = result
                    if result["fell_or_invalid_mean"] > 0.10 or result["standing_mean"] < 0.90:
                        failures.append(f"{mode}: standing/balance regression")
                    if mode == "approach" and result["approached_mean"] < 0.60:
                        failures.append("approach: fewer than 60% reach the kicking stance")
                report = {"checkpoint": str(path), "status": (("regression_advisory" if args.continue_on_regression else "stopped_by_evaluation")
                                     if failures else "passed"),
                          "failures": failures, "checks": checks}
                (log_dir / "latest_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"Deterministic evaluation: {report['status']}; "
                      f"approached={checks['approach']['approached_mean']:.1%}, "
                      f"scored={checks['approach']['scored_mean']:.1%}", flush=True)
                if failures and not args.continue_on_regression:
                    raise RuntimeError("Training stopped: " + "; ".join(failures))
            runner.save = guarded_save
        runner.add_git_repo_to_log(__file__)
        print(f"Training output: {log_dir}", flush=True)
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
