"""Measure approach-and-kick success over complete episodes, optionally record one.

uv run scripts/eval_football.py --checkpoint-file logs/.../model_1000.pt
Use --near-probability 1 for the kick-only battery; default tests approaches.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per environment")
    parser.add_argument("--near-probability", type=float, default=0.0)
    parser.add_argument("--max-distance", type=float, default=0.9)
    parser.add_argument("--idle", action="store_true", help="Validate standing with exact-zero navigation commands")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("logs/football_eval.json"))
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 1 or not 0 <= args.near_probability <= 1 or args.max_distance < 0.2:
        parser.error("Invalid evaluation batch size or spawn range")
    task = "Mjlab-Football-Flat-MicroDuck"
    cfg, agent = load_env_cfg(task, play=True), load_rl_cfg(task)
    cfg.seed = 123  # Seed startup DR too; reset(seed=...) alone is too late.
    cfg.scene.num_envs = args.num_envs
    cfg.auto_reset = False
    if args.video:
        cfg.viewer.width, cfg.viewer.height = 640, 480
        cfg.viewer.max_extra_envs = 0
        cfg.viewer.enable_shadows = False
        cfg.viewer.enable_reflections = False
    cfg.events["reset_football"].params.update(near_probability=args.near_probability, max_distance=args.max_distance)
    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(task)(wrapped, asdict(agent), device="cuda:0")
        runner.load(str(args.checkpoint_file), load_cfg={"actor": True}, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        if args.idle:
            command = env.command_manager.get_term("twist")
            command._update_command = lambda: command._command.zero_()
        obs, _ = env.reset(seed=123)
        counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        records, frames, heights, tilts = [], [], [], []
        with torch.inference_mode():
            for step in range(env.max_episode_length * args.episodes + 1):
                action = policy(obs)
                obs, _, terminated, timeout, _ = env.step(action)
                if args.video and counts[0] == 0 and step % 2 == 0:
                    frames.append(env.render())
                active = counts < args.episodes
                data = env.scene['robot'].data
                floor = env.scene.terrain.env_origins[:, 2]
                height = data.root_link_pos_w[:, 2] - floor
                q = data.root_link_quat_w
                tilt = torch.rad2deg(torch.acos((1 - 2 * (q[:, 1].square() + q[:, 2].square())).clamp(-1, 1)))
                heights.extend(height[active].tolist())
                tilts.extend(tilt[active].tolist())
                done = terminated | timeout
                cmd = env.command_manager.get_term("twist")
                for idx in torch.where(done & (counts < args.episodes))[0].tolist():
                    records.append({"struck": bool(cmd.struck[idx]), "scored": bool(cmd.scored[idx]), "success": bool(cmd.success[idx]) and not bool(terminated[idx]),
                                    "fell_or_invalid": bool(terminated[idx]),
                                    "approached": bool(cmd.approached[idx]),
                                    "standing": bool(cmd.standing[idx]),
                                    "balance_failed": bool(cmd.balance_failed[idx]),
                                    "final_root_height_m": float(height[idx]),
                                    "final_tilt_deg": float(tilt[idx]),
                                    "approach_progress_m": float(cmd.best_approach[idx]),
                                    "credited_speed_m_s": float(cmd.best_speed[idx]),
                                    "peak_ball_speed_m_s": float(cmd.peak_ball_speed[idx]),
                                    "credited_distance_m": float(cmd.best_distance[idx])})
                counts += done.long()
                if (counts >= args.episodes).all():
                    break
                if done.any():
                    obs, _ = env.reset(env_ids=torch.where(done)[0])
        assert len(records) == args.num_envs * args.episodes
        result = {"checkpoint": str(args.checkpoint_file), "seed": cfg.seed, "episodes": len(records),
                  "near_probability": args.near_probability, "max_spawn_distance_m": args.max_distance,
                  "goal_width_m": cfg.commands["twist"].goal_width,
                  "goal_height_m": cfg.commands["twist"].goal_height,
                  "goal_distance_m": cfg.commands["twist"].target_distance,
                  "goal_angle_range_rad": cfg.events["reset_football"].params["goal_angle_range"]}
        result["idle"] = args.idle
        result["root_height_quantiles_m"] = np.quantile(heights, [0.05, 0.5, 0.95]).tolist()
        result["tilt_quantiles_deg"] = np.quantile(tilts, [0.05, 0.5, 0.95]).tolist()
        for name in records[0]:
            result[name + "_mean"] = float(np.mean([r[name] for r in records]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(args.video, frames, fps=round(1 / env.step_dt / 2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
