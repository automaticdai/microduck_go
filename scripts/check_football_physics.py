"""Hold a calibrated standing ctrl for 3 s; check tilt and ball clearance.

Run before training: uv run scripts/check_football_physics.py
This isolates target-pose physics (no pushes or event-based parameter DR), not
policy skill. BAM's built-in battery-voltage variation remains enabled.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_football_env_cfg import (
    STAND_HIP_CTRL_OFFSET, STAND_ANKLE_CTRL_OFFSET, make_microduck_football_env_cfg,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--output", default="logs/football_physics.json")
    parser.add_argument("--sweep", action="store_true", help="Diagnose hip/ankle offsets instead of asserting HOME stability")
    parser.add_argument("--hip-offset", type=float, default=STAND_HIP_CTRL_OFFSET)
    parser.add_argument("--ankle-offset", type=float, default=STAND_ANKLE_CTRL_OFFSET)
    args = parser.parse_args()
    cfg = make_microduck_football_env_cfg()
    cfg.seed = 42
    cfg.scene.num_envs = 81 if args.sweep else args.num_envs
    cfg.curriculum = {}
    cfg.terminations = {}
    cfg.events = {k: v for k, v in cfg.events.items() if k in (
        "expand_bam_friction_fields", "reset_base", "reset_robot_joints", "reset_football",
    )}
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0., 0.), "y": (0., 0.), "z": (0.12, 0.12),
        "roll": (-0.015, 0.015), "pitch": (-0.015, 0.015), "yaw": (-3.14, 3.14),
    }
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.01, 0.01)
    cfg.events["reset_football"].params["near_probability"] = 1.0
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset(seed=42)
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        initial_ball = env.scene["ball"].data.root_link_pos_w.clone()
        penetrations = []
        for qpos in env.sim.data.qpos.cpu().numpy():
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                names = [model.geom(int(i)).name or "" for i in contact.geom]
                if any(n.startswith("ball/") for n in names) and any(n.startswith("robot/") for n in names):
                    penetrations.append(max(0., -float(contact.dist)))
        actions = torch.zeros(env.num_envs, 14, device=env.device)
        names = [name for name in env.scene["robot"].joint_names if not name.startswith("passive_")]
        offsets = torch.tensor([args.hip_offset, args.ankle_offset], device=env.device).repeat(env.num_envs, 1)
        if args.sweep:
            offsets = torch.cartesian_prod(torch.linspace(-0.20, 0.20, 9), torch.linspace(-0.20, 0.20, 9)).to(env.device)
        for name, column, sign in (("left_hip_pitch", 0, 1), ("right_hip_pitch", 0, -1),
                                   ("left_ankle", 1, 1), ("right_ankle", 1, -1)):
            actions[:, names.index(name)] = sign * offsets[:, column]
        for _ in range(round(3.0 / env.step_dt)):
            _, rewards, _, _, _ = env.step(actions)
            assert torch.isfinite(rewards).all()
        robot = env.scene["robot"].data
        q = robot.root_link_quat_w
        tilt = torch.rad2deg(torch.acos((1 - 2 * (q[:, 1].square() + q[:, 2].square())).clamp(-1, 1)))
        heights = robot.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
        ball_drift = torch.linalg.vector_norm(env.scene["ball"].data.root_link_pos_w[:, :2] - initial_ball[:, :2], dim=1)
        result = {
            "num_envs": env.num_envs, "seconds": 3.0,
            "upright_fraction_tilt_under_10deg": (tilt < 10).float().mean().item(),
            "tilt_degrees_p50_p95": np.percentile(tilt.cpu().numpy(), [50, 95]).tolist(),
            "root_height_m_p05_p50_p95": np.percentile(heights.cpu().numpy(), [5, 50, 95]).tolist(),
            "max_robot_ball_spawn_penetration_m": max(penetrations, default=0.0),
            "max_ball_drift_m": ball_drift.max().item(),
            "hip_ctrl_offset_rad": args.hip_offset,
            "ankle_ctrl_offset_rad": args.ankle_offset,
            "settled_joint_positions_rad": dict(zip(env.scene["robot"].joint_names, robot.joint_pos.median(dim=0).values.cpu().tolist())),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if args.sweep:
            for idx in tilt.argsort()[:12].tolist():
                print("sweep", offsets[idx].tolist(), "tilt", tilt[idx].item(), "height", heights[idx].item())
            return
        assert result["max_robot_ball_spawn_penetration_m"] < 1e-5
        assert result["upright_fraction_tilt_under_10deg"] >= 0.9
        assert result["max_ball_drift_m"] < 0.01
    finally:
        env.close()


if __name__ == "__main__":
    main()
