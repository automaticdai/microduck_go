"""Approach a ball and kick it through a physical goal, then stop upright.

The simulated ball tracker supplies a body-frame velocity command to reach a
right-foot kicking stance, with a small forward command near the ball. A valid
foot strike switches that command to exact zero. This preserves the 61D actor
contract; deploying the ONNX alone is insufficient: a tracker must reproduce
FootballCommand's navigation and strike detection. Ball state is critic-only.

A quarter of initial episodes start near the foot so kick discovery does not depend
on first learning to walk. The remaining episodes already require an approach;
the spawn curriculum extends that distance after 1000 iterations. The goal
direction varies by ±20 degrees. A whole-ball crossing through its mouth pays
once; smaller speed/distance rewards shape supported right-foot kicks with
bounded episode totals. Holding a rolling ball state never pays indefinitely.
"""

from copy import deepcopy

import mujoco
from mjlab.entity import EntityCfg

from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_CFG, MICRODUCK_STANDUP_ROBOT_CFG
from . import mdp
from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

ENABLE_SYMMETRY = False  # A right-foot strike is asymmetric.
MIN_STANDING_HEIGHT = 0.095  # Walking baseline p5=0.115 m; failed policy median=0.057 m.
EPISODE_LENGTH_S = 8.0
BALL_RADIUS = 0.035
KICK_OFFSET = (0.10, -0.042)
TARGET_BALL_SPEED = 1.0
TARGET_BALL_DISTANCE = 0.75
GOAL_WIDTH = 0.40  # Clear opening between the inner edges of the posts.
GOAL_HEIGHT = 0.30
GOAL_POST_RADIUS = 0.012
GOAL_ANGLE_RANGE = (-0.35, 0.35)  # Radians relative to reset heading.
# Measured with BAM, 64 noisy starts held for 3 s: root z median 0.1145 m,
# tilt p95 5.95 deg. Control offsets are +0.05 rad on left hip/ankle and
# -0.05 on right; loaded equilibrium differs from ctrl because BAM is compliant.
STAND_HIP_CTRL_OFFSET = 0.05
STAND_ANKLE_CTRL_OFFSET = 0.05



def get_football_goal_spec():
    """Movable fixed goal; rigid translucent panels act as a simple net."""
    w, h, r = GOAL_WIDTH / 2, GOAL_HEIGHT, GOAL_POST_RADIUS
    return mujoco.MjSpec.from_string(f"""<mujoco><worldbody>
      <body name="goal" mocap="true" pos="1.5 0 0">
        <geom name="left_post" type="capsule" fromto="0 {w+r} 0 0 {w+r} {h+r}" size="{r}" rgba="1 1 1 1"/>
        <geom name="right_post" type="capsule" fromto="0 {-w-r} 0 0 {-w-r} {h+r}" size="{r}" rgba="1 1 1 1"/>
        <geom name="crossbar" type="capsule" fromto="0 {-w-r} {h+r} 0 {w+r} {h+r}" size="{r}" rgba="1 1 1 1"/>
        <geom name="back_net" type="box" pos="0.26 0 {h/2}" size="0.005 {w+r} {h/2}" rgba="0.2 0.6 0.9 0.35"/>
        <geom name="left_net" type="box" pos="0.13 {w+r} {h/2}" size="0.13 0.005 {h/2}" rgba="0.2 0.6 0.9 0.2"/>
        <geom name="right_net" type="box" pos="0.13 {-w-r} {h/2}" size="0.13 0.005 {h/2}" rgba="0.2 0.6 0.9 0.2"/>
        <geom name="goal_line" type="box" pos="0 0 0.001" size="0.008 {w} 0.001" contype="0" conaffinity="0" rgba="1 1 1 1"/>
      </body></worldbody></mujoco>""")


def make_microduck_football_env_cfg(play: bool = False, rough: bool = False):
    if rough:
        raise ValueError("Football currently requires flat ground")
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play))
    cfg.scene.entities = {"robot": deepcopy(MICRODUCK_STANDUP_ROBOT_CFG), "ball": deepcopy(MICRODUCK_BALL_CFG),
                          "goal": EntityCfg(spec_fn=get_football_goal_spec)}
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.viewer.distance = 2.3
    cfg.viewer.elevation = -25.0
    cfg.sim.nconmax = 100
    cfg.sim.mujoco.iterations = 30

    kick_contact = ContactSensorCfg(
        name="kick_ball_contact",
        primary=ContactMatch(mode="geom", pattern="^right_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="^ball_geom$", entity="ball"),
        fields=("found", "force"), reduce="netforce", num_slots=1, history_length=cfg.decimation,
    )
    support_contact = ContactSensorCfg(
        name="kick_support_contact",
        primary=ContactMatch(mode="geom", pattern="^left_foot_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, history_length=cfg.decimation,
    )
    body_ground = ContactSensorCfg(
        name="football_body_ground",
        primary=ContactMatch(mode="body", pattern="^(?!ankle_left$|ankle_right$).*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, history_length=cfg.decimation,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, kick_contact, support_contact, body_ground)
    cfg.terminations["lost_balance"] = TerminationTermCfg(func=mdp.football_lost_balance)
    cfg.terminations["nan_state"].params["sensor_names"] = (
        "feet_ground_contact", "kick_ball_contact", "kick_support_contact", "football_body_ground",
    )
    cfg.commands["twist"] = mdp.FootballCommandCfg(
        resampling_time_range=(EPISODE_LENGTH_S + 1, EPISODE_LENGTH_S + 1),
        kick_offset=KICK_OFFSET, target_speed=TARGET_BALL_SPEED,
        target_distance=TARGET_BALL_DISTANCE, goal_width=GOAL_WIDTH,
        goal_height=GOAL_HEIGHT, ball_radius=BALL_RADIUS, min_standing_height=MIN_STANDING_HEIGHT,
    )
    # Retain every slot, with tiny ranges and an explicit idle bucket.
    for name in ("head_pose", "body_pose"):
        cmd = cfg.commands[name]
        cmd.ranges = tuple((-0.001, 0.001) for _ in cmd.ranges)
        cmd.zero_command_prob = 0.25
    for name in ("standing_envs", "head_pose_range", "body_pose_range", "head_pose_bias_weight"):
        cfg.curriculum.pop(name, None)
    for name in ("body_pose_tracking", "head_pose_bias"):
        cfg.rewards[name].weight = 0.0
    # Preserve the walking expert's leg posture and head tracking rewards.
    # The old broad whole-body pose term allowed a level but collapsed trunk.
    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["action_rate_l2"].weight = 0.0
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.0}, {"step": 500 * 24, "weight": -0.01},
        {"step": 1500 * 24, "weight": -0.03}, {"step": 3000 * 24, "weight": -0.05},
    ]
    for component, weight in (("approach", 20.0), ("speed", 1.0), ("distance", 3.0), ("goal", 8.0)):
        cfg.rewards[f"football_{component}"] = RewardTermCfg(
            func=mdp.football_progress, weight=weight, params={"component": component},
        )
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(
        func=mdp.ball_speed_overshoot_penalty, weight=-2.0,
        params={"target_speed": TARGET_BALL_SPEED},
    )
    for name, func in (("ball_position", mdp.ball_pos_in_base), ("ball_velocity", mdp.ball_vel_in_base),
                       ("football_state", mdp.football_critic_state)):
        cfg.observations["critic"].terms[name] = ObservationTermCfg(func=func)
    cfg.terminations["ball_nan"] = TerminationTermCfg(
        func=mdp.robot_state_is_nan, params={"asset_cfg": SceneEntityCfg("ball")},
    )
    # Reset ball last: read the actual randomized robot root, not stale body data.
    cfg.events["reset_football"] = EventTermCfg(
        func=mdp.reset_football, mode="reset",
        params={"near_probability": 0.25, "max_distance": 0.35, "ball_radius": BALL_RADIUS,
                "goal_angle_range": GOAL_ANGLE_RANGE},
    )
    cfg.curriculum["football_spawn"] = CurriculumTermCfg(
        func=mdp.event_param_curriculum,
        params={"event_name": "reset_football", "param_stages": [
            {"step": 0, "params": {"max_distance": 0.35, "near_probability": 0.25}},
            {"step": 1000 * 24, "params": {"max_distance": 0.6, "near_probability": 0.20}},
            {"step": 2000 * 24, "params": {"max_distance": 0.9, "near_probability": 0.2}},
        ]},
    )
    if "push_robot" in cfg.events:
        cfg.curriculum["football_push"] = CurriculumTermCfg(
            func=mdp.push_curriculum,
            params={"event_name": "push_robot", "push_stages": [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 2000 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
            ]},
        )
    for name in ("struck", "scored", "success", "approached", "standing", "balance_failed", "best_speed", "best_distance", "peak_ball_speed"):
        cfg.metrics[f"football_{name}"] = MetricsTermCfg(
            func=mdp.football_metric, params={"name": name}, reduce="last",
        )
    if play:
        cfg.curriculum.pop("football_spawn")
        cfg.events["reset_football"].params.update(max_distance=0.9, near_probability=0.0)
        cfg.events.pop("push_robot", None)
        cfg.curriculum.pop("football_push", None)
    return cfg


MicroduckFootballRlCfg = deepcopy(MicroduckRlCfg)
MicroduckFootballRlCfg.experiment_name = "football"
MicroduckFootballRlCfg.run_name = "approach_kick"
MicroduckFootballRlCfg.max_iterations = 4000
MicroduckFootballRlCfg.algorithm.symmetry_cfg = None

# Conservative fine-tuning from walking; exploration bounds are task-local.
MicroduckFootballRlCfg.actor.distribution_cfg = {
    "class_name": "mjlab_microduck.tasks.mdp:FootballGaussianDistribution",
    "init_std": 0.2, "std_type": "scalar",
}
MicroduckFootballRlCfg.algorithm.entropy_coef = 0.001
MicroduckFootballRlCfg.algorithm.learning_rate = 1e-4
MicroduckFootballRlCfg.algorithm.schedule = "fixed"
MicroduckFootballRlCfg.save_interval = 100
