"""Keep football compatible with the real robot and its walking recipe."""
import re

import pytest

from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.tasks.microduck_football_env_cfg import (
    MicroduckFootballRlCfg, make_microduck_football_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from mjlab_microduck.robot.microduck_constants import get_standup_spec, get_backlash_spec


def test_registered_with_groundcontact_backlash_twin():
    for task in ("Mjlab-Football-Flat-MicroDuck", "Mjlab-Football-Flat-Backlash-MicroDuck"):
        assert task in list_tasks()
    base = load_env_cfg("Mjlab-Football-Flat-MicroDuck")
    twin = load_env_cfg("Mjlab-Football-Flat-Backlash-MicroDuck")
    assert base.scene.entities["robot"].spec_fn is get_standup_spec
    assert twin.scene.entities["robot"].spec_fn is get_backlash_spec


def test_actor_contract_and_transfer_stack_inherited():
    cfg = make_microduck_football_env_cfg()
    walking = make_microduck_velocity_env_cfg()
    assert list(cfg.observations["actor"].terms) == list(walking.observations["actor"].terms)
    assert list(cfg.observations["actor"].terms)[-3:] == ["command", "head_command", "body_command"]
    assert "ball_position" not in cfg.observations["actor"].terms
    assert "ball_position" in cfg.observations["critic"].terms
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    for event in ("expand_bam_friction_fields", "randomize_joint_friction", "encoder_bias"):
        assert cfg.events[event].func is walking.events[event].func
    assert cfg.terminations["nan_state"].params["sensor_names"]
    assert "ball_nan" in cfg.terminations
    assert MicroduckFootballRlCfg.actor.obs_normalization
    assert MicroduckFootballRlCfg.algorithm.symmetry_cfg is None
    for name in ("head_pose", "body_pose"):
        assert all(lo < 0 < hi for lo, hi in cfg.commands[name].ranges)
        assert cfg.commands[name].zero_command_prob > 0


def test_rewards_spawns_and_play_are_independent():
    cfg = make_microduck_football_env_cfg()
    play = make_microduck_football_env_cfg(play=True)
    assert list(cfg.events)[-1] == "reset_football"
    assert cfg.events["reset_football"].params["near_probability"] == 0.25
    assert play.events["reset_football"].params["near_probability"] == 0.0
    assert "football_spawn" not in play.curriculum
    assert "standing_envs" not in cfg.curriculum
    assert cfg.rewards["ball_speed_overshoot"].weight < 0
    assert all(cfg.rewards[f"football_{name}"].weight > 0 for name in ("approach", "speed", "distance"))


@pytest.mark.parametrize("task", ["Mjlab-Football-Flat-MicroDuck", "Mjlab-Football-Flat-Backlash-MicroDuck"])
def test_contact_selectors_and_servo_names_resolve_on_actual_models(task):
    cfg = load_env_cfg(task)
    model = cfg.scene.entities["robot"].spec_fn().compile()
    servos = [model.joint(i).name for i in range(model.njnt)
              if model.jnt_type[i] == 3 and not model.joint(i).name.startswith("passive_")]
    assert len(servos) == 14
    for sensor in cfg.scene.sensors:
        if sensor.name in ("kick_ball_contact", "kick_support_contact"):
            matches = [model.geom(i).name for i in range(model.ngeom)
                       if re.fullmatch(sensor.primary.pattern, model.geom(i).name or "")]
            assert len(matches) == 1


def test_goal_physics_and_reward_match_clear_opening():
    from mjlab_microduck.tasks.microduck_football_env_cfg import GOAL_WIDTH, GOAL_HEIGHT, GOAL_POST_RADIUS
    cfg = make_microduck_football_env_cfg()
    model = cfg.scene.entities['goal'].spec_fn().compile()
    assert model.nmocap == 1
    assert model.geom('left_post').pos[1] - GOAL_POST_RADIUS == pytest.approx(GOAL_WIDTH / 2)
    assert model.geom('crossbar').pos[2] - GOAL_POST_RADIUS == pytest.approx(GOAL_HEIGHT)
    assert model.geom('back_net').contype > 0
    assert cfg.commands['twist'].goal_width == GOAL_WIDTH
    assert cfg.commands['twist'].goal_height == GOAL_HEIGHT
    assert cfg.rewards['football_goal'].weight > cfg.rewards['football_distance'].weight


def test_goal_posts_block_ball_and_mouth_allows_entry_in_cpu_physics():
    import mujoco
    from mjlab_microduck.tasks.microduck_football_env_cfg import get_football_goal_spec, GOAL_WIDTH, GOAL_POST_RADIUS
    for y, should_enter in ((0., True), (GOAL_WIDTH / 2 + GOAL_POST_RADIUS, False)):
        spec = get_football_goal_spec()
        spec.option.gravity = [0., 0., 0.]
        spec.option.timestep = 0.001
        spec.body('goal').pos = [0., 0., 0.]
        ball = spec.worldbody.add_body(name='test_ball', pos=[-0.25, y, 0.035])
        ball.add_freejoint()
        ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.035, 0., 0.], mass=0.015)
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qvel[0] = 1.0
        furthest = -0.25
        for _ in range(450):
            mujoco.mj_step(model, data)
            furthest = max(furthest, data.qpos[0])
        assert (furthest > 0.035) == should_enter
        if should_enter:
            assert furthest < 0.26  # Back panel retains the ball.


def test_rebuild_keeps_walking_pose_and_checks_nonfoot_support():
    cfg = make_microduck_football_env_cfg()
    walking = make_microduck_velocity_env_cfg()
    assert cfg.rewards['pose'].func is walking.rewards['pose'].func
    assert cfg.rewards['pose'].weight == walking.rewards['pose'].weight
    assert cfg.rewards['head_pose_tracking'].weight > 0
    assert 'lost_balance' in cfg.terminations
    assert 'football_body_ground' in cfg.terminations['nan_state'].params['sensor_names']
    assert cfg.commands['twist'].min_standing_height == pytest.approx(.095)
    assert MicroduckFootballRlCfg.actor.distribution_cfg['class_name'].endswith(':FootballGaussianDistribution')
    model = cfg.scene.entities['robot'].spec_fn().compile()
    sensor = next(s for s in cfg.scene.sensors if s.name == 'football_body_ground')
    floor_sensitive = [model.body(model.geom_bodyid[i]).name for i in range(model.ngeom)
                       if model.geom_contype[i] and re.fullmatch(sensor.primary.pattern, model.body(model.geom_bodyid[i]).name)]
    assert 'trunk_base' in floor_sensitive and 'jaw_soft' in floor_sensitive
    assert 'ankle_left' not in floor_sensitive and 'ankle_right' not in floor_sensitive
