"""Football navigation, reward gates, and per-world reset regressions (CPU)."""
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck.tasks import mdp


def test_navigation_approaches_aligns_and_stops_after_strike():
    ball = torch.tensor([[0.5, -0.042], [0.10, -0.042], [0.5, 0.2], [0.5, 0.0]])
    direction = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [1., 0.]])
    cmd = mdp.football_navigation_command(ball, direction, torch.tensor([False, False, False, True]))
    assert cmd[0, 0] == pytest.approx(0.35)
    assert cmd[0, 1] == pytest.approx(0.0)
    assert cmd[1, 0] == pytest.approx(0.08)  # Forward step cue at kicking stance.
    assert cmd[2, 2] > 0
    assert torch.equal(cmd[3], torch.zeros(3))


def make_env():
    n = 4
    robot = NS(root_link_pos_w=torch.zeros(n, 3), root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1))
    ball = NS(root_link_pos_w=torch.tensor([[0.3, -0.042, 0.035]]).repeat(n, 1),
              root_link_lin_vel_w=torch.zeros(n, 3))
    class Scene(dict):
        pass
    robot.root_link_pos_w[:, 2] = 0.116
    scene = Scene(robot=NS(data=robot), ball=NS(data=ball))
    scene.terrain = NS(env_origins=torch.zeros(n, 3))
    scene.sensors = {name: NS(data=NS(found=torch.zeros(n, 1, dtype=torch.bool)))
                     for name in ("kick_ball_contact", "kick_support_contact")}
    scene.sensors["football_body_ground"] = NS(data=NS(found=torch.zeros(n, 1, dtype=torch.bool)))
    env = NS(num_envs=n, device="cpu", step_dt=0.02, common_step_counter=0, scene=scene)
    command = mdp.FootballCommandCfg(resampling_time_range=(9., 9.)).build(env)
    command.reset_episode(torch.arange(n), ball.root_link_pos_w[:, :2], robot.root_link_pos_w[:, :2], mdp._ball_kick_dir(env))
    return env, command


def test_only_supported_upright_right_foot_contact_qualifies():
    env, command = make_env()
    env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 1.0
    env.scene.sensors["kick_ball_contact"].data.found[:3] = True
    env.scene.sensors["kick_support_contact"].data.found[[0, 2, 3]] = True
    env.scene["robot"].data.root_link_quat_w[2] = torch.tensor([0., 1., 0., 0.])
    command.update_progress()
    assert command.struck.tolist() == [True, False, False, False]
    assert command.best_speed.tolist() == [1., 0., 0., 0.]


def test_reward_is_bounded_no_rolling_jackpot_or_oscillation_farming():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 5.0
    env.scene["ball"].data.root_link_pos_w[:, 0] += 1.0
    command.update_progress()
    assert torch.allclose(command.progress["speed"] * env.step_dt, torch.ones(4))
    assert torch.allclose(command.progress["distance"] * env.step_dt, torch.full((4,), 0.75))
    for dx in (-0.5, 0.5, 0.0):
        env.common_step_counter += 1
        env.scene["ball"].data.root_link_pos_w[:, 0] += dx
        command.update_progress()
        assert not command.progress["speed"].any()
        assert not command.progress["distance"].any()


def test_lateral_or_backward_ball_does_not_score_distance():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    env.scene["ball"].data.root_link_pos_w[0, 0] -= 0.5
    env.scene["ball"].data.root_link_pos_w[1, :2] += torch.tensor([1., 0.5])
    command.update_progress()
    assert not command.progress["distance"].any()


def test_progress_is_cached_and_partial_reset_does_not_clear_other_worlds():
    env, command = make_env()
    env.scene["robot"].data.root_link_pos_w[:, 0] += 0.1
    command.update_progress()
    first = command.progress["approach"].clone()
    command.update_progress()
    assert torch.equal(command.progress["approach"], first)
    command.reset_episode(torch.tensor([1]), command.ball_start[1:2].clone(),
                          env.scene["robot"].data.root_link_pos_w[1:2, :2], mdp._ball_kick_dir(env)[1:2])
    assert command.progress["approach"][1] == 0
    assert torch.equal(command.progress["approach"][[0, 2, 3]], first[[0, 2, 3]])
    env.common_step_counter += 1
    command.update_progress()
    assert not command.progress["approach"].any()


def test_falling_after_kick_clears_success():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    env.scene["ball"].data.root_link_pos_w[:, 0] += 1.
    command.update_progress()
    assert command.success.all()
    env.common_step_counter += 1
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([0., 1., 0., 0.])
    command.update_progress()
    assert not command.success.any()


def test_brief_contacts_require_support_in_the_same_physics_substep():
    env, command = make_env()
    for sensor in env.scene.sensors.values():
        sensor.data.force_history = torch.zeros(4, 1, 4, 3)
    kick = env.scene.sensors["kick_ball_contact"].data.force_history
    support = env.scene.sensors["kick_support_contact"].data.force_history
    kick[:, 0, 2, 0] = 1.0
    support[0, 0, 2, 2] = 1.0
    support[1, 0, 1, 2] = 1.0  # Support at a different instant is not enough.
    command.update_progress()
    assert command.struck.tolist() == [True, False, False, False]


def test_invalid_ball_state_cannot_score():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    env.scene["ball"].data.root_link_lin_vel_w[:] = float("inf")
    command.update_progress()
    assert not command.struck.any()
    assert not command.progress["speed"].any()


def test_wrong_heading_cannot_qualify_a_strike():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    command.update_progress()
    assert not command.struck.any()


def test_actual_speed_metric_is_not_clipped_to_the_reward_target():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    env.scene["ball"].data.root_link_lin_vel_w[:, 0] = 3.0
    command.update_progress()
    assert (command.best_speed == 1.0).all()
    assert (command.peak_ball_speed == 3.0).all()


def test_scored_ball_rolling_beyond_the_target_does_not_undo_success():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    env.scene["ball"].data.root_link_pos_w[:, 0] += 0.8
    command.update_progress()
    assert command.success.all()
    env.common_step_counter += 1
    env.scene["ball"].data.root_link_pos_w[:, :2] += torch.tensor([1., 0.5])
    command.update_progress()
    assert command.success.all()
    assert not command.progress["distance"].any()



def test_goal_requires_whole_ball_between_posts_below_crossbar():
    env, command = make_env()
    for name in ("kick_ball_contact", "kick_support_contact"):
        env.scene.sensors[name].data.found[:] = True
    ball = env.scene['ball'].data.root_link_pos_w
    ball[:, 0] += 0.80
    ball[1, 1] += 0.18  # Center inside, but sphere clips a post.
    ball[2, 2] = 0.40  # Over the crossbar.
    ball[3, 0] -= 0.03  # Center over line, trailing edge not yet over.
    command.update_progress()
    assert command.scored.tolist() == [True, False, False, False]
    assert (command.progress['goal'] * env.step_dt).tolist() == [1., 0., 0., 0.]
    env.common_step_counter += 1
    command.update_progress()
    assert not command.progress['goal'].any()


def test_missed_goal_cannot_score_by_entering_from_side_or_recrossing():
    env, command = make_env()
    command.struck[:] = True
    ball = env.scene['ball'].data.root_link_pos_w
    ball[:, :2] += torch.tensor([0.80, 0.50])
    command.update_progress()
    assert not command.scored.any()
    for dx, dy in ((0., -0.50), (-0.20, 0.), (0.20, 0.)):
        ball[:, :2] += torch.tensor([dx, dy])
        env.common_step_counter += 1
        command.update_progress()
        assert not command.scored.any()


def test_goal_uses_interpolated_crossing_and_resets_selected_worlds():
    env, command = make_env()
    command.struck[:] = True
    ball = env.scene['ball'].data.root_link_pos_w
    # First reach just before the whole-ball line, then a fast diagonal shot.
    ball[:, 0] += 0.78
    command.update_progress()
    env.common_step_counter += 1
    ball[:, :2] += torch.tensor([0.22, 0.40])
    command.update_progress()
    assert command.scored.all()  # Crossing is in the mouth despite end position.
    command.reset_episode(torch.tensor([1]), command.ball_start[1:2].clone(),
                          env.scene['robot'].data.root_link_pos_w[1:2, :2], mdp._ball_kick_dir(env)[1:2])
    assert command.scored.tolist() == [True, False, True, True]
    assert not command.crossed_goal_line[1]


def test_unqualified_ball_crossing_does_not_score():
    env, command = make_env()
    env.scene['ball'].data.root_link_pos_w[:, 0] += 0.8
    command.update_progress()
    assert not command.scored.any()


def test_level_but_collapsed_duck_cannot_score_and_terminates():
    env, command = make_env()
    env.scene['robot'].data.root_link_pos_w[:, 2] = 0.057
    for name in ('kick_ball_contact', 'kick_support_contact'):
        env.scene.sensors[name].data.found[:] = True
    env.scene['ball'].data.root_link_pos_w[:, 0] += 0.8
    for _ in range(7):
        command.update_progress()
        env.common_step_counter += 1
    assert not command.struck.any()
    assert not command.scored.any()
    assert command.balance_failed.all()


def test_nonfoot_floor_contact_rejects_standing_and_failure_is_latched():
    env, command = make_env()
    assert mdp.football_standing(env).all()
    env.scene.sensors['football_body_ground'].data.found[:] = True
    for _ in range(7):
        command.update_progress()
        env.common_step_counter += 1
    env.scene.sensors['football_body_ground'].data.found[:] = False
    command.update_progress()
    assert command.balance_failed.all()
    command.reset_episode(torch.tensor([1]), command.ball_start[1:2].clone(),
                          env.scene['robot'].data.root_link_pos_w[1:2, :2], mdp._ball_kick_dir(env)[1:2])
    assert command.balance_failed.tolist() == [True, False, True, True]
    assert command.bad_posture_time[1] == 0


def test_exploration_is_bounded_without_altering_deterministic_actions():
    distribution = mdp.FootballGaussianDistribution(14, init_std=12.)
    actions = torch.randn(2, 14)
    distribution.update(actions)
    assert torch.allclose(distribution._distribution.scale, torch.full_like(actions, 0.3))
    assert torch.equal(distribution.deterministic_output(actions), actions)
    assert torch.equal(distribution.as_deterministic_output_module()(actions), actions)
