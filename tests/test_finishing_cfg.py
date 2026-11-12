"""Finishing transfer: actual pitch geometry, reset isolation, PPO and keeper contracts."""
from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path

import mujoco
import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_finishing_env_cfg import (
    make_microduck_finishing_env_cfg, MicroduckFinishingRlCfg, MODE_PROBS)
from mjlab_microduck.tasks.microduck_arena_env_cfg import make_microduck_arena_v3_env_cfg
from test_arena_rewards import arena


@pytest.mark.parametrize('play', (False,True))
def test_recipe_preserves_motor_contract_and_mature_stack(play):
    cfg = make_microduck_finishing_env_cfg(play=play)
    base = make_microduck_arena_v3_env_cfg(play=play)
    assert list(cfg.observations['actor'].terms) == list(base.observations['actor'].terms)
    assert cfg.decimation*cfg.sim.mujoco.timestep == .02
    assert cfg.episode_length_s == 20
    assert (cfg.commands['twist'].pitch_half_length,cfg.commands['twist'].pitch_half_width)==(1.5,1.)
    assert cfg.actions['joint_pos'].keeper_max_speed == (.2,.3,.15,.75)
    assert cfg.actions['joint_pos'].actuator_names == ('^(?!passive_).*',)
    assert 'expand_bam_friction_fields' in cfg.events
    assert list(cfg.events)[-1] == 'reset_arena'
    assert cfg.events['reset_arena'].func is mdp.reset_arena_finishing
    assert cfg.events['reset_arena'].params['mode_stages'] == [{'step':0,'probs':MODE_PROBS}]
    assert MODE_PROBS == (.2,.8,0.)
    assert 'arena_ball_out' not in cfg.terminations
    assert not cfg.rewards['arena_timeout'].params['include_ball_out']
    assert cfg.rewards['arena_goal_delta'].weight == 100
    assert cfg.rewards['arena_fall_event'].weight < 0
    assert MicroduckFinishingRlCfg.actor.obs_normalization
    assert MicroduckFinishingRlCfg.algorithm.symmetry_cfg is None
    for key in ('com_range','head_com_range'):
        if key in cfg.curriculum:
            assert len(cfg.curriculum[key].params['range_stages']) == 1
            assert cfg.curriculum[key].params['range_stages'][0]['step'] == 0


def test_real_model_resolves_fourteen_servo_joints_and_board_lines():
    cfg = make_microduck_finishing_env_cfg()
    model = cfg.scene.entities['robot'].spec_fn().compile()
    assert model.nu == 14
    assert all(not model.joint(int(j)).name.startswith('passive_') for j in model.actuator_trnid[:,0])
    boards = cfg.scene.entities['boards'].spec_fn().compile()
    assert boards.geom('board_0').pos[1] == pytest.approx(1.015)
    assert boards.geom('board_2').pos[0] == pytest.approx(1.54)


class Entity:
    def __init__(self,n):
        self.data = NS(root_link_pos_w=torch.zeros(n,3),root_link_quat_w=torch.tensor([[1.,0,0,0]]).repeat(n,1),
            default_joint_pos=torch.zeros(n,14),projected_gravity_b=torch.tensor([[0.,0,-1.]]).repeat(n,1))
    def write_root_link_pose_to_sim(self,pose,ids):
        self.data.root_link_pos_w[ids] = pose[:,:3]
        self.data.root_link_quat_w[ids] = pose[:,3:]
    write_mocap_pose_to_sim = write_root_link_pose_to_sim
    def write_root_link_velocity_to_sim(self,velocity,ids):
        pass
    def write_joint_state_to_sim(self,position,velocity,env_ids):
        pass


def finishing_env():
    env,_ = arena()
    for name in ('robot','opponent','ball','goal','own_goal','boards'):
        env.scene[name] = Entity(env.num_envs)
    env.scene.entities = env.scene
    env.scene.terrain.env_origins[:] = torch.tensor([[0.,0,0],[5.,0,0],[0.,5,0],[5.,5,0]])
    term = mdp.ArenaFinishingCommandCfg(resampling_time_range=(21.,21.),aim='open',
        pitch_half_length=1.5,pitch_half_width=1.).build(env)
    env.command_manager = NS(get_term=lambda _:term)
    env.action_manager = NS(get_term=lambda _:NS(cfg=mdp.ArenaFinishingJointPositionActionCfg(
        entity_name='robot',actuator_names=('^(?!passive_).*',))))
    env.arena_recovery = mdp.ArenaRecovery(env)
    return env,term


def test_reset_isolated_and_relative_to_actual_pitch():
    env,term = finishing_env()
    ids = torch.tensor([0,2])
    env.arena_recovery.active[:] = True
    term.initial_distance[:] = 99
    mdp.reset_arena_finishing(env,ids,[{'step':0,'probs':(0.,1.,0.)}])
    origin = env.scene.terrain.env_origins[ids]
    robot = env.scene['robot'].data.root_link_pos_w[ids]-origin
    opponent = env.scene['opponent'].data.root_link_pos_w[ids]-origin
    torch.testing.assert_close(robot[:,0],torch.full((2,),-.55))
    torch.testing.assert_close(robot[:,1].abs(),torch.full((2,),.42))
    torch.testing.assert_close(opponent[:,:2],torch.tensor([[1.3,0.],[1.3,0.]]))
    torch.testing.assert_close(env.scene['goal'].data.root_link_pos_w[ids,0]-origin[:,0],torch.full((2,),1.5))
    ball = term.previous_ball[ids,:2]-origin[:,:2]
    expected = (mdp.arena_kick_stance(ball,term.aim_xy[ids])-robot[:,:2]).norm(dim=-1)
    torch.testing.assert_close(term.initial_distance[ids],expected)
    assert (term.aim_xy[ids,0] >= 1.5).all()
    assert (term.initial_distance[[1,3]] == 99).all()
    assert not env.arena_recovery.active[ids].any()
    assert env.arena_recovery.active[[1,3]].all()


def test_empty_goal_parks_keeper_and_aims_at_actual_goal_center():
    env,term = finishing_env()
    mdp.reset_arena_finishing(env,None,[{'step':0,'probs':(1.,0.,0.)}])
    term._update_command()
    torch.testing.assert_close(term.aim_xy,torch.tensor([[1.5,0.]]).repeat(4,1))
    origin = env.scene.terrain.env_origins
    assert (env.scene['opponent'].data.root_link_pos_w[:,1]-origin[:,1] > 2).all()


def test_recovery_hook_only_infers_opponent_and_does_not_override_learner():
    obs = torch.zeros(2,61)
    seen = []
    action = object.__new__(mdp.ArenaFinishingJointPositionAction)
    action._env = NS(arena_opponent_policy=lambda _:torch.ones(2,14),
        arena_recovery=NS(select=lambda name,observation,motor:seen.append(name) or motor*2))
    torch.testing.assert_close(action._infer_opponent(obs),torch.full((2,14),2.))
    assert seen == ['opponent']
    assert mdp.ArenaFinishingJointPositionAction.process_actions is mdp.ArenaJointPositionAction.process_actions


def test_fixed_evaluation_uses_keeper_checkpoint_not_baseline():
    path = Path(__file__).parents[1]/'scripts/train_arena.py'
    spec = importlib.util.spec_from_file_location('train_arena_finishing_test',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    policy = lambda value:lambda _:torch.full((3,14),value)
    mode = torch.tensor([[mdp.SOLO],[mdp.KEEPER],[mdp.ATTACKER]])
    result = module.mixed_opponent_action({},mode,policy(1.),policy(9.),policy(3.),torch.zeros(3,1,dtype=torch.bool))
    torch.testing.assert_close(result[:,0],torch.tensor([1.,3.,1.]))
