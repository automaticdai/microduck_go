"""Recovered-pose resets preserve servo layout, reset isolation, and action history."""
from types import SimpleNamespace as NS
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_finishing_env_cfg import make_microduck_return_to_play_env_cfg
from test_finishing_cfg import finishing_env


def test_recipe_preserves_contract_and_penalty_signs():
    cfg=make_microduck_return_to_play_env_cfg()
    assert cfg.decimation*cfg.sim.mujoco.timestep==.02
    assert isinstance(cfg.actions['joint_pos'],mdp.ArenaReturnToPlayActionCfg)
    assert cfg.events['reset_arena'].params['start_probability']==.5
    assert cfg.events['reset_arena'].params['mode_stages'][0]['probs']==(.8,.2,0.)
    assert cfg.terminations['return_to_play_failed'].func is mdp.arena_return_to_play_failed
    assert cfg.rewards['arena_fall_event'].weight<0
    assert cfg.rewards['arena_goal_delta'].weight>0
    assert 'expand_bam_friction_fields' in cfg.events


def test_recovered_pose_resets_only_selected_worlds(monkeypatch):
    env,_=finishing_env();robot=env.scene['robot'];calls=[]
    # The base reset writes all servos too; accept and record the bank's explicit selector.
    robot.write_joint_state_to_sim=lambda p,v,env_ids,joint_ids=None:calls.append((p,v,env_ids,joint_ids))
    monkeypatch.setattr(mdp,'_servo_joint_ids',lambda e,a:list(range(14)))
    env.arena_return_starts={'height':torch.tensor([.11]),'quat':torch.tensor([[1.,0,0,0]]),
        'velocity':torch.zeros(1,6),'joint_pos':torch.full((1,14),.2),
        'joint_vel':torch.zeros(1,14),'last_action':torch.full((1,14),.3)}
    ids=torch.tensor([0,2]);before=robot.data.root_link_pos_w.clone()
    mdp.reset_arena_return_to_play(env,ids,[{'step':0,'probs':(1.,0.,0.)}],start_probability=1.)
    assert env.arena_return_start_indices.tolist()==[0,-1,0,-1]
    torch.testing.assert_close(robot.data.root_link_pos_w[ids,2],torch.full((2,),.11))
    torch.testing.assert_close(robot.data.root_link_pos_w[[1,3]],before[[1,3]])
    assert calls[-1][3]==list(range(14))
    torch.testing.assert_close(calls[-1][0],torch.full((2,14),.2))


def test_action_reset_restores_actual_previous_expert_action(monkeypatch):
    action=object.__new__(mdp.ArenaReturnToPlayAction)
    manager=NS(action=torch.zeros(4,14),prev_action=torch.zeros(4,14))
    action._env=NS(num_envs=4,device='cpu',action_manager=manager,
        arena_return_start_indices=torch.tensor([0,-1,0,-1]),
        arena_return_starts={'last_action':torch.full((1,14),.3)})
    action._raw_actions=torch.zeros(4,14)
    monkeypatch.setattr(mdp.ArenaFinishingJointPositionAction,'reset',lambda self,ids:None)
    action.reset(torch.tensor([0,1]))
    assert (action._raw_actions[0]==.3).all() and not action._raw_actions[1:].any()
    torch.testing.assert_close(manager.action,action._raw_actions)
    torch.testing.assert_close(manager.prev_action,action._raw_actions)
