"""Contract and task behavior for isolated handoff practice."""
from types import SimpleNamespace as NS
import torch
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_recovery_walk_env_cfg import make_microduck_recovery_walk_env_cfg


def test_isolated_task_preserves_physics_and_safety_contract():
    c = make_microduck_recovery_walk_env_cfg()
    assert set(c.scene.entities) == {'robot'}
    assert c.decimation*c.sim.mujoco.timestep == .02
    assert isinstance(c.actions['joint_pos'],mdp.RecoveryWalkActionCfg)
    assert c.actions['joint_pos'].scale == 1.
    assert 'expand_bam_friction_fields' in c.events
    assert 'nan_state' in c.terminations
    assert 'push_robot' not in c.events
    assert set(c.curriculum) <= {'com_range','head_com_range'}
    assert all(t.params['range_stages'][0]['step'] == 0 for t in c.curriculum.values())
    assert c.rewards['action_rate_l2'].weight < 0
    assert c.events['recovery_start'].params['start_probability'] == .75
    assert list(c.events)[-1] == 'recovery_start'
    assert not any('arena' in n or 'football' in n for n in c.rewards)
    assert len(c.commands['head_pose'].ranges) == 4
    assert len(c.commands['body_pose'].ranges) == 6


def test_stop_restart_and_fixed_evaluation_commands():
    term = object.__new__(mdp.RecoveryWalkCommand)
    term._env = NS(episode_length_buf=torch.tensor([0,149,150,199,200,299]),step_dt=.02)
    term.motion = torch.tensor([[.2,0,0]]*6)
    term.vel_command_b = torch.zeros(6,3); term.vel_command_w = torch.zeros(6,3)
    term._update_command()
    torch.testing.assert_close(term.command[:,0],torch.tensor([.2,.2,0,0,.2,.2]))
    term._env.recovery_walk_eval_command = torch.tensor([[0,0,.5]]*6)
    term._update_command()
    torch.testing.assert_close(term.command[:,2],torch.tensor([.5,.5,0,0,.5,.5]))
    assert not term.command[:,:2].any()


def test_action_history_restores_bank_without_touching_normal_spawns(monkeypatch):
    action = object.__new__(mdp.RecoveryWalkAction)
    action._raw_actions = torch.zeros(3,14)
    manager = NS(action=torch.zeros(3,14),prev_action=torch.zeros(3,14))
    action._env = NS(num_envs=3,device='cpu',action_manager=manager,
        arena_return_start_indices=torch.tensor([0,-1,0]),
        arena_return_starts={'last_action':torch.full((1,14),.3)})
    monkeypatch.setattr(mdp.JointPositionAction,'reset',lambda self,ids:None)
    action.reset(torch.tensor([0,1]))
    torch.testing.assert_close(manager.action[0],torch.full((14,),.3))
    assert not manager.action[1:].any()
    torch.testing.assert_close(manager.action,manager.prev_action)


def test_real_model_and_actor_match_source_contract():
    from mjlab_microduck.tasks.microduck_arena_env_cfg import make_microduck_arena_v3_env_cfg
    c = make_microduck_recovery_walk_env_cfg()
    source = make_microduck_arena_v3_env_cfg()
    assert list(c.observations['actor'].terms) == list(source.observations['actor'].terms)
    model = c.scene.entities['robot'].spec_fn().compile()
    assert model.nu == 14
    assert all(not model.joint(int(j)).name.startswith('passive_') for j in model.actuator_trnid[:,0])


def test_reset_bank_mapping_is_exact_and_scoped(monkeypatch):
    from test_finishing_cfg import finishing_env
    env,_ = finishing_env()
    robot = env.scene['robot']
    monkeypatch.setattr(mdp,'_servo_joint_ids',lambda e,a:list(range(14)))
    calls=[]
    robot.write_joint_state_to_sim=lambda p,v,joint_ids,env_ids:calls.append((p,v,joint_ids,env_ids))
    env.arena_return_starts={'height':torch.tensor([.11,.12]),'quat':torch.tensor([[1.,0,0,0]]*2),
        'velocity':torch.zeros(2,6),'joint_pos':torch.full((2,14),.2),
        'joint_vel':torch.zeros(2,14),'last_action':torch.full((2,14),.3)}
    env.recovery_walk_bank_indices=torch.tensor([1,0,0,1])
    mdp.reset_recovery_walk(env,torch.tensor([0,2]),start_probability=1.)
    assert env.arena_return_start_indices.tolist()==[1,-1,0,-1]
    torch.testing.assert_close(robot.data.root_link_pos_w[:,2],torch.tensor([.12,0,.11,0]))
    assert calls[-1][2]==list(range(14))
    mdp.reset_recovery_walk(env,torch.tensor([0]),start_probability=0.)
    assert env.arena_return_start_indices.tolist()==[-1,-1,0,-1]


def tracking_env():
    d=NS(root_link_lin_vel_b=torch.zeros(3,3),root_link_ang_vel_b=torch.zeros(3,3))
    command=torch.tensor([[.1,0,0],[0,.1,0],[0,0,.5]])
    env=NS(num_envs=3,device='cpu',step_dt=.02,scene={'robot':NS(data=d)},
           command_manager=NS(get_command=lambda _:command))
    return env,d,command


def test_tracking_rewards_movement_and_ignores_uncommanded_axes():
    env,d,command=tracking_env()
    linear=mdp.RecoveryWalkTracking(None,env)
    yaw=mdp.RecoveryWalkTracking(None,env)
    idle=linear(env,'linear',.1,tau_s=0)
    assert torch.all(idle[:2]<.38)
    assert yaw(env,'angular',.25,tau_s=0)[2]<.02
    d.root_link_lin_vel_b[:,:2]=command[:,:2]
    d.root_link_lin_vel_b[:,2]=3.  # Body bobbing is not planar tracking error.
    d.root_link_ang_vel_b[:,:2]=4.
    d.root_link_ang_vel_b[:,2]=command[:,2]
    torch.testing.assert_close(linear(env,'linear',.1,tau_s=0),torch.ones(3))
    torch.testing.assert_close(yaw(env,'angular',.25,tau_s=0),torch.ones(3))
    command[:]=0  # Stop must punish continued planar motion.
    assert torch.all(linear(env,'linear',.1,tau_s=0)[:2]<.38)


def test_tracking_history_is_reset_per_environment_and_nan_safe():
    env,d,_=tracking_env();term=mdp.RecoveryWalkTracking(None,env)
    d.root_link_lin_vel_b[:]=1
    term(env,'linear',.1)
    d.root_link_lin_vel_b[:]=0
    term.reset(torch.tensor([1]))
    term(env,'linear',.1)
    assert not term.velocity[1].any()
    assert term.velocity[0,0]>.8 and term.velocity[2,0]>.8
    d.root_link_lin_vel_b[1]=float('nan')
    assert torch.isfinite(term(env,'linear',.1)).all()


def test_movement_weights_dominate_posture_support():
    c=make_microduck_recovery_walk_env_cfg()
    assert c.rewards['track_linear_velocity'].func is mdp.RecoveryWalkTracking
    assert c.rewards['track_linear_velocity'].weight==4.
    assert sum(c.rewards[k].weight for k in ['upright','pose','head_pose_tracking'])<1.31


def test_directional_progress_is_bounded_wrong_way_negative_and_idle_zero():
    env,d,command=tracking_env();term=mdp.RecoveryWalkTracking(None,env)
    assert not term(env,'linear',1.,tau_s=0,progress=True).any()
    d.root_link_lin_vel_b[:,:2]=command[:,:2]*10
    torch.testing.assert_close(term(env,'linear',1.,tau_s=0,progress=True),torch.tensor([1.,1.,0.]))
    d.root_link_lin_vel_b[:,:2]*=-1
    torch.testing.assert_close(term(env,'linear',1.,tau_s=0,progress=True),torch.tensor([-1.,-1.,0.]))
    d.root_link_ang_vel_b[:,2]=command[:,2]*3
    torch.testing.assert_close(term(env,'angular',1.,tau_s=0,progress=True),torch.tensor([0.,0.,1.]))
    command[:]=0
    assert not term(env,'linear',1.,tau_s=0,progress=True).any()


def test_v3_practice_mix_and_longer_training_stop_keep_eval_fixed():
    c=make_microduck_recovery_walk_env_cfg();cfg=c.commands['twist']
    p=cfg.bucket_probabilities
    assert abs(sum(p)-1.)<1e-8
    assert abs(sum(p[i] for i in [4,6,7,8])-.65)<1e-8
    assert sum(p[1:4])>=.15-1e-8 and p[5]>=.1 and p[0]>=.1
    assert c.episode_length_s==8 and cfg.stop_end_s==5
    term=object.__new__(mdp.RecoveryWalkCommand);term.cfg=cfg
    term._env=NS(episode_length_buf=torch.tensor([149,150,199,200,249,250]),step_dt=.02)
    term.motion=torch.tensor([[.2,0,0]]*6)
    term.vel_command_b=torch.zeros(6,3);term.vel_command_w=torch.zeros(6,3)
    term._update_command()
    torch.testing.assert_close(term.command[:,0],torch.tensor([.2,0,0,0,0,.2]))
    term._env.recovery_walk_eval_command=term.motion
    term._update_command()
    torch.testing.assert_close(term.command[:,0],torch.tensor([.2,0,0,.2,.2,.2]))


def test_sampled_commands_match_the_declared_mix():
    c=make_microduck_recovery_walk_env_cfg();n=20000
    term=object.__new__(mdp.RecoveryWalkCommand)
    term.cfg=c.commands['twist'];term._env=NS(num_envs=n,device='cpu')
    term.motion=torch.zeros(n,3);term.vel_command_b=torch.zeros(n,3);term.vel_command_w=torch.zeros(n,3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17);term._resample_command(torch.arange(n))
    weak=(term.motion[:,0]<0)|(term.motion[:,1]<0)|(term.motion[:,2]!=0)
    assert abs(float(weak.float().mean())-.65)<.02
    assert abs(float((term.motion[:,0]>0).float().mean())-.15)<.02
    torch.testing.assert_close(term.motion,term.command)
