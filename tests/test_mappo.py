"""MAPPO credit assignment, team symmetry, larger geometry and bounded motor interface."""
from types import SimpleNamespace as NS
import numpy as np
import torch
from mjlab_microduck.mappo import TacticalMAPPO,generalized_advantages,update
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_arena_teams_env_cfg import make_microduck_arena_teams_env_cfg
from mjlab_microduck.tasks.microduck_arena_env_cfg import arena_boards
from test_arena_teams_cfg import navigation_env
from test_arena_rewards import arena


def test_larger_pitch_boards_match_viewer_and_training():
    from mjlab_microduck.sim.football_arena_gpu import playback_cfg
    cfg=make_microduck_arena_teams_env_cfg()
    assert (cfg.commands['twist'].pitch_half_length,cfg.commands['twist'].pitch_half_width)==(1.5,1.)
    assert cfg.episode_length_s==20 and cfg.scene.env_spacing>=5
    training=cfg.scene.entities['boards'].spec_fn().compile()
    viewer=playback_cfg(teams=True).scene.entities['pitch'].spec_fn().compile()
    for i in range(6):
        np.testing.assert_allclose(training.geom(f'board_{i}').pos,viewer.geom(f'board_{i}').pos)
        np.testing.assert_allclose(training.geom(f'board_{i}').size,viewer.geom(f'board_{i}').size)
    assert arena_boards(1.5,1.)[0][1]>1.


def test_goal_crossing_uses_new_line_and_whole_ball():
    env,term=arena();term.cfg.pitch_half_length=1.5;term.cfg.track_training_signals=False
    term.previous_ball[:,0]=torch.tensor([.9,1.49,-1.49,1.49])
    env.scene['ball'].data.root_link_pos_w[:,0]=torch.tensor([.95,1.54,-1.54,1.52])
    term.update_match()
    torch.testing.assert_close(env.arena_goal_result,torch.tensor([0.,1.,-1.,0.]))


def test_keeper_tracks_ball_in_extended_play_area():
    command=mdp.arena_keeper_command(torch.tensor([[-1.3,0.]]),torch.zeros(1),
        torch.tensor([[1.,.8]]),torch.tensor([[-1.5,0.]]),-1,half_length=1.5,half_width=1.)
    assert command.abs().sum()>0


def test_tactical_commands_preserve_tracker_idle_and_speed_bounds():
    env=navigation_env()
    for skill in range(5):
        commands=mdp.arena_tactical_commands(env,torch.full((2,2),skill))
        assert set(commands)==set(mdp.MAPPO_BLUE)
        for name,value in commands.items():
            assert value.shape==(2,3) and torch.isfinite(value).all()
            assert value[:,0].abs().max()<=.35
            assert value[:,1].abs().max()<=.2
            assert value[:,2].abs().max()<=1
            if skill==0:
                expected,_=mdp.arena_team_navigation(env,name,torch.zeros(2,dtype=torch.long))
                torch.testing.assert_close(value,expected)
            if skill==4: assert not value.any()


def test_observation_contains_teammate_and_role_and_centralized_state():
    env=navigation_env()
    env.episode_length_buf=torch.zeros(2);env.max_episode_length=1000
    for name in mdp.ARENA_TEAM_PLAYERS:
        d=env.scene[name].data
        d.root_link_lin_vel_w=torch.zeros(2,3);d.root_link_lin_vel_b=torch.zeros(2,3)
        d.root_link_ang_vel_b=torch.zeros(2,3);d.projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(2,1)
    env.scene['ball'].data.root_link_lin_vel_w=torch.zeros(2,3)
    obs,state=mdp.arena_tactical_observations(env)
    assert obs.shape==(2,2,41) and state.shape==(2,2,87)
    assert torch.isfinite(obs).all() and torch.isfinite(state).all()
    before=obs.clone()
    env.scene['blue_attacker'].data.root_link_pos_w[:,1]+=.1
    changed,_=mdp.arena_tactical_observations(env)
    assert not torch.equal(before[:,0],changed[:,0])
    assert torch.equal(state[:,0,-2:],torch.tensor([[1.,0.]]).repeat(2,1))
    assert torch.equal(state[:,1,-2:],torch.tensor([[0.,1.]]).repeat(2,1))


def test_gae_terminal_cuts_off_next_episode_and_preserves_agent_axis():
    rewards=torch.tensor([[[1.,2.]],[[100.,200.]]])
    values=torch.zeros_like(rewards)
    done=torch.tensor([[True],[False]])
    advantages,returns=generalized_advantages(rewards,values,done,torch.full((2,1),.9),torch.tensor([[10.,20.]]),lam=1.)
    torch.testing.assert_close(advantages[0],rewards[0])
    torch.testing.assert_close(returns[1],torch.tensor([[109.,218.]]))


def test_discounted_potential_cannot_reward_a_closed_loop():
    gamma=.997
    phi=torch.tensor([0.,.5,-.2,0.])
    shaping=gamma*phi[1:]-phi[:-1]
    discounted=sum(gamma**i*shaping[i] for i in range(3))
    assert abs(float(discounted))<1e-7


def test_team_reward_treats_both_attackers_equally_and_costs_are_negative():
    env,term=arena();term.cfg.track_training_signals=False
    from copy import deepcopy
    env.scene['blue_attacker']=deepcopy(env.scene['robot'])
    env.scene['blue_attacker'].data.root_link_pos_w[:,1]=.5
    env.reset_buf=torch.zeros(4,dtype=torch.bool)
    env.arena_mappo_previous_potential=torch.full((4,),1/3)
    mdp.arena_mappo_team_reward(env)
    env.scene['robot'].data.root_link_pos_w[:,2]=.05
    a=mdp.arena_mappo_team_reward(env)
    env.scene['robot'].data.root_link_pos_w[:,2]=.12
    env.scene['blue_attacker'].data.root_link_pos_w[:,2]=.05
    b=mdp.arena_mappo_team_reward(env)
    torch.testing.assert_close(a,b)
    assert (env.arena_mappo_diagnostics['fall_cost']<=0).all()
    assert (env.arena_mappo_diagnostics['crowding_cost']<=0).all()


def test_ppo_learns_both_agent_samples_and_updates_centralized_critic():
    torch.manual_seed(42)
    model=TacticalMAPPO(4,7);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    obs=torch.randn(4,3,2,4);state=torch.randn(4,3,2,7)
    with torch.no_grad():
        dist=model.distribution(obs);actions=dist.sample();logp=dist.log_prob(actions);values=model.value(state)
    before={k:v.clone() for k,v in model.state_dict().items()}
    metrics=update(model,opt,(obs,state,actions,logp,values,torch.randn_like(values),values+1),epochs=2)
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before['actor.4.weight'],model.state_dict()['actor.4.weight'])
    assert not torch.equal(before['critic.4.weight'],model.state_dict()['critic.4.weight'])


def test_initial_prior_explores_without_excessive_switching_and_taxes_are_small():
    model=TacticalMAPPO(41,87)
    probs=model.distribution(torch.zeros(1,41)).probs[0]
    assert .90 < probs[0] < .95
    assert (probs[1:]>.01).all()
    assert mdp.MAPPO_FALL_COST_PER_SECOND*20 <= 1
    assert mdp.MAPPO_CROWDING_COST_PER_SECOND*20 <= .2


def test_corrupt_transition_cuts_gae_trace_without_poisoning_healthy_arenas():
    rewards=torch.ones(3,2,2); rewards[1,0]=float('nan')
    values=torch.full_like(rewards,2.)
    dones=torch.zeros(3,2,dtype=torch.bool); dones[1,0]=True
    valid=torch.ones_like(dones); valid[1,0]=False
    advantages,returns=generalized_advantages(rewards,values,dones,
        torch.full((3,2),.9),torch.full((2,2),2.),lam=1.,valid=valid)
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()
    torch.testing.assert_close(advantages[1,0],torch.zeros(2))
    # Bootstrap at the last valid pre-failure state, without the following episode.
    torch.testing.assert_close(advantages[0,0],torch.full((2,),.8))
    assert (advantages[0,1]>advantages[0,0]).all()


def test_ppo_completely_excludes_invalid_rows_even_with_nonfinite_inputs():
    from copy import deepcopy
    torch.manual_seed(7)
    a=TacticalMAPPO(4,7); b=deepcopy(a)
    obs=torch.randn(2,3,2,4); state=torch.randn(2,3,2,7)
    with torch.no_grad():
        dist=a.distribution(obs); actions=dist.sample(); logp=dist.log_prob(actions); values=a.value(state)
    batch=(obs,state,actions,logp,values,torch.randn_like(values),values+1)
    corrupted=tuple(x.clone() for x in batch)
    for x in corrupted:
        x[0,0]=999 if not x.is_floating_point() else float('nan')
    valid=torch.ones(2,3,dtype=torch.bool);valid[0,0]=False
    for model,data in ((a,batch),(b,corrupted)):
        torch.manual_seed(11)
        update(model,torch.optim.Adam(model.parameters(),lr=1e-3),data,valid=valid)
    for x,y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_numerical_guard_saves_sensor_failure_before_reset(tmp_path):
    import json
    env,_=arena(); d=env.scene['robot'].data
    d.joint_pos=torch.zeros(4,14);d.joint_vel=torch.zeros(4,14)
    d.root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(4,1)
    d.root_link_ang_vel_w=torch.zeros(4,3)
    sensor=env.scene.sensors['kick_ball_contact'].data
    sensor.force=torch.zeros(4,1,3);sensor.force[2,0,1]=float('inf')
    env.arena_numerical_report_dir=tmp_path
    env.sim=NS(data=NS(qpos=torch.zeros(4,20),qvel=torch.zeros(4,19)))
    bad=mdp.arena_numerical_guard(env,sensor_names=('kick_ball_contact',))
    assert bad.tolist()==[False,False,True,False]
    report=json.loads((tmp_path/'events.jsonl').read_text())
    assert report['env_ids']==[2]
    assert report['nonfinite_fields']=={'sensor/kick_ball_contact':1}
    snapshot=torch.load(tmp_path/'step_0_robot.pt',weights_only=False)
    assert torch.isinf(snapshot['tensors']['sensor/kick_ball_contact']).any()


def recovery_env():
    env=navigation_env(); env.step_dt=.02
    for name in mdp.ARENA_TEAM_PLAYERS:
        d=env.scene[name].data
        d.projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(2,1)
        d.root_link_ang_vel_b=torch.zeros(2,3)
    observed=[]
    def expert(obs):
        observed.append(obs['actor'].clone())
        return torch.full((2,14),7.)
    env.arena_recovery_policy=expert
    env.arena_recovery=mdp.ArenaRecovery(env)
    return env,observed


def test_recovery_hysteresis_and_stable_handoff():
    env,_=recovery_env(); r=env.arena_recovery
    obs=torch.ones(2,61); football=torch.ones(2,14)
    d=env.scene['robot'].data
    d.projected_gravity_b[0,2]=0.
    for _ in range(7):
        torch.testing.assert_close(r.select('robot',obs,football),football)
    for _ in range(2): output=r.select('robot',obs,football)
    assert r.active[:,0].tolist()==[True,False]
    assert (output[0]==7).all() and (output[1]==1).all()
    # A momentary upright state must not hand off to the football actor.
    d.projected_gravity_b[0,2]=-1.
    for _ in range(10): r.select('robot',obs,football)
    assert r.active[0,0]
    d.projected_gravity_b[0,2]=0.
    r.select('robot',obs,football)
    assert r.stable_time[0,0]==0
    d.projected_gravity_b[0,2]=-1.
    for _ in range(20): output=r.select('robot',obs,football)
    assert not r.active.any() and r.succeeded[0,0]
    torch.testing.assert_close(output,football)


def test_recovery_preserves_proprioception_and_uses_zero_command():
    env,observed=recovery_env()
    obs=torch.arange(122.).reshape(2,61)
    env.arena_recovery.select('blue_attacker',obs,torch.zeros(2,14))
    torch.testing.assert_close(observed[-1][:,:48],obs[:,:48])
    assert not observed[-1][:,48:].any()
    assert obs[:,48:].any()  # Input is never mutated.


def test_recovery_reset_isolated_and_fast_spin_cannot_exit():
    env,_=recovery_env(); r=env.arena_recovery
    r.active[:]=True; r.fallen_time[:]=1.; r.stable_time[:]=1.
    r.reset(torch.tensor([0]))
    assert not r.active[0].any() and r.active[1].all()
    assert r.fallen_time[0].sum()==0 and r.fallen_time[1].sum()==6
    env.scene['robot'].data.root_link_ang_vel_b[1,0]=5.
    for _ in range(30): r.select('robot',torch.zeros(2,61),torch.zeros(2,14))
    assert r.active[1,0] and r.stable_time[1,0]==0
    r.reset()
    assert not r.active.any()


def test_recovery_disabled_preserves_existing_actions():
    action=torch.randn(2,14)
    assert mdp.arena_recovery_action(navigation_env(),'robot',torch.zeros(2,61),action) is action
