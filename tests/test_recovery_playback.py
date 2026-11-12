"""Playback recovery must replace primary actions and clear latches on resets."""
from types import SimpleNamespace
import torch
from mjlab_microduck.sim.football_arena_gpu import MatchEnv, playback_cfg
from mjlab_microduck.tasks import mdp


def test_two_player_recovery_preserves_action_configuration():
    cfg=playback_cfg(teams=False)
    original=cfg.actions['joint_pos']
    recovery=mdp.ArenaFinishingJointPositionActionCfg(**vars(original))
    assert vars(recovery)==vars(original)
    assert recovery.build.__func__ is mdp.ArenaFinishingJointPositionActionCfg.build


def test_primary_playback_executes_recovery_and_clears_done(monkeypatch):
    from mjlab.rl import RslRlVecEnvWrapper
    calls=[];obs={'actor':torch.randn(1,61)}
    recovery=SimpleNamespace(reset=lambda ids:calls.append(('reset',ids.clone())))
    env=SimpleNamespace(arena_recovery=recovery,step_dt=.02,arena_goal_result=torch.zeros(1),
                        cfg=SimpleNamespace(episode_length_s=20.))
    # super() requires an instance of MatchEnv; its base env property delegates .unwrapped.
    match=object.__new__(MatchEnv)
    match.env=SimpleNamespace(unwrapped=env)
    match.cached_obs=obs;match.elapsed=0.;match.goal_this_round=False
    match.score=[0,0];match.round=1;match.last_event=''
    def select(e,name,observation,action):
        assert e is env and name=='robot' and observation is obs['actor']
        return torch.full_like(action,7.)
    monkeypatch.setattr('mjlab_microduck.sim.football_arena_gpu.arena_recovery_action',select)
    def step(self,action):
        assert (action==7).all()
        return obs,torch.zeros(1),torch.ones(1,dtype=torch.bool),{}
    monkeypatch.setattr(RslRlVecEnvWrapper,'step',step)
    match.step(torch.zeros(1,14))
    assert calls[0][0]=='reset' and calls[0][1].tolist()==[0]


def test_cpu_adapter_passes_real_history_and_zeroes_only_commands():
    import numpy as np
    from mjlab_microduck.sim.football_arena import CpuRecovery
    seen=[]
    session=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='actor')],
        run=lambda _,inputs:seen.append(inputs['actor'].copy()) or [np.full((1,14),7.,dtype=np.float32)])
    recovery=CpuRecovery(session,.02)
    obs=np.arange(61,dtype=np.float32);obs[:6]=[0,0,0,0,0,-1]
    football=np.ones(14,dtype=np.float32)
    for _ in range(9):out=recovery.select(1,obs,football,.03)
    np.testing.assert_array_equal(out,np.full(14,7.))
    np.testing.assert_array_equal(seen[-1][0,:48],obs[:48])
    assert not seen[-1][0,48:].any()
    assert recovery.labels()==['Football','Recovery']
    for _ in range(20):out=recovery.select(1,obs,football,.12)
    np.testing.assert_array_equal(out,football)
    assert recovery.labels()==['Football','Football']
    assert obs[48:].any()
