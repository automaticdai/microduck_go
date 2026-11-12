"""Three-a-side contracts, mirrored roles, isolated histories and reset placement."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import mujoco
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_arena_env_cfg import make_microduck_arena_env_cfg
from mjlab_microduck.tasks.microduck_arena_teams_env_cfg import (
    make_microduck_arena_teams_env_cfg, MicroduckArenaTeamsRlCfg,
)


def test_six_players_keep_policy_and_physics_contract():
    cfg = make_microduck_arena_teams_env_cfg()
    base = make_microduck_arena_env_cfg()
    assert Counter(mdp.ARENA_TEAM_PLAYERS.values()) == {
        (1,'attacker'):2,(1,'defender'):1,(-1,'attacker'):2,(-1,'defender'):1}
    assert set(cfg.scene.entities) == set(mdp.ARENA_TEAM_PLAYERS) | {'ball','goal','own_goal','boards'}
    assert list(cfg.observations['actor'].terms) == list(base.observations['actor'].terms)
    assert 'team_state' in cfg.observations['critic'].terms
    assert cfg.decimation*cfg.sim.mujoco.timestep == .02
    assert 'expand_bam_friction_fields' in cfg.events
    assert cfg.events['reset_arena'].func is mdp.reset_arena_teams
    assert cfg.rewards['arena_goal_delta'].weight > 0
    assert cfg.rewards['arena_fallen'].weight < 0
    assert MicroduckArenaTeamsRlCfg.actor.obs_normalization
    assert MicroduckArenaTeamsRlCfg.algorithm.symmetry_cfg is None
    for name in mdp.ARENA_TEAM_PLAYERS:
        model = cfg.scene.entities[name].spec_fn().compile()
        joints = [model.joint(j).name for j in range(model.njnt)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        assert len([j for j in joints if not j.startswith('passive_')]) == 14
    assert len({id(cfg.scene.entities[n]) for n in mdp.ARENA_TEAM_PLAYERS}) == 6


def test_mirrored_formations_have_clearance():
    spawns = mdp.ARENA_TEAM_SPAWNS
    for blue,orange in (('robot','opponent'),('blue_attacker','orange_attacker'),
                        ('blue_defender','orange_defender')):
        np.testing.assert_allclose(spawns[blue],-np.array(spawns[orange]))
    xy = np.array(list(spawns.values()))
    distances = np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    np.fill_diagonal(distances,np.inf)
    assert distances.min() > .3


def test_compiled_world_has_84_independent_servo_targets():
    from mjlab.scene import Scene
    cfg = make_microduck_arena_teams_env_cfg(play=True)
    cfg.scene.num_envs = 1
    model = Scene(cfg.scene,device='cpu').compile()
    assert model.nu == 6*14
    targets = set()
    for name in mdp.ARENA_TEAM_PLAYERS:
        ids = [i for i in range(model.nu) if model.actuator(i).name.startswith(name+'/')]
        assert len(ids) == 14
        joints = set(model.actuator_trnid[ids,0])
        assert len(joints) == 14 and targets.isdisjoint(joints)
        targets.update(joints)


class Scene(dict):
    terrain = SimpleNamespace(env_origins=torch.zeros(2,3))


def navigation_env():
    scene = Scene()
    for name,(sign,_) in mdp.ARENA_TEAM_PLAYERS.items():
        xy = torch.tensor(mdp.ARENA_TEAM_SPAWNS[name]).repeat(2,1)
        quat = torch.tensor([1.,0.,0.,0.] if sign == 1 else [0.,0.,0.,1.]).repeat(2,1)
        scene[name] = SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.cat((xy,torch.full((2,1),.125)),dim=1),root_link_quat_w=quat))
    scene['ball'] = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.,0.,.035]]).repeat(2,1)))
    cfg = mdp.ArenaTeamJointPositionActionCfg(entity_name='robot',actuator_names=('^(?!passive_).*',))
    return SimpleNamespace(scene=scene,num_envs=2,device='cpu',
                           action_manager=SimpleNamespace(get_term=lambda _:SimpleNamespace(cfg=cfg)),
                           command_manager=SimpleNamespace(get_term=lambda _:SimpleNamespace(
                               cfg=mdp.ArenaTeamCommandCfg(resampling_time_range=(21.,21.),
                                   pitch_half_length=1.5,pitch_half_width=1.))))


def test_defenders_hold_own_goals_and_attackers_mirror():
    env = navigation_env()
    side = lambda:torch.zeros(2,dtype=torch.long)
    for name in ('blue_defender','orange_defender'):
        command,aim = mdp.arena_team_navigation(env,name,side())
        torch.testing.assert_close(command,torch.zeros(2,3))
        assert aim is None
    blue,blue_aim = mdp.arena_team_navigation(env,'robot',side())
    orange,orange_aim = mdp.arena_team_navigation(env,'opponent',side())
    torch.testing.assert_close(blue,orange)
    torch.testing.assert_close(blue_aim,-orange_aim)
    assert (blue_aim[:,0] > .9).all()


def test_partial_reset_clears_each_players_history_only_for_selected_world():
    action = object.__new__(mdp.ArenaTeamJointPositionAction)
    action._raw_actions = torch.ones(2,14)
    action._processed_actions = torch.ones(2,14)
    reset_ids = []
    action.players = {name:SimpleNamespace(reset=lambda ids:reset_ids.append(ids))
                      for name in mdp.ARENA_TEAM_PLAYERS if name != 'robot'}
    action.velocities = {name:torch.ones(2,14) for name in action.players}
    action.aim_sides = {name:torch.ones(2,dtype=torch.long) for name in action.players}
    action.reset(torch.tensor([0]))
    assert len(reset_ids) == 5
    for name in action.players:
        assert not action.velocities[name][0].any() and action.velocities[name][1].all()
        assert action.aim_sides[name].tolist() == [0,1]


def test_viewer_has_six_team_models_and_one_set_of_boards():
    from mjlab_microduck.sim.football_arena_gpu import playback_cfg
    cfg = playback_cfg(teams=True)
    assert 'boards' not in cfg.scene.entities
    assert cfg.commands['twist'].track_training_signals is False
    assert cfg.events['reset_arena'].params == {}
    assert all(name in cfg.scene.entities for name in mdp.ARENA_TEAM_PLAYERS)
    for name in mdp.ARENA_TEAM_PLAYERS:
        if name != 'robot':
            assert f'{name}_nan' in cfg.terminations
