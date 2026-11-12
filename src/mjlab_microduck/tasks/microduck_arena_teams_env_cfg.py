"""3v3 football: two attackers and a goal-line defender per team.

One blue attacker is trained; the other five players use frozen policies.
The actor stays 61D/14 actions; the critic sees all five other players.
Use train_arena.py with --keeper-policy to initialize the defenders.
"""
from copy import deepcopy

from mjlab.managers import EventTermCfg, ObservationTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import mdp
from .microduck_arena_env_cfg import (
    MicroduckArenaRlCfg, add_arena_boards, make_microduck_arena_env_cfg,
)

ENABLE_SYMMETRY = False
ENABLE_BOARDS = True
PITCH_HALF_LENGTH = 1.5
PITCH_HALF_WIDTH = 1.0
ROUND_SECONDS = 20.0
REWARD_RECIPE = 'three-a-side-v1'


def make_microduck_arena_teams_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_arena_env_cfg(play=play,rough=rough)
    for name in mdp.ARENA_TEAM_PLAYERS:
        if name not in cfg.scene.entities:
            cfg.scene.entities[name] = deepcopy(cfg.scene.entities['robot'])
        if name not in ('robot','opponent'):
            cfg.terminations[f'{name}_nan'] = TerminationTermCfg(
                func=mdp.robot_state_is_nan,params={'asset_cfg':SceneEntityCfg(name)})
    cfg.sim.nconmax = 750
    cfg.commands['twist'] = mdp.ArenaTeamCommandCfg(
        resampling_time_range=(21.,21.),aim='open',
        pitch_half_length=PITCH_HALF_LENGTH,pitch_half_width=PITCH_HALF_WIDTH)
    cfg.episode_length_s = ROUND_SECONDS
    cfg.scene.env_spacing = 5.0
    if ENABLE_BOARDS:
        add_arena_boards(cfg)
    cfg.actions['joint_pos'] = mdp.ArenaTeamJointPositionActionCfg(
        entity_name='robot',actuator_names=('^(?!passive_).*',),scale=1.)
    cfg.events['reset_arena'] = EventTermCfg(func=mdp.reset_arena_teams,mode='reset')
    cfg.observations['critic'].terms.pop('opponent_state')
    cfg.observations['critic'].terms['team_state'] = ObservationTermCfg(func=mdp.arena_team_state)
    return cfg


MicroduckArenaTeamsRlCfg = deepcopy(MicroduckArenaRlCfg)
MicroduckArenaTeamsRlCfg.experiment_name = 'football_arena_3v3'
MicroduckArenaTeamsRlCfg.run_name = 'three-a-side'
