"""Keeper-focused motor fine-tuning on the 3v3 pitch.

The 2026-10-04 controlled battery scored 44/64 against an empty goal, 6/64
against the learned keeper, and 2/64 in 3v3. Train the existing 61D motor
interface before asking tactical MAPPO to compose it. Preserve the v3 bounded
shot/goal rewards and mature locomotion randomization. Both attacker starting
sides are sampled; 20% empty-goal rounds retain the original finishing skill.

Only the opponent may swap to frozen recovery during PPO. The learner always
executes its sampled action; deployment recovery is assessed in the 3v3 battery.
"""
from copy import deepcopy

from . import mdp
from .microduck_arena_env_cfg import make_microduck_arena_v3_env_cfg, add_arena_boards, MicroduckArenaV3RlCfg

ENABLE_BOARDS = True
ENABLE_SYMMETRY = False
PITCH_HALF_LENGTH = 1.5
PITCH_HALF_WIDTH = 1.0
ROUND_SECONDS = 20.0
MODE_PROBS = (.2, .8, 0.)
REWARD_RECIPE = 'keeper-finishing-v1'


def make_microduck_finishing_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_arena_v3_env_cfg(play=play, rough=rough)
    cfg.commands['twist'] = mdp.ArenaFinishingCommandCfg(
        resampling_time_range=(21.,21.), aim='open',
        pitch_half_length=PITCH_HALF_LENGTH, pitch_half_width=PITCH_HALF_WIDTH)
    cfg.episode_length_s = ROUND_SECONDS
    cfg.scene.env_spacing = 5.
    cfg.actions['joint_pos'] = mdp.ArenaFinishingJointPositionActionCfg(
        entity_name='robot',actuator_names=('^(?!passive_).*',),scale=1.,
        keeper_max_speed=(.2,.3,.15,.75))
    cfg.events['reset_arena'].func = mdp.reset_arena_finishing
    cfg.events['reset_arena'].params = {'mode_stages':[{'step':0,'probs':MODE_PROBS}]}
    # Boards and full 20 s match rules match the evaluation arena. A bounce is live.
    cfg.terminations.pop('arena_ball_out')
    cfg.rewards['arena_timeout'].params['include_ball_out'] = False
    if ENABLE_BOARDS:
        add_arena_boards(cfg)
    return cfg


MicroduckFinishingRlCfg = deepcopy(MicroduckArenaV3RlCfg)
MicroduckFinishingRlCfg.experiment_name = 'football_finishing'
MicroduckFinishingRlCfg.run_name = 'keeper-finishing'
MicroduckFinishingRlCfg.max_iterations = 1000


def make_microduck_return_to_play_env_cfg(play: bool = False, rough: bool = False):
    """Reverse curriculum at the recovery/football boundary, without overriding PPO."""
    from mjlab.managers import TerminationTermCfg
    cfg=make_microduck_finishing_env_cfg(play=play,rough=rough)
    cfg.episode_length_s=8.
    cfg.events['reset_arena'].func=mdp.reset_arena_return_to_play
    cfg.events['reset_arena'].params={
        'mode_stages':[{'step':0,'probs':(.8,.2,0.)}], 'start_probability':.5}
    cfg.actions['joint_pos']=mdp.ArenaReturnToPlayActionCfg(**vars(cfg.actions['joint_pos']))
    # A failed handoff is a terminal event; preserve the once-per-fall v3 cost.
    cfg.terminations['return_to_play_failed']=TerminationTermCfg(func=mdp.arena_return_to_play_failed)
    return cfg


MicroduckReturnToPlayRlCfg=deepcopy(MicroduckFinishingRlCfg)
MicroduckReturnToPlayRlCfg.experiment_name='football_return_to_play'
MicroduckReturnToPlayRlCfg.run_name='recovery-exit'
RETURN_TO_PLAY_RECIPE='recovery-exit-v1'
