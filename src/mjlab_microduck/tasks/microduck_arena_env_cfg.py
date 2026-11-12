"""Microduck Go football: shared physical opponent, center-disk kickoffs, 10 s rounds.

Use train_arena.py: the opponent must be explicitly initialized from a policy.
The actor remains 61D; the critic additionally observes the opponent. No fall or
out-of-bounds termination: only goals, timeout, and numerical-safety failures.
"""
from copy import deepcopy
from functools import partial
import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .microduck_football_env_cfg import make_microduck_football_env_cfg, MicroduckFootballRlCfg
from . import mdp

ENABLE_SYMMETRY = False
EPISODE_LENGTH_S = 10.0
REWARD_RECIPE = 'goal-first-v2'
GOAL_REWARD = 100.0
FALL_COST_PER_SECOND = 5.0
TIMEOUT_COST = 5.0
# These bounded auxiliaries pay at most 8 points over a full ten-second round.
SUPPORT_REWARD_WEIGHTS = {
    'track_linear_velocity': .20, 'track_angular_velocity': .10,
    'upright': .30, 'pose': .10, 'head_pose_tracking': .10, 'air_time': 0.,
}


# Low boards around the pitch: side lines plus end lines beside each goal (goal + net close the
# mouth). (x, y, half-x, half-y) of each box; the ball stays in play instead of rolling away.
BOARD_HEIGHT = .08
def arena_boards(half_length=.9, half_width=.65):
    end = half_length + .04
    outer_y = half_width + .015
    segment = (outer_y - .21) / 2
    center = .21 + segment
    return ((0,outer_y,end+.005,.005),(0,-outer_y,end+.005,.005),
            (end,center,.005,segment),(end,-center,.005,segment),
            (-end,center,.005,segment),(-end,-center,.005,segment))


ARENA_BOARDS = arena_boards()


def get_arena_boards_spec(half_length=.9, half_width=.65):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='boards',mocap=True)  # Moved to each arena origin by reset_arena.
    for i,(x,y,sx,sy) in enumerate(arena_boards(half_length,half_width)):
        body.add_geom(name=f'board_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,y,BOARD_HEIGHT/2],
                      size=[sx,sy,BOARD_HEIGHT/2],rgba=[.95,.95,.95,.9],contype=1,conaffinity=1)
    return spec


def add_arena_boards(cfg):
    """Opt-in: collide the ball and ducks with pitch boards (changes the task; off by default)."""
    term = cfg.commands['twist']
    cfg.scene.entities['boards'] = EntityCfg(spec_fn=partial(
        get_arena_boards_spec,term.pitch_half_length,term.pitch_half_width))
    return cfg


def make_microduck_arena_env_cfg(play=False,rough=False):
    cfg = make_microduck_football_env_cfg(play=play,rough=rough)
    cfg.scene.entities['opponent'] = deepcopy(cfg.scene.entities['robot'])
    cfg.scene.entities['own_goal'] = deepcopy(cfg.scene.entities['goal'])
    cfg.scene.env_spacing = 4.0
    cfg.sim.nconmax = 250
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.is_finite_horizon = True
    cfg.commands['twist'] = mdp.ArenaCommandCfg(resampling_time_range=(11.,11.))
    cfg.actions['joint_pos'] = mdp.ArenaJointPositionActionCfg(
        entity_name='robot',actuator_names=('^(?!passive_).*',),scale=1.0)
    for name in list(cfg.rewards):
        if name.startswith('football_') or name == 'ball_speed_overshoot':
            del cfg.rewards[name]
    for name,weight in SUPPORT_REWARD_WEIGHTS.items():
        cfg.rewards[name].weight = weight
    for name,weight in (('approach_delta',5.),('ball_delta',15.),
                        ('strike_delta',3.),('shot_speed_delta',3.),('goal_delta',GOAL_REWARD)):
        cfg.rewards['arena_'+name] = RewardTermCfg(func=mdp.arena_progress,weight=weight,params={'component':name})
    cfg.rewards['arena_fallen'] = RewardTermCfg(func=mdp.arena_fallen_penalty,weight=-FALL_COST_PER_SECOND)
    cfg.rewards['arena_timeout'] = RewardTermCfg(func=mdp.arena_timeout_penalty,weight=-TIMEOUT_COST)
    cfg.rewards['action_rate_l2'].weight = -.02
    cfg.curriculum.pop('action_rate_weight',None)
    for name in list(cfg.curriculum):
        if name.startswith('football_'):
            del cfg.curriculum[name]
    # Inherited DR remains mature when transferring the trained actor.
    for name in ('com_range','head_com_range'):
        if name in cfg.curriculum:
            stages = cfg.curriculum[name].params['range_stages']
            cfg.curriculum[name].params['range_stages'] = [{**stages[-1],'step':0}]
    cfg.events.pop('reset_football')
    cfg.events.pop('push_robot',None)
    cfg.events['reset_arena'] = EventTermCfg(func=mdp.reset_arena,mode='reset')
    cfg.observations['critic'].terms.pop('football_state')
    cfg.observations['critic'].terms['opponent_state'] = ObservationTermCfg(func=mdp.arena_opponent_state)
    cfg.observations['critic'].terms['time_remaining'] = ObservationTermCfg(func=mdp.arena_time_remaining)
    cfg.terminations.pop('lost_balance',None)
    # Keep the template's numerical guards, but not locomotion fall endings.
    for name in list(cfg.terminations):
        if name not in ('time_out','nan_state','ball_nan'):
            del cfg.terminations[name]
    cfg.terminations['arena_goal'] = TerminationTermCfg(func=mdp.arena_goal_done)
    cfg.terminations['opponent_nan'] = TerminationTermCfg(func=mdp.robot_state_is_nan,
        params={'asset_cfg':SceneEntityCfg('opponent')})
    cfg.metrics = {f'arena_{name}':MetricsTermCfg(func=mdp.arena_metric,params={'name':name},reduce='last')
                   for name in ('scored','conceded','standing','struck')}
    return cfg


MicroduckArenaRlCfg = deepcopy(MicroduckFootballRlCfg)
MicroduckArenaRlCfg.experiment_name = 'football_arena'
MicroduckArenaRlCfg.run_name = 'selfplay'
MicroduckArenaRlCfg.max_iterations = 5000

# At 50 Hz, gamma=.997 gives a ~6.7 s credit horizon (old .99: ~2 s).
MicroduckArenaRlCfg.algorithm.gamma = .997


# --- kick-first-v3 --------------------------------------------------------------
# Built from the 2026-09-30 shot-outcome eval (scripts/eval_arena_shots.py). v2 lost the
# source kick (1.1 -> 0.3 m/s, aim 8 -> 16 deg; solo goals 83% -> 37%). Its rounds were
# decided by ~2 s and then ran dead (ball out of play at 10 s in 89% of rounds). Against
# an attacker, outcomes were close to random (kickoff scrum), so the kick got no clean
# credit. v3 starts from the football kicker, ends dead rounds and prices a fall once.
# It stages opponents so kicking is learned before contesting: empty goal, keeper, attacker.
V3_RECIPE = 'kick-first-v3'
ENABLE_V3_BALL_OUT_ENDS_ROUND = True
ENABLE_V3_OPEN_AIM = True
V3_FALL_COST = 10.0  # Once per fall (0.2 s down); v2 charged 5/s for the rest of the round.
V3_DEAD_ROUND_COST = TIMEOUT_COST  # Timeout or ball out without a goal, once.
# (solo, keeper, attacker) opponent probabilities; env steps = iteration * 24.
V3_MODE_STAGES = [
    {'step': 0, 'probs': (1., 0., 0.)},
    {'step': 250*24, 'probs': (.4, .6, 0.)},
    {'step': 750*24, 'probs': (.2, .4, .4)},
    {'step': 1250*24, 'probs': (.15, .35, .5)},
]


def make_microduck_arena_v3_env_cfg(play=False,rough=False):
    cfg = make_microduck_arena_env_cfg(play=play,rough=rough)
    cfg.events['reset_arena'].params['mode_stages'] = deepcopy(V3_MODE_STAGES)
    cfg.commands['twist'].aim = 'open' if ENABLE_V3_OPEN_AIM else 'center'
    del cfg.rewards['arena_fallen']
    cfg.rewards['arena_fall_event'] = RewardTermCfg(func=mdp.arena_progress,weight=-V3_FALL_COST,
        params={'component':'fall_delta'})  # fall_delta >= 0: negative weight.
    cfg.rewards['arena_timeout'] = RewardTermCfg(func=mdp.arena_timeout_penalty,weight=-V3_DEAD_ROUND_COST,
        params={'include_ball_out':ENABLE_V3_BALL_OUT_ENDS_ROUND})
    if ENABLE_V3_BALL_OUT_ENDS_ROUND:
        cfg.terminations['arena_ball_out'] = TerminationTermCfg(func=mdp.arena_ball_out)
    cfg.observations['critic'].terms['opponent_mode'] = ObservationTermCfg(func=mdp.arena_opponent_mode_obs)
    for mode in mdp.ARENA_OPPONENT_MODES:
        cfg.metrics[f'arena_mode_{mode}'] = MetricsTermCfg(func=mdp.arena_metric,
            params={'name':f'mode_{mode}'},reduce='last')
        cfg.metrics[f'arena_scored_vs_{mode}'] = MetricsTermCfg(func=mdp.arena_metric,
            params={'name':f'scored_vs_{mode}'},reduce='last')
    return cfg


MicroduckArenaV3RlCfg = deepcopy(MicroduckArenaRlCfg)
MicroduckArenaV3RlCfg.experiment_name = 'football_arena_v3'
MicroduckArenaV3RlCfg.run_name = 'kick-first'
MicroduckArenaV3RlCfg.max_iterations = 2000


# --- keeper-v1: the learner defends -----------------------------------------------
# The learner keeps its own (-x) goal via the keeper tracker (mdp.arena_keeper_command) against
# a frozen attacker (use train_arena.py --no-self-play). The scripted keeper made from the
# football actor fell ~15-20% of the time under keeper commands; this trains that walking.
KEEPER_RECIPE = 'keeper-v1'
KEEPER_CLEAN_ROUND_REWARD = 5.0  # Once: timeout or ball out without conceding.
KEEPER_FALL_COST = V3_FALL_COST


def make_microduck_arena_keeper_env_cfg(play=False,rough=False):
    cfg = make_microduck_arena_env_cfg(play=play,rough=rough)
    cfg.commands['twist'].role = 'keeper'
    cfg.actions['joint_pos'].opponent_aim = 'open'  # Train against the shots the eval/attacker policy takes.
    cfg.events['reset_arena'].params['mode_stages'] = [{'step':0,'probs':(0.,0.,1.)}]
    for name in ('arena_approach_delta','arena_ball_delta','arena_strike_delta','arena_shot_speed_delta','arena_fallen'):
        del cfg.rewards[name]
    # goal_delta keeps weight GOAL_REWARD: a conceded goal is -100 (a keeper goal would be +100).
    cfg.rewards['arena_timeout'] = RewardTermCfg(func=mdp.arena_timeout_penalty,weight=KEEPER_CLEAN_ROUND_REWARD,
        params={'include_ball_out':True})  # Positive here: surviving the round is the keeper's success.
    cfg.rewards['arena_fall_event'] = RewardTermCfg(func=mdp.arena_progress,weight=-KEEPER_FALL_COST,
        params={'component':'fall_delta'})
    cfg.terminations['arena_ball_out'] = TerminationTermCfg(func=mdp.arena_ball_out)
    cfg.metrics = {f'arena_{name}':MetricsTermCfg(func=mdp.arena_metric,params={'name':name},reduce='last')
                   for name in ('scored','conceded','standing')}
    return cfg


MicroduckArenaKeeperRlCfg = deepcopy(MicroduckArenaRlCfg)
MicroduckArenaKeeperRlCfg.experiment_name = 'football_arena_keeper'
MicroduckArenaKeeperRlCfg.run_name = 'keeper'
MicroduckArenaKeeperRlCfg.max_iterations = 1500
