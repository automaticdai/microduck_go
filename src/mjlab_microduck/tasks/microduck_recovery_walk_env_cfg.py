"""Isolated recovery-to-walking motor practice; no football entities or rewards.

Measured recovery endings supply 75% of resets; normal locomotion resets retain
basic walking. The 61D actor and groundcontact BAM robot match the source actor.
Use scripts/train_recovery_walk.py to attach the measured state bank.
"""
from copy import deepcopy
from mjlab.managers import EventTermCfg, TerminationTermCfg, RewardTermCfg
from . import mdp
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

ENABLE_SYMMETRY = False
START_PROBABILITY = .75
RECIPE = 'isolated-recovery-walk-v3'
COMMAND_BUCKET_PROBABILITIES = (.10,.05,.05,.05,.20,.10,.15,.15,.15)
LINEAR_TRACK_STD = .10
YAW_TRACK_STD = .25
VELOCITY_REWARD_TAU_S = .15
TASK = 'Mjlab-RecoveryWalk-Flat-MicroDuck'


def make_microduck_recovery_walk_env_cfg(play=False, rough=False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.scene.entities['robot'] = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)
    cfg.sim.nconmax = 100
    cfg.sim.mujoco.iterations = 30
    cfg.episode_length_s = 8.
    cfg.actions['joint_pos'] = mdp.RecoveryWalkActionCfg(
        entity_name='robot', actuator_names=('^(?!passive_).*',), scale=1.)
    cfg.commands['twist'] = mdp.RecoveryWalkCommandCfg(**vars(cfg.commands['twist']))
    cfg.commands['twist'].resampling_time_range = (9., 9.)
    cfg.commands['twist'].bucket_probabilities = COMMAND_BUCKET_PROBABILITIES
    cfg.commands['twist'].stop_end_s = 5.
    for name, dim in [('head_pose',4),('body_pose',6)]:
        cfg.commands[name].ranges = ((-.001,.001),)*dim
        cfg.commands[name].zero_command_prob = .25
    cfg.events.pop('push_robot', None)
    cfg.events['recovery_start'] = EventTermCfg(func=mdp.reset_recovery_walk,
        mode='reset', params={'start_probability':START_PROBABILITY})
    # Match mature source randomization, without inherited pose/regularizer ramps.
    cfg.curriculum = {k:v for k,v in cfg.curriculum.items() if k in ('com_range','head_com_range')}
    for term in cfg.curriculum.values():
        term.params['range_stages'] = [{**term.params['range_stages'][-1], 'step':0}]
    cfg.rewards['track_linear_velocity'] = RewardTermCfg(
        func=mdp.RecoveryWalkTracking, weight=4., params={
            'component':'linear', 'std':LINEAR_TRACK_STD, 'tau_s':VELOCITY_REWARD_TAU_S})
    cfg.rewards['track_angular_velocity'] = RewardTermCfg(
        func=mdp.RecoveryWalkTracking, weight=2., params={
            'component':'angular', 'std':YAW_TRACK_STD, 'tau_s':VELOCITY_REWARD_TAU_S})
    for name, component, weight in [('linear_progress','linear',4.), ('yaw_progress','angular',2.)]:
        cfg.rewards[name] = RewardTermCfg(func=mdp.RecoveryWalkTracking, weight=weight,
            params={'component':component, 'std':1., 'tau_s':VELOCITY_REWARD_TAU_S, 'progress':True})
    # Movement is the task. Posture and head alignment are supporting terms.
    for name, weight in {'upright':1., 'pose':.1, 'head_pose_tracking':.2,
                         'air_time':.3, 'body_ang_vel':-.01,
                         'angular_momentum':-.005, 'action_rate_l2':-.02}.items():
        cfg.rewards[name].weight = weight
    cfg.terminations['handoff_failed'] = TerminationTermCfg(func=mdp.fallen_too_long,
        params={'gate_z_below':.095,'gate_tilt_above_deg':30.,'max_duration_s':.2})
    return cfg


MicroduckRecoveryWalkRlCfg = deepcopy(MicroduckRlCfg)
MicroduckRecoveryWalkRlCfg.experiment_name = 'recovery_walk'
MicroduckRecoveryWalkRlCfg.run_name = 'isolated-handoff'
MicroduckRecoveryWalkRlCfg.max_iterations = 1000
MicroduckRecoveryWalkRlCfg.algorithm.learning_rate = 1e-4
MicroduckRecoveryWalkRlCfg.algorithm.schedule = 'fixed'
MicroduckRecoveryWalkRlCfg.algorithm.entropy_coef = .001
