"""CUDA/Warp arena playback using the same BAM physics as cloud training."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import mjlab.tasks  # Initialize the registry before robot/task imports.
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viewer_config import ViewerConfig
from mjlab_microduck.robot.microduck_constants import get_standup_spec

TASK = 'Mjlab-FootballArena-Flat-MicroDuck'
TEAMS_TASK = 'Mjlab-FootballArena-3v3-Flat-MicroDuck'


from mjlab_microduck.tasks.microduck_arena_env_cfg import arena_boards, BOARD_HEIGHT
from mjlab_microduck.tasks.microduck_football_env_cfg import get_football_goal_spec
from mjlab_microduck.tasks.mdp import (
    ARENA_TEAM_PLAYERS, ArenaRecovery, ArenaFinishingJointPositionActionCfg,
    arena_recovery_action,
)

TEAM_COLORS = {'robot': (25,115,255), 'opponent': (255,77,13)}


def team_goal_spec(team):
    spec=get_football_goal_spec()
    for geom in spec.geoms:
        geom.rgba=[*(channel/255 for channel in TEAM_COLORS[team]),geom.rgba[3]]
    return spec


def pitch_spec(boards=False, half_length=.9, half_width=.65):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='pitch')
    def box(name,pos,size,color):
        body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,
                      rgba=color,contype=0,conaffinity=0)
    box('green_tile',[0,0,0],[half_length,half_width,.0005],[.08,.38,.16,1])
    for i,(x,y,sx,sy) in enumerate(((0,half_width,half_length,.005),(0,-half_width,half_length,.005),
                                   (half_length,0,.005,half_width),(-half_length,0,.005,half_width),(0,0,.004,half_width))):
        box(f'line_{i}',[x,y,.001],[sx,sy,.001],[1,1,1,1])
    angles = np.linspace(0,2*np.pi,65)
    for a,b in zip(angles[:-1],angles[1:]):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[.003,0,0],
            fromto=[.22*np.cos(a),.22*np.sin(a),.001,.22*np.cos(b),.22*np.sin(b),.001],
            rgba=[1,1,1,1],contype=0,conaffinity=0)
    if boards:
        # Low boards: side lines, plus end lines beside each goal (the goal and net close the mouth).
        h = BOARD_HEIGHT/2
        for i,(x,y,sx,sy) in enumerate(arena_boards(half_length,half_width)):
            body.add_geom(name=f'board_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,y,h],size=[sx,sy,h],
                          rgba=[.95,.95,.95,.9],contype=1,conaffinity=1)
    return spec


def blue_spec():
    spec=get_standup_spec()
    spec.body('trunk_base').add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[.016,0,0],pos=[0,0,.055],rgba=[.1,.45,1,1],mass=0,contype=0,conaffinity=0)
    return spec


def orange_spec():
    spec=blue_spec()
    spec.body('trunk_base').geoms[-1].rgba=[1,.3,.05,1]
    return spec


def playback_cfg(boards=True,goal_hold_s=2.,teams=False):
    cfg=load_env_cfg(TEAMS_TASK if teams else TASK,play=True)
    cfg.scene.num_envs=1
    # The viewer's pitch provides boards; avoid duplicating the training entity.
    cfg.scene.entities.pop('boards',None)
    cfg.scene.entities['pitch']=EntityCfg(spec_fn=partial(pitch_spec,boards=boards,half_length=cfg.commands['twist'].pitch_half_length,
        half_width=cfg.commands['twist'].pitch_half_width))
    cfg.commands['twist'].goal_hold_s=goal_hold_s
    cfg.scene.entities['robot'].spec_fn=blue_spec
    cfg.scene.entities['opponent'].spec_fn=orange_spec
    if teams:
        for name,(sign,_) in ARENA_TEAM_PLAYERS.items():
            cfg.scene.entities[name].spec_fn=blue_spec if sign == 1 else orange_spec
    cfg.scene.entities['own_goal'].spec_fn=partial(team_goal_spec,'robot')
    cfg.scene.entities['goal'].spec_fn=partial(team_goal_spec,'opponent')
    # Playback needs actor inputs and match rules, not training diagnostics.
    cfg.commands['twist'].track_training_signals=False
    cfg.observations={'actor':cfg.observations['actor']}
    cfg.observations['actor'].enable_corruption=False
    cfg.rewards={}
    cfg.metrics={}
    cfg.curriculum={}
    cfg.scene.sensors=()
    cfg.terminations['nan_state'].params['sensor_names']=()
    cfg.sim.nan_guard.enabled=False
    cfg.viewer.origin_type=ViewerConfig.OriginType.WORLD
    cfg.viewer.lookat=(0,0,.1)
    cfg.viewer.distance=4.0 if teams else 2.6
    cfg.viewer.azimuth=125
    cfg.viewer.elevation=-35
    cfg.viewer.enable_reflections=False
    return cfg


class MatchEnv(RslRlVecEnvWrapper):
    def __init__(self,env,clip_actions=None):
        self.score=[0,0]
        self.round=1
        self.elapsed=0.
        self.last_event='Kickoff'
        self.goal_this_round=False
        self.cached_obs=None
        super().__init__(env,clip_actions)

    def get_observations(self):
        if self.cached_obs is None:
            self.cached_obs=super().get_observations()
        return self.cached_obs

    @torch.inference_mode()
    def reset(self):
        result=super().reset()
        recovery=getattr(self.unwrapped,'arena_recovery',None)
        if recovery is not None:
            recovery.reset()
        self.cached_obs=result[0]
        self.elapsed=0.
        return result

    def step(self,actions):
        if getattr(self.unwrapped,'arena_recovery',None) is not None:
            actions=arena_recovery_action(self.unwrapped,'robot',
                self.get_observations()['actor'],actions)
        result=super().step(actions)
        recovery=getattr(self.unwrapped,'arena_recovery',None)
        if recovery is not None:
            recovery.reset(result[2].nonzero().flatten())
        self.cached_obs=result[0]
        self.elapsed+=self.unwrapped.step_dt
        # One small GPU-to-CPU transfer for the referee, outside physics substeps.
        event=torch.stack((self.unwrapped.arena_goal_result[0],result[2][0])).cpu().tolist()
        if event[0]:
            player=0 if event[0]>0 else 1
            self.score[player]+=1
            self.last_event=f'{("Blue","Orange")[player]} scores!'
            self.goal_this_round=True
        elif event[1] and not self.goal_this_round:
            self.last_event='Timeout' if self.elapsed>=self.unwrapped.cfg.episode_length_s-.01 else 'Numerical reset'
        if event[1]:
            print(json.dumps({'score':self.score,'round':self.round,'event':self.last_event}),flush=True)
            self.round+=1
            self.elapsed=0.
            self.goal_this_round=False
        return result


class ArenaViewer(ViserPlayViewer):
    def __init__(self,*args,checkpoint,opponent_label='mirror',**kwargs):
        self.checkpoint=checkpoint
        self.opponent_label=opponent_label
        self.last_panel_update=0.
        self.last_perf_log=0.
        super().__init__(*args,**kwargs)

    def setup(self):
        super().setup()
        self._scene.debug_visualization_enabled=False
        self._scene.camera_tracking_enabled=False
        self.panel=self._server.gui.add_markdown('')
        self.score_label=self._server.scene.add_label('/match_score','Blue 0:0 Orange',
            position=(0,.77,.45),font_screen_scale=2.,anchor='center-center')
        players={name:(sign,role) for name,(sign,role) in ARENA_TEAM_PLAYERS.items()
                 if name in self.env.unwrapped.scene.entities}
        # Downward-pointing triangular pyramids stay upright above each player.
        # These are viewer overlays, so they have no mass or collision geometry.
        vertices=np.array([[0,0,0],[-.055,-.032,.09],
                           [.055,-.032,.09],[0,.064,.09]],dtype=np.float32)
        faces=np.array([[0,1,2],[0,2,3],[0,3,1],[1,3,2]],dtype=np.uint32)
        self.team_markers={
            name:self._server.scene.add_mesh_simple(
                f'/team_markers/{name}',vertices,faces,color=color,
                flat_shading=True,side='double')
            for name,(sign,_) in players.items()
            for color in (TEAM_COLORS['robot' if sign == 1 else 'opponent'],)
        }
        self.role_labels={name:self._server.scene.add_label(
            f'/team_roles/{name}',role.title(),font_screen_scale=.7)
            for name,(_,role) in players.items()}

    @torch.inference_mode()
    def _handle_gui_reset(self, all_envs):
        super()._handle_gui_reset(all_envs)
        match=self.env
        match.cached_obs=None
        # A manual reset starts a new match: clear the scoreboard and round counter too.
        match.score=[0,0]
        match.round=1
        match.elapsed=0.
        match.goal_this_round=False
        match.last_event='Kickoff'
        self.score_label.text='Blue 0:0 Orange'

    def _execute_step(self):
        with torch.inference_mode():
            return super()._execute_step()

    def sync_env_to_viewer(self):
        super().sync_env_to_viewer()
        positions=torch.stack([
            self.env.unwrapped.scene[name].data.root_link_pos_w[0]
            for name in self.team_markers
        ]).detach().cpu().numpy()
        for (name,marker),position in zip(self.team_markers.items(),positions):
            marker.position=tuple(position+np.array([0,0,.24]))
            if name in self.role_labels:
                self.role_labels[name].position=tuple(position+np.array([0,0,.36]))
        now=time.perf_counter()
        if now-self.last_panel_update < .25:
            return
        self.last_panel_update=now
        match=self.env
        recovery=getattr(match.unwrapped,'arena_recovery',None)
        active=(recovery.active[0].cpu().tolist() if recovery is not None
                else [False]*len(ARENA_TEAM_PLAYERS))
        for i,(name,(_,role)) in enumerate(ARENA_TEAM_PLAYERS.items()):
            if name in self.role_labels:
                self.role_labels[name].text='Recovery' if active[i] else role.title()
        score=f'Blue {match.score[0]}:{match.score[1]} Orange'
        if self.score_label.text!=score:
            self.score_label.text=score
        status=self.get_status()
        if now-self.last_perf_log >= 5:
            self.last_perf_log=now
            print(json.dumps({'realtime_factor':status.actual_realtime,'viewer_fps':status.smoothed_fps}),flush=True)
        self.panel.content=(f'## {score}\n\nBlue: **{self.checkpoint.stem}** · Orange: **{self.opponent_label}**\n\n'
            f'**{torch.cuda.get_device_name(0)} · CUDA physics + inference**\n\n'
            f'Playback: **{status.actual_realtime:.2f}× real time**\n\n'
            f'Recovery: **{"enabled" if recovery is not None else "disabled"}**\n\n'
            f'Round {match.round} · {match.elapsed:.1f} / {match.unwrapped.cfg.episode_length_s:g} s · {match.last_event}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--teams',action='store_true',help='3v3: two attackers and one defender per team')
    parser.add_argument('--defender-policy',type=Path,help='Keeper checkpoint for both 3v3 defenders')
    parser.add_argument('--recovery-policy',type=Path,
                        help='Get-up checkpoint, including its own observation normalizer')
    parser.add_argument('--port',type=int,default=8081)
    parser.add_argument('--benchmark-steps',type=int,default=0)
    parser.add_argument('--compile-friction',action='store_true')
    parser.add_argument('--opponent-role',choices=('attacker','keeper','solo'),default='attacker',
                        help='Orange duck: mirror attacker, goalkeeper, or parked off-pitch')
    parser.add_argument('--opponent-policy',type=Path,help='Orange checkpoint; default: same as --checkpoint')
    parser.add_argument('--goal-hold',type=float,default=2.,help='Seconds of play after a goal before the reset')
    parser.add_argument('--no-boards',action='store_true',help='Open pitch: the ball can roll away forever')
    parser.add_argument('--aim',choices=('center','open'),default='center',help="Blue tracker aim (open = beside orange's shadow)")
    args=parser.parse_args()
    if args.teams and not args.defender_policy:
        parser.error('--teams requires --defender-policy')
    if args.teams and args.opponent_role != 'attacker':
        parser.error('--teams uses fixed 3v3 roles; omit --opponent-role')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this viewer; use play_football_arena.py for CPU playback')
    torch.set_num_threads(1)
    configure_torch_backends()
    cfg=playback_cfg(boards=not args.no_boards,goal_hold_s=args.goal_hold,teams=args.teams)
    if args.recovery_policy and not args.teams:
        cfg.actions['joint_pos']=ArenaFinishingJointPositionActionCfg(
            **vars(cfg.actions['joint_pos']))
    probs={'solo':(1.,0.,0.),'keeper':(0.,1.,0.),'attacker':(0.,0.,1.)}[args.opponent_role]
    if not args.teams:
        cfg.events['reset_arena'].params['mode_stages']=[{'step':0,'probs':probs}]
        cfg.commands['twist'].aim=args.aim
    task=TEAMS_TASK if args.teams else TASK
    agent=load_rl_cfg(task)
    agent.obs_groups={'actor':('actor',),'critic':('actor',)}
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    if args.compile_friction:
        for name in ARENA_TEAM_PLAYERS:
            if name not in env.scene.entities:
                continue
            for actuator in env.scene[name].actuators:
                actuator._compute_friction_budget=torch.compile(
                    actuator._compute_friction_budget,mode='reduce-overhead')
    wrapped=MatchEnv(env,agent.clip_actions)
    try:
        runner=load_runner_cls(task)(wrapped,asdict(agent),device='cuda:0')
        runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location='cuda:0')
        policy=runner.get_inference_policy(device='cuda:0')
        if args.recovery_policy:
            recovery=deepcopy(runner.alg.actor).eval().requires_grad_(False)
            recovery.load_state_dict(torch.load(args.recovery_policy,map_location='cuda:0',
                                                weights_only=False)['actor_state_dict'])
            env.arena_recovery_policy=recovery
            env.arena_recovery=ArenaRecovery(env)
        print(f'Recovery policy: {args.recovery_policy or "DISABLED"}',flush=True)
        env.arena_opponent_policy=policy
        if args.opponent_policy:
            opponent=deepcopy(runner.alg.actor).eval().requires_grad_(False)
            opponent.load_state_dict(torch.load(args.opponent_policy,map_location='cuda:0',
                                                weights_only=False)['actor_state_dict'])
            env.arena_opponent_policy=opponent
        if args.teams:
            env.arena_teammate_policy=policy
            defender=deepcopy(runner.alg.actor).eval().requires_grad_(False)
            defender.load_state_dict(torch.load(args.defender_policy,map_location='cuda:0',
                                                weights_only=False)['actor_state_dict'])
            env.arena_defender_policy=defender
        assert next(runner.alg.actor.parameters()).device.type=='cuda'
        print(f'Physics and all player policies on {torch.cuda.get_device_name(0)}',flush=True)
        # Set only the background plane color; pitch markings are separate visuals.
        model=env.sim.mj_model
        for i in range(model.ngeom):
            if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE:
                model.geom_rgba[i]=[.32,.34,.36,1]
        if args.benchmark_steps:
            with torch.inference_mode():
                for _ in range(25):
                    wrapped.step(policy(wrapped.get_observations()))
                torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(args.benchmark_steps):
                    wrapped.step(policy(wrapped.get_observations()))
                torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            print(json.dumps({'device':torch.cuda.get_device_name(0),'wall_seconds':seconds,
                'sim_seconds':args.benchmark_steps*env.step_dt,
                'realtime_factor':args.benchmark_steps*env.step_dt/seconds}),flush=True)
        else:
            # Warm the compiled kernels before opening the interactive viewer.
            with torch.inference_mode():
                for _ in range(25):
                    wrapped.step(policy(wrapped.get_observations()))
            wrapped.reset()
            import viser
            server=viser.ViserServer(host='0.0.0.0',port=args.port,label='Microduck Go · CUDA')
            with torch.inference_mode():
                ArenaViewer(wrapped,policy,viser_server=server,checkpoint=args.checkpoint,
                    opponent_label=f'{"3v3" if args.teams else args.opponent_role} '
                                   f'({(args.opponent_policy or args.checkpoint).stem})').run()
    finally:
        env.close()


if __name__=='__main__':
    main()
