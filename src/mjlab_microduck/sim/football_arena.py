"""Two independent 61D football policies in one CPU MuJoCo/BAM match.

This is an inference arena, not a self-play-trained policy. Both players use
identical weights and the training navigation command; opponents enter through
physical contact only. The referee resets only after goals or the round timeout.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import re
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from bam.model import load_model

import mjlab.tasks  # Initialize the task registry before importing robot configs.

from mjlab_microduck.actuator.cpu_bam import CpuBamController
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_ball_spec, get_standup_spec
from mjlab_microduck.tasks.microduck_football_env_cfg import get_football_goal_spec
from mjlab_microduck.tasks.mdp import ArenaRecovery, ARENA_TEAM_PLAYERS

HALF_LENGTH = 0.9
HALF_WIDTH = 0.65
BALL_RADIUS = 0.035
CENTER_CIRCLE_RADIUS = 0.22


def goal_crossing(previous, current):
    """Award Blue (+x) or Orange (-x) after the whole ball crosses the mouth."""
    for player, direction in enumerate((1, -1)):
        plane = HALF_LENGTH + BALL_RADIUS
        a, b = direction * previous[0], direction * current[0]
        if a < plane <= b:
            fraction = (plane - a) / (b - a)
            crossing = previous + fraction * (current - previous)
            if abs(crossing[1]) <= 0.2 - BALL_RADIUS and BALL_RADIUS - .005 <= crossing[2] <= 0.30 - BALL_RADIUS:
                return player
    return None


def navigation(ball_xy, direction):
    """NumPy equivalent of football_navigation_command, without a strike latch.

    Re-engagement is necessary in a match: the original episode ends after one
    strike. No ball force, opponent tracking or extra policy inputs are added.
    """
    side = np.array([-direction[1], direction[0]])
    error = ball_xy - .10 * direction + .042 * side
    heading = np.arctan2(direction[1], direction[0])
    velocity = (2 * error + .08 * np.exp(-(np.linalg.norm(error) / .06) ** 2) * direction)
    velocity *= max(0, np.cos(heading))
    return np.clip([*velocity, 2 * heading], [-.25, -.2, -1], [.35, .2, 1])


def build_world():
    spec = mujoco.MjSpec.from_string('''<mujoco><option timestep="0.005" iterations="30" integrator="implicitfast"/>
      <worldbody><light pos="0 0 3"/><geom name="floor" type="plane" size="3 3 .1"
      rgba=".32 .34 .36 1" friction="1 .005 .0001"/></worldbody></mujoco>''')
    # Visual pitch tile bounded by the touchlines; floor contact stays at z=0.
    spec.worldbody.add_geom(
        name='playground_tile', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0], size=[HALF_LENGTH, HALF_WIDTH, .0005],
        rgba=[.08, .38, .16, 1], contype=0, conaffinity=0,
    )
    for i in range(2):
        robot = get_standup_spec()
        for geom in robot.geoms:
            if geom.name.endswith('_collision'):
                geom.contype = geom.conaffinity = 1
                geom.condim = 3 if geom.name in ('left_foot_collision', 'right_foot_collision') else 1
                if geom.condim == 3:
                    geom.priority = 1
                    geom.friction = [1, .005, .0001]
        # A small visual-only team marker attached to the trunk.
        robot.body('trunk_base').add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[.016, 0, 0], pos=[0, 0, .055], contype=0, conaffinity=0,
            mass=0, rgba=([.1, .45, 1, 1] if i == 0 else [1, .3, .05, 1]))
        spec.attach(robot, prefix=f'p{i}_', frame=spec.worldbody.add_frame())
        goal = get_football_goal_spec()
        body = goal.body('goal')
        body.mocap = False
        body.pos = [HALF_LENGTH * (1 if i == 0 else -1), 0, 0]
        body.quat = [1, 0, 0, 0] if i == 0 else [0, 0, 0, 1]
        spec.attach(goal, prefix=f'g{i}_', frame=spec.worldbody.add_frame())
    spec.attach(get_ball_spec(), prefix='', frame=spec.worldbody.add_frame())
    for x, y, sx, sy in ((0, HALF_WIDTH, HALF_LENGTH, .005), (0, -HALF_WIDTH, HALF_LENGTH, .005),
                          (HALF_LENGTH, 0, .005, HALF_WIDTH), (-HALF_LENGTH, 0, .005, HALF_WIDTH),
                          (0, 0, .004, HALF_WIDTH)):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x,y,.001],
            size=[sx,sy,.001], rgba=[1,1,1,1], contype=0, conaffinity=0)
    # A thin, non-colliding center-circle marking around the kickoff region.
    angles = np.linspace(0, 2 * np.pi, 97)
    for a, b in zip(angles[:-1], angles[1:]):
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[.003, 0, 0],
            fromto=[CENTER_CIRCLE_RADIUS*np.cos(a), CENTER_CIRCLE_RADIUS*np.sin(a), .001,
                    CENTER_CIRCLE_RADIUS*np.cos(b), CENTER_CIRCLE_RADIUS*np.sin(b), .001],
            rgba=[1, 1, 1, 1], contype=0, conaffinity=0,
        )
    bam = load_model(motor_name='xl330', model='m6')
    # Keep the M6 firmware current limit, matching the training actuator.
    bam.actuator.kp, bam.actuator.vin = 200, 7.4
    limit = 7.4 * bam.kt.value / bam.R.value
    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited, act.ctrllimited = True, False
        act.forcerange = [-limit, limit]
        act.gear = [1,0,0,0,0,0]
        joint = spec.joint(act.target)
        joint.damping = np.zeros((3,1))
        joint.frictionloss = 0
        joint.solref_friction = [-5e4,-2e2]
        joint.solimp_friction = [.99,.9999,.001,.5,2]
    return spec.compile(), bam


class CpuRecovery:
    """Adapt CPU MuJoCo/ONNX to the same recovery controller as GPU matches."""
    def __init__(self, session, step_dt):
        self.session=session
        class Scene(dict):
            terrain=SimpleNamespace(env_origins=torch.zeros(1,3))
        self.env=SimpleNamespace(scene=Scene(),num_envs=1,device='cpu',step_dt=step_dt,
                                arena_recovery_policy=self.infer)
        self.controller=ArenaRecovery(self.env)

    def infer(self, observation):
        result=self.session.run(None,{self.session.get_inputs()[0].name:observation['actor'].numpy()})[0]
        return torch.from_numpy(result)

    def select(self, player, observation, action, height):
        name=('robot','opponent')[player]
        obs=torch.from_numpy(observation[None])
        self.env.scene[name]=SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.,0.,height]]),
            projected_gravity_b=obs[:,3:6],root_link_ang_vel_b=obs[:,:3]))
        return self.controller.select(name,obs,torch.from_numpy(action[None])).numpy()[0]

    def labels(self):
        players=tuple(ARENA_TEAM_PLAYERS)
        return ['Recovery' if self.controller.active[0,players.index(name)] else 'Football'
                for name in ('robot','opponent')]


class Arena:
    def __init__(self, policy: Path, seed=0, recovery_policy: Path | None = None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.policy = ort.InferenceSession(str(policy), options, providers=['CPUExecutionProvider'])
        if self.policy.get_inputs()[0].shape[-1] != 61 or self.policy.get_outputs()[0].shape[-1] != 14:
            raise ValueError('Arena requires the normalized 61D → 14D football ONNX policy')
        self.model, bam = build_world()
        self.recovery=None
        if recovery_policy is not None:
            session=ort.InferenceSession(str(recovery_policy),options,providers=['CPUExecutionProvider'])
            if session.get_inputs()[0].shape[-1]!=61 or session.get_outputs()[0].shape[-1]!=14:
                raise ValueError('Recovery requires a normalized 61D → 14D ONNX policy')
            self.recovery=CpuRecovery(session,4*self.model.opt.timestep)
        self.data = mujoco.MjData(self.model)
        self.controllers, self.roots, self.homes = [], [], []
        for i in range(2):
            names = [self.model.actuator(a).name for a in range(self.model.nu)
                     if self.model.actuator(a).name.startswith(f'p{i}_')]
            controller = CpuBamController(bam, names, self.model, self.data)
            self.controllers.append(controller)
            home = []
            for j in controller.joint_indexes:
                name = self.model.joint(j).name[3:]
                home.append(next(v for pattern,v in HOME_FRAME.joint_pos.items() if re.fullmatch(pattern,name)))
            self.homes.append(np.array(home))
            self.roots.append(self.model.body(f'p{i}_trunk_base').id)
        self.ball_qadr = self.model.joint('ball_free').qposadr[0]
        self.rng = np.random.default_rng(seed)
        self.score = [0,0]
        self.resets = {'goal':0, 'timeout':0}
        self.round = 0
        self.last_event = 'Kickoff'
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.recovery is not None:
            self.recovery.controller.reset()
        self.last_action = np.zeros((2,14))
        self.action_delay = deque([np.zeros((2,14)) for _ in range(4)])
        self.last_velocity = np.zeros((2,14))
        self.steps = 0
        self.round += 1
        # Uniform by area across the center disk (sqrt avoids center bias).
        radius = CENTER_CIRCLE_RADIUS * np.sqrt(self.rng.random())
        angle = self.rng.uniform(0, 2 * np.pi)
        self.data.qpos[self.ball_qadr:self.ball_qadr+3] = [
            radius * np.cos(angle), radius * np.sin(angle), BALL_RADIUS,
        ]
        for i, controller in enumerate(self.controllers):
            root_joint = self.model.body(self.roots[i]).jntadr[0]
            adr = self.model.jnt_qposadr[root_joint]
            self.data.qpos[adr:adr+3] = [(-.38 if i == 0 else .38), 0, .125]
            self.data.qpos[adr+3:adr+7] = [1,0,0,0] if i == 0 else [0,0,0,1]
            self.data.qpos[controller.qpos_indexes] = self.homes[i]
            controller.reset(self.data.qpos)
            controller.last_ts = 0
        mujoco.mj_forward(self.model, self.data)
        self.previous_ball = self.ball.copy()

    @property
    def ball(self):
        return self.data.qpos[self.ball_qadr:self.ball_qadr+3]

    def observation(self, i):
        root = self.roots[i]
        rotation = self.data.xmat[root].reshape(3,3)
        position = self.data.xpos[root]
        ball_xy = (rotation.T @ (self.ball-position))[:2]
        goal = np.array([HALF_LENGTH*(1 if i == 0 else -1),0,0])
        direction = goal[:2]-self.ball[:2]
        direction /= max(np.linalg.norm(direction),1e-8)
        direction = (rotation.T @ np.r_[direction,0])[:2]
        command = navigation(ball_xy, direction)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,root,velocity,1)
        controller = self.controllers[i]
        obs = np.r_[velocity[:3], -rotation[2,:],
            self.data.qpos[controller.qpos_indexes]-self.homes[i], self.last_velocity[i],
            self.last_action[i],command,np.zeros(10)].astype(np.float32)
        self.last_velocity[i] = self.data.qvel[controller.dof_indexes]
        return obs

    def step(self):
        if self.steps % 4 == 0:
            for i,controller in enumerate(self.controllers):
                observation=self.observation(i)
                action = self.policy.run(None,{self.policy.get_inputs()[0].name:observation[None]})[0][0]
                if self.recovery is not None:
                    action=self.recovery.select(i,observation,action,self.data.xpos[self.roots[i],2])
                if not np.isfinite(action).all():
                    raise RuntimeError('Non-finite policy action')
                self.last_action[i] = action
        self.action_delay.append(self.last_action.copy())
        delayed_action = self.action_delay.popleft()
        for i, controller in enumerate(self.controllers):
            controller.q_target[:] = self.homes[i] + delayed_action[i]
            controller.update()
        mujoco.mj_step(self.model,self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite arena physics')
        scorer = goal_crossing(self.previous_ball,self.ball)
        reason = None
        if scorer is not None:
            self.score[scorer] += 1
            reason = 'goal'
            self.last_event = f'{("Blue","Orange")[scorer]} scores!'
        elif self.steps * self.model.opt.timestep >= 10:
            reason = 'timeout'
        self.previous_ball = self.ball.copy()
        if reason:
            self.resets[reason] += 1
            if reason != 'goal':
                self.last_event = f'{reason.title()} — new kickoff (no point)'
            print(json.dumps(self.status()),flush=True)
            self.reset()

    def status(self):
        return {'score':self.score, 'round':self.round, 'event':self.last_event, 'resets':self.resets,
                'controllers':self.recovery.labels() if self.recovery else ['Football','Football'],
                'recovery_enabled':self.recovery is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,default=Path('logs/football_final.onnx'))
    parser.add_argument('--recovery-policy',type=Path,help='Normalized get-up ONNX exported with scripts/export.py')
    parser.add_argument('--port',type=int,default=8081)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--headless-seconds',type=float,default=0)
    args = parser.parse_args()
    arena = Arena(args.policy,args.seed,args.recovery_policy)
    if args.headless_seconds:
        for _ in range(round(args.headless_seconds/.005)):
            arena.step()
        print(json.dumps(arena.status()))
        return
    import viser
    from mjviser import Viewer
    server = viser.ViserServer(host='0.0.0.0',port=args.port)
    scoreboard = server.gui.add_markdown('')
    score_label = server.scene.add_label(
        '/match_score', 'Blue 0:0 Orange', position=(0, HALF_WIDTH + .12, .45),
        font_screen_scale=2.0, anchor='center-center',
    )
    def render(scene):
        scene.update_from_mjdata(arena.data)
        score_text = f'Blue {arena.score[0]}:{arena.score[1]} Orange'
        if score_label.text != score_text:
            score_label.text = score_text
        scoreboard.content = (f'## {score_text}\n'
            f'Policy: **{args.policy.stem}**\n\n'
            f'Recovery: **{"enabled" if arena.recovery else "disabled"}** · '
            f'Controllers: **{" / ".join(arena.status()["controllers"])}**\n\n'
            f'Blue attacks +X · Orange attacks −X\n\nRound {arena.round} · '
            f'{arena.steps * arena.model.opt.timestep:.1f} / 10 s · {arena.last_event}\n\n'
            'Same trained kicking policy on both teams. Rounds end only on a goal or the 10-second timeout.')
    @server.on_client_connect
    def connect(client):
        client.camera.position = (1.7,-2.1,1.7)
        client.camera.look_at = (0,0,.1)
    viewer = Viewer(arena.model,arena.data,step_fn=lambda m,d:arena.step(),render_fn=render,
                    reset_fn=lambda m,d:arena.reset(),server=server)
    viewer.scene.camera_tracking_enabled = False
    viewer.run()


if __name__ == '__main__':
    main()
