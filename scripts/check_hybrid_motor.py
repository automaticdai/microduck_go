"""3v3 first-round battery: blue football motor vs walking-expert/football hybrid.

Read-only evaluation, no training. Orange always keeps the original football
motor and both keepers stay frozen, so goals and concessions are a direct A/B.
Blue motor modes:

  football  the original attacker motor everywhere (baseline)
  walker    the walking expert everywhere (navigation commands only)
  hybrid    walking expert by default; football motor within --enter m of the
            ball (released beyond --exit m); walker forced for --settle s after
            each recovery handoff

Every mode keeps the 61D observation, unfiltered actions, frozen normalizers and
the shared recovery controller. Motor paths default to the tactical checkpoint's
metadata; override them with local copies of the same files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg

import train_mappo as tm


class HybridMotor:
    """Per-player motor switch with distance hysteresis and post-recovery settle."""
    def __init__(self, env, name, walker, football, enter, exit, settle):
        self.env, self.name, self.walker, self.football = env, name, walker, football
        self.index = tuple(tm.mdp.ARENA_TEAM_PLAYERS).index(name)
        self.enter, self.exit, self.settle = enter, exit, settle
        self.near = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.since_recovery = torch.full((env.num_envs,), 1e3, device=env.device)
        self.use_football = torch.zeros_like(self.near)

    def __call__(self, obs):
        env = self.env
        new_round = env.episode_length_buf <= 1
        self.near[new_round] = False
        self.since_recovery[new_round] = 1e3
        origin = env.scene.terrain.env_origins[:, :2]
        player = env.scene[self.name].data.root_link_pos_w[:, :2]-origin
        ball = env.scene['ball'].data.root_link_pos_w[:, :2]-origin
        distance = (player-ball).norm(dim=-1)
        self.near = torch.where(self.near, distance < self.exit, distance < self.enter)
        recovering = env.arena_recovery.active[:, self.index]
        self.since_recovery = torch.where(recovering, 0., self.since_recovery+env.step_dt)
        self.use_football = self.near & (self.since_recovery >= self.settle-.5*env.step_dt)
        return torch.where(self.use_football[:, None], self.football(obs), self.walker(obs))


def nearest_roles(env, striker, partner, margin=.05):
    """Nearer blue attacker pursues; the other plays `partner` ('cover' or 'support').

    `striker` (index into MAPPO_BLUE) changes only when the other is `margin` m closer.
    Support takes the side toward the pitch centre line (y = 0).
    """
    origin = env.scene.terrain.env_origins[:, :2]
    ball = env.scene['ball'].data.root_link_pos_w[:, :2]-origin
    distance = torch.stack([(env.scene[n].data.root_link_pos_w[:, :2]-origin-ball).norm(dim=-1)
                            for n in tm.mdp.MAPPO_BLUE], 1)
    other = 1-striker
    striker = torch.where(distance.gather(1, other[:, None])[:, 0]
                          < distance.gather(1, striker[:, None])[:, 0]-margin, other, striker)
    role = torch.full_like(striker, 3) if partner == 'cover' else torch.where(ball[:, 1] > 0, 2, 1)
    skills = torch.stack((role, role), 1)
    skills.scatter_(1, striker[:, None], 0)
    return striker, skills


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True, help='Tactical MAPPO checkpoint')
    p.add_argument('--walker', type=Path, required=True)
    p.add_argument('--attacker', type=Path)
    p.add_argument('--blue-attacker', type=Path, help='Blue football motor; orange keeps --attacker')
    p.add_argument('--keeper', type=Path)
    p.add_argument('--recovery', type=Path)
    p.add_argument('--blue-motor', choices=['football', 'walker', 'hybrid'], required=True)
    p.add_argument('--tactics', choices=['policy', 'tracker', 'nearest-cover', 'nearest-support'], default='policy')
    p.add_argument('--enter', type=float, default=.35)
    p.add_argument('--exit', type=float, default=.50)
    p.add_argument('--settle', type=float, default=1.)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--round-seconds', type=float, default=20.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 < a.enter <= a.exit or a.settle < 0:
        p.error('Need 0 < enter <= exit and settle >= 0')
    tm.configure_torch_backends()
    torch.manual_seed(a.seed)
    ck = torch.load(a.checkpoint, map_location='cuda:0', weights_only=False)
    meta = ck['metadata']
    paths = {k: getattr(a, k) or Path(meta[k]) for k in ('attacker', 'keeper', 'recovery')}
    cfg = tm.load_env_cfg(tm.TASK)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    cfg.episode_length_s = a.round_seconds
    cfg.sim.nan_guard.enabled = False
    cfg.commands['twist'].track_training_signals = False
    cfg.curriculum = {}; cfg.metrics = {}
    names = tuple(tm.mdp.ARENA_TEAM_PLAYERS)
    cfg.scene.sensors = (*cfg.scene.sensors, *(ContactSensorCfg(name='diag_ball_'+n,
        primary=ContactMatch(mode='body', pattern='.*', entity=n),
        secondary=ContactMatch(mode='geom', pattern='^ball_geom$', entity='ball'),
        fields=('found', 'force'), reduce='netforce', num_slots=1,
        history_length=cfg.decimation) for n in names))
    frames = []
    alive = torch.ones(a.num_envs, dtype=torch.bool, device='cuda:0')
    hybrids = {}

    def measured_reward(env, gamma):
        reward = tm.mdp.arena_mappo_team_reward(env, gamma)
        origin = env.scene.terrain.env_origins
        standing = torch.stack([(env.scene[n].data.root_link_pos_w[:, 2]-origin[:, 2] >= .095)
            & (env.scene[n].data.projected_gravity_b[:, 2] < -.8660254) for n in names], 1)
        contact = torch.stack([torch.nan_to_num(env.scene.sensors['diag_ball_'+n].data.force_history)
            .abs().flatten(1).sum(-1).gt(1e-6) for n in names], 1)
        nan = torch.stack([env.termination_manager.get_term(k)
            for k in env.termination_manager.active_terms if 'nan' in k]).any(0)
        football = torch.stack([hybrids[n].use_football if n in hybrids
            else torch.full_like(alive, a.blue_motor == 'football') for n in tm.mdp.MAPPO_BLUE], 1)
        frames.append({k: v.detach().clone() for k, v in dict(
            alive=alive, standing=standing, contact=contact, recovery=env.arena_recovery.active,
            entered=env.arena_recovery.entered, goal=env.arena_goal_result,
            done=env.reset_buf, nan=nan, football=football,
            ball_x=env.scene['ball'].data.root_link_pos_w[:, 0]-origin[:, 0]).items()})
        return reward

    cfg.rewards = {'mappo_team': tm.RewardTermCfg(func=measured_reward, weight=1., params={'gamma': tm.GAMMA})}
    env = tm.ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        env.arena_recovery_policy = tm.frozen_motor(paths['recovery'], 'cuda:0')
        env.arena_recovery = tm.mdp.ArenaRecovery(env)
        football = tm.frozen_motor(paths['attacker'], 'cuda:0')
        walker = tm.frozen_motor(a.walker, 'cuda:0')
        arena = tm.TacticalArena(env, football, tm.frozen_motor(paths['keeper'], 'cuda:0'))
        if a.blue_attacker:
            football = arena.motor = env.arena_teammate_policy = tm.frozen_motor(a.blue_attacker, 'cuda:0')
        if a.blue_motor == 'walker':
            arena.motor = env.arena_teammate_policy = walker
        elif a.blue_motor == 'hybrid':
            for n in tm.mdp.MAPPO_BLUE:
                hybrids[n] = HybridMotor(env, n, walker, football, a.enter, a.exit, a.settle)
            arena.motor = hybrids['robot']
            env.arena_teammate_policy = hybrids['blue_attacker']
        model = tm.TacticalMAPPO(ck['actor_dim'], ck['critic_dim']).to('cuda:0')
        model.load_state_dict(ck['model']); model.eval()
        skills = torch.zeros(a.num_envs, 2, dtype=torch.long, device='cuda:0')
        striker = torch.zeros(a.num_envs, dtype=torch.long, device='cuda:0')
        with torch.inference_mode():
            while alive.any():
                obs, _ = arena.observations()
                if a.tactics == 'policy':
                    skills = model.actor(obs).argmax(-1)
                elif a.tactics.startswith('nearest'):
                    striker, skills = nearest_roles(env, striker, a.tactics.split('-')[1])
                arena.evaluation_mask = alive.float()
                _, _, done, _, _ = arena.step(skills)
                alive &= ~done
                if len(frames) % 250 < 5:
                    print('PROGRESS', len(frames), int(alive.sum()), flush=True)
        data = {k: torch.stack([f[k] for f in frames]).cpu().numpy() for k in frames[0]}
        live = data['alive'] & ~data['nan']
        dt = env.step_dt
        players = {}
        for i, n in enumerate(names):
            touched = data['contact'][:, :, i] & live
            players[n] = dict(
                standing_fraction=float((data['standing'][:, :, i] & live).sum()/live.sum()),
                recovering_fraction=float((data['recovery'][:, :, i] & live).sum()/live.sum()),
                recovery_entries_per_round=float((data['entered'][:, :, i] & live).sum()/a.num_envs),
                touched_rounds=int(touched.any(0).sum()), contact_seconds=float(touched.sum()*dt))
            if n in tm.mdp.MAPPO_BLUE:
                players[n]['football_motor_fraction'] = float(
                    (data['football'][:, :, tm.mdp.MAPPO_BLUE.index(n)] & live).sum()/live.sum())
        goal = data['goal'] * data['alive']
        report = dict(blue_motor=a.blue_motor, tactics=a.tactics, enter=a.enter, exit=a.exit,
            settle=a.settle, seed=a.seed, round_seconds=a.round_seconds, rounds=a.num_envs,
            scored=int((goal > 0).sum()), conceded=int((goal < 0).sum()),
            numerical_rounds=int((data['nan'] & data['alive']).any(0).sum()),
            mean_ball_x=float(data['ball_x'][live].mean()),
            ball_in_blue_attacking_half=float((data['ball_x'][live] > 0).mean()),
            motors={k: str(v) for k, v in {**paths, 'blue_attacker': a.blue_attacker or paths['attacker'], 'walker': a.walker, 'checkpoint': a.checkpoint}.items()},
            players=players)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2))
        print('RESULT', json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
