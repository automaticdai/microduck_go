"""Train/evaluate isolated recovery-to-walking on CUDA, with disjoint pose banks."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks.microduck_recovery_walk_env_cfg import TASK, RECIPE
from train_arena import critic_warmup


def load_bank(path, device):
    bank = torch.load(path, map_location=device, weights_only=True)
    n = len(bank['height'])
    shapes = {'height':(n,), 'quat':(n,4), 'velocity':(n,6),
              'joint_pos':(n,14), 'joint_vel':(n,14), 'last_action':(n,14)}
    if n < 16 or set(bank) != set(shapes) or any(
            tuple(bank[k].shape) != shape or not torch.isfinite(bank[k]).all()
            for k,shape in shapes.items()):
        raise ValueError('Need at least 16 finite, complete 14-servo handoff states')
    if not torch.allclose(bank['quat'].norm(dim=-1), torch.ones(n,device=device), atol=1e-4):
        raise ValueError('Bank quaternions must be normalized')
    return bank


def evaluate(env, wrapped, actor, bank, command_scale=1.):
    """Each unseen pose crosses all commands. Any reset invalidates the original trial.

    ``command_scale`` multiplies the seven commands; pass/fail thresholds are unchanged."""
    n = len(bank['height'])
    commands = torch.tensor([[0.,0,0],[.2,0,0],[-.1,0,0],[0,.1,0],
                             [0,-.1,0],[0,0,.5],[0,0,-.5]], device=env.device)*command_scale
    env.recovery_walk_eval_command = commands.repeat_interleave(n,dim=0)
    env.recovery_walk_bank_indices = torch.arange(n,device=env.device).repeat(len(commands))
    env.reset()
    obs = wrapped.get_observations()
    alive = torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
    upright3 = alive.clone(); upright6 = alive.clone()
    travel = torch.zeros(env.num_envs,3,device=env.device)
    restart = torch.zeros_like(travel)
    stop_speed = torch.zeros(env.num_envs,device=env.device)
    stop_yaw = torch.zeros_like(stop_speed)
    numerical = torch.zeros_like(alive)
    actor.eval()
    with torch.inference_mode():
        for step in range(300):
            obs, _, done, _ = wrapped.step(actor(obs))
            d = env.scene['robot'].data
            alive &= ~done.bool()
            # No timeout at 6 s: evaluation config uses a 7 s episode.
            up = -d.projected_gravity_b[:,2]
            height = d.root_link_pos_w[:,2] - env.scene.terrain.env_origins[:,2]
            upright = alive & (up >= .8660254) & (height >= .095)
            upright6 &= upright
            if step < 150:
                upright3 &= upright
                travel += torch.cat((d.root_link_lin_vel_b[:,:2],d.root_link_ang_vel_b[:,2:3]),-1)*env.step_dt*alive[:,None]
            if 175 <= step < 200:
                stop_speed += d.root_link_lin_vel_b[:,:2].norm(dim=-1)/25
                stop_yaw += d.root_link_ang_vel_b[:,2].abs()/25
            if step >= 200:
                restart += torch.cat((d.root_link_lin_vel_b[:,:2],d.root_link_ang_vel_b[:,2:3]),-1)*env.step_dt*alive[:,None]
            numerical |= env.termination_manager.get_term('nan_state').bool()
    reports = {}
    names = ['idle','forward','backward','left','right','turn_left','turn_right']
    for i,name in enumerate(names):
        sl = slice(i*n,(i+1)*n)
        axis = 0 if i < 3 else 1 if i < 5 else 2
        sign = -1 if i in (2,4,6) else 1
        threshold = .2 if i == 1 else .1 if i < 5 else .6
        moving = travel[sl,axis]*sign >= threshold
        if i == 0:
            moving = travel[sl,:2].norm(dim=-1) <= .1
        restarting=restart[sl,axis]*sign >= threshold*2/3
        if i==0: restarting=restart[sl,:2].norm(dim=-1)<=.1
        stopped=(stop_speed[sl]<=.03)&(stop_yaw[sl]<=.15)
        reports[name] = {
            'trials':n, 'upright_3s':int(upright3[sl].sum()),
            'upright_6s':int(upright6[sl].sum()),
            'upright_and_command_3s':int((upright3[sl]&moving).sum()),
            'signed_travel_3s_mean':float((travel[sl,axis]*sign).mean()),
            'signed_restart_2s_mean':float((restart[sl,axis]*sign).mean()),
            'stop_linear_speed_mean':float(stop_speed[sl].mean()),
            'stop_yaw_speed_mean':float(stop_yaw[sl].mean()),
            'upright_and_stopped':int((upright6[sl]&stopped).sum()),
            'upright_and_restarted':int((upright6[sl]&restarting).sum()),
            'complete_sequence':int((upright6[sl]&moving&stopped&restarting).sum()),
            'numerically_invalid':int(numerical[sl].sum()),
        }
    return {'commands':reports, 'movement_gate_passed':all(
        r['upright_and_command_3s']/r['trials'] >= .9 and r['numerically_invalid']==0 for r in reports.values()),
        'passed':all(r['complete_sequence']/r['trials'] >= .9 and r['numerically_invalid']==0 for r in reports.values()),
        'gate':'Each command: >=90% upright 6s, initial movement, stop (mean <=0.03 m/s and <=0.15 rad/s at 3.5-4s), and restart; idle drift <=10cm',
        'note':'Signed travel integrates body-frame velocity; reset trials always fail. Pose banks omit actuator/delay caches.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-policy',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--eval-bank',type=Path)
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--critic-warmup',type=int,default=0)
    p.add_argument('--warmup-lr',type=float,default=1e-3)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--initial-action-std-floor',type=float,default=0.,help='One-time exploration floor; actor mean and normalizer unchanged')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-seed',type=int,default=2026)
    p.add_argument('--run-name',default='smoke')
    p.add_argument('--evaluate',type=Path,help='Write held-out evaluation JSON instead of training')
    p.add_argument('--play',action='store_true',help='Local CUDA native viewer with measured-pose resets')
    args = p.parse_args()
    if min(args.num_envs,args.iterations,args.eval_every) <= 0:
        p.error('Counts must be positive')
    if args.initial_action_std_floor<0 or not math.isfinite(args.initial_action_std_floor):
        p.error('Exploration floor must be finite and nonnegative')
    if args.initial_action_std_floor and (args.evaluate or args.play):
        p.error('Exploration floor applies only to training')
    configure_torch_backends()
    torch.manual_seed(args.seed)
    bank = load_bank(args.bank,'cuda:0')
    if args.eval_bank:
        other = load_bank(args.eval_bank,'cpu')
        train_states = torch.cat([v.flatten(1) if v.ndim>1 else v[:,None] for v in bank.values()],-1).cpu()
        eval_states = torch.cat([other[k].flatten(1) if other[k].ndim>1 else other[k][:,None] for k in bank],-1)
        if (train_states[:,None,:] == eval_states[None,:,:]).all(-1).any():
            raise ValueError('Training and held-out banks overlap')
    if args.iterations > 5 and not args.eval_bank and not (args.evaluate or args.play):
        p.error('Long runs require --eval-bank')
    cfg, agent = load_env_cfg(TASK), load_rl_cfg(TASK)
    cfg.seed = agent.seed = args.seed
    cfg.scene.num_envs = len(bank['height'])*7 if args.evaluate else args.num_envs
    cfg.sim.nan_guard.enabled = args.iterations <= 5 and not args.play
    if args.evaluate:
        cfg.episode_length_s = 7.
        cfg.events['recovery_start'].params['start_probability'] = 1.
    agent.logger = 'tensorboard'; agent.save_interval = 50
    agent.run_name = args.run_name
    out = Path('logs/rsl_rl/recovery_walk')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
    out.mkdir(parents=True)
    dump_yaml(out/'params/env.yaml',asdict(cfg)); dump_yaml(out/'params/agent.yaml',asdict(agent))
    metadata = {**vars(args), 'recipe':RECIPE,
        'bank_sha256':hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        'eval_bank_sha256':hashlib.sha256(args.eval_bank.read_bytes()).hexdigest() if args.eval_bank else None,
        'actor_only_initialization':True}
    (out/'initialization.json').write_text(json.dumps(metadata,default=str,indent=2))
    shutil.copyfile(args.bank,out/'recovery_starts.pt')
    if args.eval_bank: shutil.copyfile(args.eval_bank,out/'heldout_starts.pt')
    env = ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        env.arena_return_starts = bank
        wrapped = RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapped,asdict(agent),str(out),device='cuda:0')
        runner.load(str(args.from_policy),load_cfg={'actor':True},map_location='cuda:0')
        if args.initial_action_std_floor:
            distribution=runner.alg.actor.distribution
            if distribution.std_type!='scalar': raise ValueError('Exploration floor requires scalar std parameterization')
            with torch.no_grad(): distribution.std_param.clamp_(min=args.initial_action_std_floor)
            print(f'Initial exploration std: {distribution.std_param.detach().cpu().tolist()}',flush=True)
        env.common_step_counter = 0; runner.current_learning_iteration = 0
        env.reset()
        assert wrapped.get_observations()['actor'].shape[-1] == 61
        print(f'Training output: {out}',flush=True)
        if args.play:
            from mjlab.viewer import NativeMujocoViewer
            policy = runner.get_inference_policy(device='cuda:0')
            NativeMujocoViewer(wrapped,policy).run()
            return
        if args.evaluate:
            report = evaluate(env,wrapped,runner.alg.actor,bank)
            args.evaluate.parent.mkdir(parents=True,exist_ok=True)
            args.evaluate.write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            return
        def check(checkpoint, updates):
            if not args.eval_bank: return
            report = out/f'eval_{updates}.json'
            cmd = [sys.executable,'-u',__file__,'--from-policy',str(checkpoint),
                   '--bank',str(args.eval_bank),'--seed',str(args.eval_seed),'--evaluate',str(report)]
            with (out/f'eval_{updates}.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'Held-out evaluation {updates}: {report.read_text()}',flush=True)
        check(args.from_policy,0)
        if args.critic_warmup: critic_warmup(runner,env,args)
        updates = 0
        while updates < args.iterations:
            count = min(args.eval_every,args.iterations-updates)
            runner.learn(num_learning_iterations=count,init_at_random_ep_len=False)
            if not all(torch.isfinite(v).all() for v in runner.alg.actor.state_dict().values()):
                raise RuntimeError('Nonfinite actor state; refusing to continue')
            updates += count
            checkpoint = out/f'model_{runner.current_learning_iteration}.pt'
            runner.save(str(checkpoint))
            check(checkpoint,updates)
            (out/'status.json').write_text(json.dumps({'actor_updates':updates,'checkpoint':str(checkpoint),
                'complete':updates==args.iterations},indent=2))
            runner.current_learning_iteration += 1
    finally:
        env.close()


if __name__ == '__main__':
    main()
