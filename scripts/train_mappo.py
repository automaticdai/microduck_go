"""Stage 1 MAPPO: two tactical attackers, four frozen rivals/keepers, frozen motor skills."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp
from mjlab_microduck.mappo import TacticalMAPPO,generalized_advantages,update

TASK='Mjlab-FootballArena-3v3-Flat-MicroDuck'
GAMMA=.997
MOTOR_STEPS=5


def frozen_motor(path,device):
    cfg=asdict(load_rl_cfg(TASK).actor)
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61,device=device)},batch_size=[1]),
        {'actor':['actor']},'actor',14,hidden_dims=cfg['hidden_dims'],
        activation=cfg['activation'],obs_normalization=True,distribution_cfg=cfg['distribution_cfg']).to(device)
    actor.load_state_dict(torch.load(path,map_location=device,weights_only=False)['actor_state_dict'])
    return actor.eval().requires_grad_(False)


class TacticalArena:
    def __init__(self,env,attacker,keeper):
        self.env=env
        self.motor=attacker
        env.arena_opponent_policy=attacker
        env.arena_teammate_policy=attacker
        env.arena_defender_policy=keeper
        self.previous_velocity=torch.zeros(env.num_envs,14,device=env.device)
        env.reset()

    def observations(self):
        return mdp.arena_tactical_observations(self.env)

    @torch.no_grad()
    def step(self,skills):
        env=self.env
        reward=torch.zeros(env.num_envs,device=env.device)
        diagnostics={}
        valid=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        eval_mask=getattr(self,'evaluation_mask',torch.ones(env.num_envs,device=env.device))
        for substep in range(MOTOR_STEPS):
            env.arena_tactical_commands=mdp.arena_tactical_commands(env,skills)
            # Both blue motor actors use the same clean 61D convention and frozen normalizer.
            asset=env.scene['robot']; d=asset.data; ids=mdp._servo_joint_ids(env,asset)
            obs=torch.cat((d.root_link_ang_vel_b,d.projected_gravity_b,
                d.joint_pos[:,ids]-d.default_joint_pos[:,ids],self.previous_velocity,
                env.action_manager.get_term('joint_pos').raw_action,
                env.arena_tactical_commands['robot'],torch.zeros(env.num_envs,10,device=env.device)),-1)
            if obs.shape[-1] != 61 or not torch.isfinite(obs).all():
                raise RuntimeError('Motor observation contract or numerical failure')
            self.previous_velocity.copy_(d.joint_vel[:,ids])
            length,_=mdp.arena_field(env)
            env.arena_mappo_previous_potential=((env.scene['ball'].data.root_link_pos_w[:,0]
                -env.scene.terrain.env_origins[:,0])/length).clamp(-1,1).clone()
            motor_action=mdp.arena_recovery_action(env,'robot',obs,self.motor({'actor':obs}))
            _,r,terminated,timeout,_=env.step(motor_action)
            done=terminated|timeout
            if not torch.isfinite(r).all():
                raise RuntimeError('Non-finite team reward')
            reward+=GAMMA**substep*r
            for key,value in env.arena_mappo_diagnostics.items():
                diagnostics[key]=diagnostics.get(key,0.)+(value*eval_mask).sum()
            recovery=getattr(env,'arena_recovery',None)
            if recovery is not None:
                for key,value in (('recovery_entries',recovery.entered),
                                  ('recovery_successes',recovery.succeeded),
                                  ('recovering_players',recovery.active)):
                    diagnostics[key]=diagnostics.get(key,0.)+(value.float()*eval_mask[:,None]).sum()
            nan_mask=torch.stack([env.termination_manager.get_term(key)
                for key in env.termination_manager.active_terms if 'nan' in key]).any(0)
            valid &= ~nan_mask
            diagnostics['numerical_resets']=diagnostics.get('numerical_resets',0.)+nan_mask.float().sum()
            if nan_mask.any():
                # ManagerBasedRlEnv has already reset these worlds. Check raw physics,
                # not sanitized tactical observations, before accepting the recovery.
                remaining=torch.stack([mdp.robot_state_is_nan(env,
                    asset_cfg=env.termination_manager.get_term_cfg(key).params.get(
                        'asset_cfg',mdp._DEFAULT_ASSET_CFG),
                    sensor_names=env.termination_manager.get_term_cfg(key).params.get('sensor_names',()))
                    for key in env.termination_manager.active_terms if 'nan' in key]).any(0)
                if remaining[nan_mask].any() or not done[nan_mask].all():
                    raise RuntimeError('Numerical state did not recover after arena reset')
                # Never train on a fabricated reward for the corrupt transition.
                reward[nan_mask]=0
                if getattr(env,'arena_strict_numerics',False):
                    raise RuntimeError('Strict numerical guard; see numerics/events.jsonl')
            self.previous_velocity[done]=0
            # End this shared tactical interval at any round boundary: no cross-reset GAE,
            # and no unrecorded motor steps in the next episode.
            if done.any():
                break
        count=substep+1
        diagnostics=dict(zip(diagnostics,torch.stack(list(diagnostics.values())).cpu().tolist()))
        diagnostics['motor_steps']=count
        self.last_valid=valid
        return self.observations(),reward[:,None].expand(-1,2),done,torch.full_like(reward,GAMMA**count),diagnostics


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacker',type=Path,required=True)
    parser.add_argument('--keeper',type=Path,required=True)
    parser.add_argument('--recovery',type=Path,help='Frozen 61D stand-up expert for all six players')
    parser.add_argument('--upgrade-recovery',action='store_true',
                        help='Explicitly migrate a v2 checkpoint to automatic motor recovery')
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--iterations',type=int,default=5)
    parser.add_argument('--rollout-steps',type=int,default=24)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--critic-warmup',type=int,default=0)
    parser.add_argument('--eval-every',type=int,default=0)
    parser.add_argument('--run-name',default='stage1-smoke')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--strict-numerics',action='store_true',
                        help='Stop on even a recovered numerical reset (use for smoke tests)')
    parser.add_argument('--evaluate',choices=['tracker','policy'])
    args=parser.parse_args()
    if min(args.num_envs,args.iterations,args.rollout_steps) < 1:
        parser.error('Counts must be positive')
    if min(args.critic_warmup,args.eval_every) < 0:
        parser.error('Warm-up and evaluation intervals must be nonnegative')
    if args.evaluate == 'policy' and not args.resume:
        parser.error('--evaluate policy requires --resume')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no training has been launched')
    if args.upgrade_recovery and (not args.recovery or not args.resume):
        parser.error('--upgrade-recovery requires --recovery and --resume')
    configure_torch_backends(); torch.manual_seed(args.seed)
    cfg=load_env_cfg(TASK)
    cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed
    cfg.sim.nan_guard.enabled=args.iterations <= 5
    cfg.commands['twist'].track_training_signals=False
    cfg.rewards={'mappo_team':RewardTermCfg(func=mdp.arena_mappo_team_reward,weight=1.,params={'gamma':GAMMA})}
    cfg.curriculum={}; cfg.metrics={}
    for name,term in cfg.terminations.items():
        if 'nan' in name:
            if term.func is not mdp.robot_state_is_nan:
                raise RuntimeError(f'Unsupported numerical guard: {name}')
            term.func=mdp.arena_numerical_guard
    out=Path('logs/rsl_rl/football_mappo')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
    out.mkdir(parents=True,exist_ok=False)
    metadata={**vars(args),'recipe':('tactical-mappo-stage1-v3-recovery' if args.recovery else 'tactical-mappo-stage1-v2'),'pitch':[2*cfg.commands['twist'].pitch_half_length,2*cfg.commands['twist'].pitch_half_width],
        'round_seconds':cfg.episode_length_s,'motor_hz':50,'tactical_hz':10,'skills':mdp.MAPPO_SKILLS,
        'fall_cost_per_second':mdp.MAPPO_FALL_COST_PER_SECOND,
        'crowding_cost_per_second':mdp.MAPPO_CROWDING_COST_PER_SECOND,
        'attacker_sha256':fingerprint(args.attacker),'keeper_sha256':fingerprint(args.keeper)}
    if args.recovery:
        metadata.update(recovery_sha256=fingerprint(args.recovery),recovery_config=mdp.ARENA_RECOVERY_CONFIG)
    (out/'initialization.json').write_text(json.dumps(metadata,default=str,indent=2))
    print(f'Training output: {out}',flush=True)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    env.arena_numerical_report_dir=out/'numerics'
    env.arena_strict_numerics=args.strict_numerics
    writer=SummaryWriter(str(out))
    try:
        attacker=frozen_motor(args.attacker,'cuda:0'); keeper=frozen_motor(args.keeper,'cuda:0')
        motors=[attacker,keeper]
        if args.recovery:
            env.arena_recovery_policy=frozen_motor(args.recovery,'cuda:0')
            env.arena_recovery=mdp.ArenaRecovery(env)
            motors.append(env.arena_recovery_policy)
        frozen_states=[{key:value.clone() for key,value in motor.state_dict().items()}
                       for motor in motors]
        arena=TacticalArena(env,attacker,keeper)
        actor_obs,critic_obs=arena.observations()
        model=TacticalMAPPO(actor_obs.shape[-1],critic_obs.shape[-1]).to('cuda:0')
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        start=0
        if args.resume:
            checkpoint=torch.load(args.resume,map_location='cuda:0',weights_only=False)
            keys=['pitch','skills','attacker_sha256','keeper_sha256']
            migrating=(args.upgrade_recovery and checkpoint['metadata']['recipe']=='tactical-mappo-stage1-v2')
            if not migrating:
                keys.append('recipe')
                if args.recovery: keys.extend(('recovery_sha256','recovery_config'))
            for key in keys:
                if checkpoint['metadata'][key] != metadata[key]:
                    raise ValueError(f'Incompatible checkpoint {key}')
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            start=checkpoint['iteration']+1
            if migrating:
                print('Upgrading v2 to automatic recovery; tactical weights and optimizer retained.',flush=True)
        if args.evaluate:
            # First round per arena; every arena gets exactly one scored trial.
            remaining=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            scored=conceded=rounds=numerical_rounds=0; standing=steps=0.
            recovery_entries=recovery_successes=0.
            while remaining.any():
                with torch.no_grad():
                    skills=(model.actor(actor_obs).argmax(-1) if args.evaluate == 'policy'
                            else torch.zeros(env.num_envs,2,dtype=torch.long,device=env.device))
                    arena.evaluation_mask=remaining.float()
                    (actor_obs,critic_obs),_,done,_,diag=arena.step(skills)
                ended=done&remaining
                scored+=int(((env.arena_goal_result>0)&ended).sum())
                conceded+=int(((env.arena_goal_result<0)&ended).sum())
                numerical_rounds+=int((ended & ~arena.last_valid).sum())
                rounds+=int(ended.sum())
                recovery_entries+=diag.get('recovery_entries',0.)
                recovery_successes+=diag.get('recovery_successes',0.)
                standing+=diag['standing']; steps+=diag['motor_steps']*int(remaining.sum())
                remaining &= ~done
            report=dict(rounds=rounds,scored=scored,conceded=conceded,
                        standing_fraction=standing/steps,mode=args.evaluate,numerical_rounds=numerical_rounds,
                        recovery_entries=recovery_entries,recovery_successes=recovery_successes)
            (out/'evaluation.json').write_text(json.dumps(report,indent=2)); print(report,flush=True)
            return
        def save_checkpoint(path,iteration):
            temporary=path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,
                metadata=metadata,actor_dim=actor_obs.shape[-1],critic_dim=critic_obs.shape[-1]),temporary)
            os.replace(temporary,path)
        # Rolling restart point exists before physics, then advances after every update.
        save_checkpoint(out/'latest.pt',start-1)
        for iteration in range(start,start+args.iterations):
            model.actor.requires_grad_(iteration >= args.critic_warmup)
            began=time.monotonic(); storage=[[] for _ in range(8)]; totals={}; discount_storage=[]
            for _ in range(args.rollout_steps):
                with torch.no_grad():
                    distribution=model.distribution(actor_obs)
                    actions=distribution.sample(); logp=distribution.log_prob(actions)
                    values=model.value(critic_obs)
                    next_obs,rewards,dones,discounts,diag=arena.step(actions)
                for bucket,value in zip(storage,(actor_obs,critic_obs,actions,logp,values,rewards,dones,arena.last_valid)):
                    bucket.append(value.clone())
                discount_storage.append(discounts)
                actor_obs,critic_obs=next_obs
                for key,value in diag.items(): totals[key]=totals.get(key,0.)+value
            obs,state,actions,logp,values,rewards,dones,valid=map(torch.stack,storage)
            with torch.no_grad():
                advantages,returns=generalized_advantages(rewards,values,dones,
                    torch.stack(discount_storage),model.value(critic_obs),valid=valid)
            metrics=update(model,optimizer,(obs,state,actions,logp,values,advantages,returns),valid=valid)
            save_checkpoint(out/'latest.pt',iteration)
            samples=totals['motor_steps']*env.num_envs
            metrics.update(iteration=iteration,critic_warmup=iteration < args.critic_warmup,seconds=time.monotonic()-began,
                reward=float(rewards.mean()),scored=totals['scored'],conceded=totals['conceded'],
                standing_fraction=totals['standing']/samples,crowding_fraction=totals['crowding']/samples,
                fall_cost=totals['fall_cost'],crowding_cost=totals['crowding_cost'],
                motor_steps=totals['motor_steps'],episodes=int(dones.sum()),
                numerical_resets=totals['numerical_resets'],invalid_transitions=int((~valid).sum()))
            if args.recovery:
                metrics.update(recovery_entries=totals['recovery_entries'],
                    recovery_successes=totals['recovery_successes'],
                    recovering_fraction=totals['recovering_players']/(samples*len(mdp.ARENA_TEAM_PLAYERS)))
            for index,name in enumerate(mdp.MAPPO_SKILLS):
                metrics[f'skill_{name}_fraction']=float((actions==index).float().mean())
            print(json.dumps(metrics),flush=True)
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
            for key,value in metrics.items(): writer.add_scalar(key,value,iteration)
            evaluate=args.eval_every > 0 and (iteration+1) % args.eval_every == 0
            if iteration % 50 == 0 or iteration == start+args.iterations-1 or evaluate:
                save_checkpoint(out/f'model_{iteration}.pt',iteration)
            if evaluate:
                command=[sys.executable,'-u',__file__,'--attacker',str(args.attacker),
                    '--keeper',str(args.keeper),'--resume',str(out/f'model_{iteration}.pt'),
                    '--evaluate','policy','--num-envs','64','--run-name',f'{args.run_name}-eval-{iteration}']
                if args.recovery: command.extend(['--recovery',str(args.recovery)])
                with (out/f'eval_{iteration}.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        for motor,before in zip(motors,frozen_states):
            if any(not torch.equal(before[key],value) for key,value in motor.state_dict().items()):
                raise RuntimeError('Frozen motor weights or normalizer changed')
        # Separate tactical ONNX contract: observation -> five skill logits. Motor export
        # remains scripts/export.py and includes its trained observation normalizer.
        model.cpu().eval()
        torch.onnx.export(model.actor,torch.zeros(1,actor_obs.shape[-1]),str(out/'tactical.onnx'),
            input_names=['tactical_observation'],output_names=['skill_logits'],opset_version=17,dynamo=False)
        import onnxruntime as ort
        session=ort.InferenceSession(str(out/'tactical.onnx'),providers=['CPUExecutionProvider'])
        sample=actor_obs[:2].flatten(0,1).cpu()
        for row in sample:
            expected=model.actor(row[None]).detach().numpy()
            actual=session.run(None,{'tactical_observation':row[None].numpy()})[0]
            import numpy as np
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
        print('Tactical ONNX parity passed; motor policies remained frozen.',flush=True)
    finally:
        writer.close(); env.close()


if __name__ == '__main__':
    main()
