"""Paired physical get-up and post-handoff battery; no training or policy edits.

Each scenario runs 64 orientation-balanced trials per controller: normal football
handoff versus retaining the recovery expert after its first stable detection.
Near-board and contact-start cases are stress tests, reported separately.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import train_mappo as tm


def summarize(data, dt=.02):
    reports={}
    for mode in range(2):
        sl=slice(mode*64,(mode+1)*64)
        z={k:v[:,sl] for k,v in data.items()}
        first=[]; stable3=0; refall=0; moved=0; eligible=0
        for e in range(64):
            valid=~z['invalid'][:,e]
            hs=np.flatnonzero(z['handoff'][:,e]&valid)
            if not len(hs): continue
            t=int(hs[0]); first.append((t+1)*dt)
            if t+151>len(valid) or not valid[t:t+151].all():continue
            eligible+=1
            window=z['upright'][t+1:t+151,e]
            stable3+=int(window.all() and (t+1)*dt<=5.)
            refall+=int(z['entered'][t+1:t+151,e].any())
            moved+=int(window.mean()>=.9 and np.linalg.norm(z['position'][t+150,e,:2]-z['position'][t,e,:2])>=.2)
        reports['football_handoff' if mode==0 else 'keep_recovery']={
            'trials':64,'first_stable_within_5s':sum(t<=5 for t in first),
            'first_stable_median_s':float(np.median(first)) if first else None,
            'eligible_3s_followup':eligible,'stand_within_5s_and_upright_next_3s':stable3,
            'refall_within_3s':refall,'upright_and_move_20cm_next_3s':moved,
            'numerically_invalid_trials':int(z['invalid'].any(0).sum()),
            'upright_fraction':float(z['upright'][~z['invalid']].mean())}
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',choices=['isolated','board','contact','collision'],required=True)
    p.add_argument('--handoff-mode',choices=['normal','zero','ramp'],default='normal')
    p.add_argument('--seconds',type=float,default=9.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--attacker',type=Path,help='Candidate motor; orange keeps the original')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);meta=ck['metadata']
    tm.configure_torch_backends();torch.manual_seed(a.seed)
    cfg=tm.load_env_cfg(tm.TASK);cfg.scene.num_envs=128;cfg.seed=a.seed
    cfg.commands['twist'].track_training_signals=False
    cfg.episode_length_s=12.;cfg.curriculum={};cfg.metrics={}
    cfg.rewards={'mappo_team':tm.RewardTermCfg(func=tm.mdp.arena_mappo_team_reward,weight=1.,params={'gamma':tm.GAMMA})};cfg.terminations={k:v for k,v in cfg.terminations.items() if 'nan' in k}
    if a.scenario=='collision':
        from mjlab.sensor import ContactMatch,ContactSensorCfg
        cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='recovery_collision',
            primary=ContactMatch(mode='body',pattern='.*',entity='robot'),
            secondary=ContactMatch(mode='body',pattern='.*',entity='opponent'),
            fields=('force',),reduce='netforce',num_slots=1,history_length=cfg.decimation))
    env=tm.ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        expert=tm.frozen_motor(meta['recovery'],'cuda:0')
        env.arena_recovery_policy=expert;env.arena_recovery=tm.mdp.ArenaRecovery(env)
        arena=tm.TacticalArena(env,tm.frozen_motor(a.attacker or meta['attacker'],'cuda:0'),tm.frozen_motor(meta['keeper'],'cuda:0'))
        if a.attacker:
            env.arena_opponent_policy=tm.frozen_motor(meta['attacker'],'cuda:0')
        recovery=env.arena_recovery; latched=torch.zeros(128,dtype=torch.bool,device=env.device)
        bank=[]
        football_time=torch.zeros(128,device=env.device)
        normal_select=recovery.select
        def select(name,obs,action):
            output=normal_select(name,obs,action)
            if name=='robot':
                ids=(recovery.succeeded[:,0]&~latched&(torch.arange(128,device=env.device)<64)).nonzero().flatten()
                if len(ids):
                    servo=tm.mdp._servo_joint_ids(env,robot)
                    bank.append({'height':(d.root_link_pos_w[ids,2]-origin[ids,2]).clone(),
                        'quat':d.root_link_quat_w[ids].clone(),
                        'velocity':torch.cat((d.root_link_lin_vel_w[ids],d.root_link_ang_vel_w[ids]),-1).clone(),
                        'joint_pos':d.joint_pos[ids][:,servo].clone(),
                        'joint_vel':d.joint_vel[ids][:,servo].clone(),
                        'last_action':obs[ids,34:48].clone()})
                latched.logical_or_(recovery.succeeded[:,0])
                football_time.copy_(torch.where(recovery.active[:,0]|recovery.succeeded[:,0],0.,football_time+env.step_dt))
                if a.handoff_mode!='normal':
                    adjusted=obs.clone()
                    scale=torch.zeros_like(football_time) if a.handoff_mode=='zero' else (football_time/1.).clamp(0,1)
                    # Change only the handoff, keeping pre-fall actions identical.
                    scale=torch.where(latched,scale,1.)
                    adjusted[:,48:51]*=scale[:,None]
                    football=arena.motor({'actor':adjusted})
                    output=torch.where(recovery.active[:,0,None],output,football)
                commandless=obs.clone();commandless[:,48:]=0.
                keep=latched & (torch.arange(128,device=env.device)>=64)
                output=torch.where(keep[:,None],expert({'actor':commandless}),output)
            return output
        recovery.select=select
        origin=env.scene.terrain.env_origins;robot=env.scene['robot'];d=robot.data
        kinds=torch.arange(128,device=env.device)%4
        q=torch.tensor([[1.,0.,1.,0.],[1.,0.,-1.,0.],[1.,1.,0.,0.],[1.,-1.,0.,0.]],device=env.device)/(2**.5)
        pose=torch.zeros(128,7,device=env.device);pose[:,:3]=origin+torch.tensor([-.45,.85 if a.scenario=='board' else 0.,.20],device=env.device);pose[:,3:]=q[kinds]
        robot.write_root_link_pose_to_sim(pose);robot.write_root_link_velocity_to_sim(torch.zeros(128,6,device=env.device))
        # Match initial joint states within each paired trial; DR remains independently sampled.
        robot.write_joint_state_to_sim(d.default_joint_pos[:64].repeat(2,1),torch.zeros_like(d.joint_vel))
        parked={}
        for i,name in enumerate(tm.mdp.ARENA_TEAM_PLAYERS):
            if name=='robot':continue
            entity=env.scene[name];fixed=torch.zeros_like(pose);fixed[:,:3]=origin+torch.tensor([0.,2.+i*.4,.125],device=env.device);fixed[:,3]=1.
            if a.scenario in ('contact','collision') and name=='opponent':
                fixed[:,:3]=origin+torch.tensor([-.10 if a.scenario=='collision' else -.20,0.,.125],device=env.device)
                fixed[:,3]=0.;fixed[:,6]=1.
                velocity=torch.zeros(128,6,device=env.device)
                if a.scenario=='collision':velocity[:,0]=-.4
                entity.write_root_link_pose_to_sim(fixed);entity.write_root_link_velocity_to_sim(velocity)
            else:parked[name]=fixed
        def park():
            for name,fixed in parked.items():
                entity=env.scene[name];entity.write_root_link_pose_to_sim(fixed)
                entity.write_root_link_velocity_to_sim(torch.zeros(128,6,device=env.device))
                entity.write_joint_state_to_sim(entity.data.default_joint_pos,torch.zeros_like(entity.data.joint_vel))
        ball=env.scene['ball'];bp=torch.zeros_like(pose);bp[:,:3]=origin+torch.tensor([.5,0.,.035],device=env.device);bp[:,3]=1
        ball.write_root_link_pose_to_sim(bp);ball.write_root_link_velocity_to_sim(torch.zeros(128,6,device=env.device))
        park();env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
        frames=[];renders=[];invalid=torch.zeros(128,dtype=torch.bool,device=env.device)
        if a.scenario=='collision':
            navigation=tm.mdp.arena_tactical_commands
            def incoming_commands(e,skills):
                commands=navigation(e,skills)
                if len(frames)<150:
                    commands['opponent']=torch.tensor([.35,0.,0.],device=e.device).repeat(e.num_envs,1)
                return commands
            tm.mdp.arena_tactical_commands=incoming_commands
        original_step=env.step
        def step(action):
            park();result=original_step(action)
            invalid.logical_or_(result[2])
            state={'upright':((d.root_link_pos_w[:,2]-origin[:,2]>=.095)&(-d.projected_gravity_b[:,2]>=.8660254)),
                'entered':recovery.entered[:,0],'handoff':recovery.succeeded[:,0],
                'recovering':recovery.active[:,0],'invalid':invalid,
                'position':d.root_link_pos_w-origin}
            if a.scenario=='collision':
                state['collision']=torch.nan_to_num(env.scene.sensors['recovery_collision'].data.force_history).abs().flatten(1).sum(-1)>1e-6
            frames.append({k:v.clone() for k,v in state.items()})
            if (len(frames)-1)%5==0:
                ids=[0,1,2,3,64,65,66,67]
                renders.append({k:getattr(env.sim.data,k)[ids].clone() for k in ('qpos','mocap_pos','mocap_quat')})
            return result
        env.step=step;began=time.monotonic()
        with torch.inference_mode():
            while len(frames)<round(a.seconds/env.step_dt):
                arena.step(torch.zeros(128,2,dtype=torch.long,device=env.device))
                if len(frames)%100==0:print('PROGRESS',len(frames),round(time.monotonic()-began),flush=True)
        data={k:torch.stack([f[k] for f in frames]).cpu().numpy() for k in frames[0]}
        if bank:
            torch.save({k:torch.cat([b[k] for b in bank]).cpu() for k in bank[0]},a.output/'handoff_states.pt')
        np.savez_compressed(a.output/'traces.npz',**data)
        np.savez_compressed(a.output/'render.npz',**{k:torch.stack([f[k] for f in renders]).cpu().numpy() for k in renders[0]})
        mujoco.mj_saveModel(env.sim.mj_model,str(a.output/'render.mjb'))
        report={'scenario':a.scenario,'seed':a.seed,'handoff_mode':a.handoff_mode,'candidate_attacker':str(a.attacker) if a.attacker else None,'checkpoint':str(a.checkpoint),'motors':{k:str(meta[k]) for k in ('attacker','keeper','recovery')},'results':summarize(data)}
        if 'collision' in data:
            for i,key in enumerate(('football_handoff','keep_recovery')):
                report['results'][key]['measured_collision_trials']=int(data['collision'][:,i*64:(i+1)*64].any(0).sum())
        (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
