"""Local CUDA comparison: paired videos and reward mass on held-out endings."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from train_recovery_walk import load_bank, evaluate, TASK
from train_mappo import frozen_motor
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--command-scale',type=float,default=1.,help='Multiply battery commands; thresholds unchanged')
    p.add_argument('--no-video',action='store_true',help='Measure rewards without rendering')
    command_names=['idle','forward','backward','left','right','turn_left','turn_right']
    p.add_argument('--video-commands',nargs=2,choices=command_names,default=['forward','left'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    configure_torch_backends();torch.manual_seed(2026)
    bank=load_bank(a.bank,'cuda:0');n=len(bank['height'])
    cfg=load_env_cfg(TASK);cfg.seed=2026;cfg.scene.num_envs=n*7
    cfg.episode_length_s=7.;cfg.events['recovery_start'].params['start_probability']=1.
    cfg.viewer.width=480;cfg.viewer.height=360;cfg.viewer.distance=.85
    cfg.viewer.elevation=-20.;cfg.viewer.azimuth=135.;cfg.viewer.max_extra_envs=0
    env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array')
    try:
        env.arena_return_starts=bank
        wrapped=RslRlVecEnvWrapper(env,clip_actions=load_rl_cfg(TASK).clip_actions)
        actor=frozen_motor(a.checkpoint,'cuda:0')
        alive=torch.ones(n*7,device=env.device,dtype=torch.bool)
        rewards=torch.zeros(n*7,len(env.reward_manager.active_terms),device=env.device)
        counts=torch.zeros(n*7,device=env.device)
        actor_commands=torch.zeros(2,n*7,3,device=env.device)
        original=wrapped.step;step=0;snapshots=[]
        writer=None if a.no_video else imageio.get_writer(a.output/'comparison.mp4',fps=25)
        def record(action):
            nonlocal step
            if step<150: actor_commands[0].add_(env.obs_buf['actor'][:,48:51]/150)
            if 175<=step<200: actor_commands[1].add_(env.obs_buf['actor'][:,48:51]/25)
            before=alive.clone()
            result=original(action)
            alive.logical_and_(~result[2].bool())
            if step<150:
                rewards.add_(env.reward_manager._step_reward*before[:,None]);counts.add_(before)
            if writer is not None and step%2==0:
                panels=[]
                for name in a.video_commands:
                    i=command_names.index(name)
                    cfg.viewer.env_idx=i*n
                    frame=Image.fromarray(env.render())
                    draw=ImageDraw.Draw(frame);draw.rectangle((0,0,480,48),fill='white')
                    phase='STOP' if 150<=step<200 else 'MOVE'
                    draw.text((8,5),f'{a.label} | {name} | {phase}',fill='black')
                    draw.text((8,23),f't={(step+1)*.02:.2f}s | heldout pose 0 | alive={bool(alive[i*n])}',fill='black')
                    panels.append(np.asarray(frame))
                combined=np.concatenate(panels,axis=1);writer.append_data(combined)
                if step in (0,48,98,148,198,248,298): snapshots.append(Image.fromarray(combined))
            step+=1
            return result
        wrapped.step=record
        try: report=evaluate(env,wrapped,actor,bank,a.command_scale)
        finally:
            if writer is not None: writer.close()
        names=list(report['commands'])
        report['mean_weighted_reward_first_3s_alive']={name:{term:float(rewards[i*n:(i+1)*n,j].sum()/counts[i*n:(i+1)*n].sum().clamp_min(1))
            for j,term in enumerate(env.reward_manager.active_terms)} for i,name in enumerate(names)}
        report['actor_command_mean']={phase:{name:actor_commands[j,i*n:(i+1)*n].mean(0).tolist() for i,name in enumerate(command_names)} for j,phase in enumerate(['move','stop'])}
        report['video_commands']=a.video_commands;report['command_scale']=a.command_scale
        report['checkpoint']=str(a.checkpoint);report['video_pose_index']=0
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        sheet=Image.new('RGB',(960,360*len(snapshots)))
        for i,im in enumerate(snapshots):sheet.paste(im,(0,i*360))
        if snapshots: sheet.save(a.output/'frames.jpg')
        print('RESULT',a.output,flush=True)
    finally:env.close()

if __name__=='__main__':main()
