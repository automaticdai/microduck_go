"""GCP physical recovery battery: 64 arenas, face-down/up/left/right starts."""
import argparse
import json
from pathlib import Path
import torch
import train_mappo as tm

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--recovery',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
meta=ck['metadata']
tm.configure_torch_backends(); torch.manual_seed(42)
cfg=tm.load_env_cfg(tm.TASK); cfg.scene.num_envs=64; cfg.seed=42
cfg.commands['twist'].track_training_signals=False
cfg.rewards={'mappo_team':tm.RewardTermCfg(func=tm.mdp.arena_mappo_team_reward,weight=1.,params={'gamma':tm.GAMMA})}
cfg.curriculum={};cfg.metrics={}
env=tm.ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    env.arena_recovery_policy=tm.frozen_motor(a.recovery,'cuda:0')
    env.arena_recovery=tm.mdp.ArenaRecovery(env)
    arena=tm.TacticalArena(env,tm.frozen_motor(meta['attacker'],'cuda:0'),tm.frozen_motor(meta['keeper'],'cuda:0'))
    model=tm.TacticalMAPPO(ck['actor_dim'],ck['critic_dim']).to('cuda:0')
    model.load_state_dict(ck['model']);model.eval()
    d=env.scene['robot'].data
    pose=torch.cat((d.root_link_pos_w.clone(),d.root_link_quat_w.clone()),-1)
    kinds=torch.arange(64,device=env.device)%4
    quats=torch.tensor([[1.,0.,1.,0.],[1.,0.,-1.,0.],[1.,1.,0.,0.],[1.,-1.,0.,0.]],device=env.device)/(2**.5)
    pose[:,3:]=quats[kinds];pose[:,2]=env.scene.terrain.env_origins[:,2]+.20
    env.scene['robot'].write_root_link_pose_to_sim(pose)
    env.scene['robot'].write_root_link_velocity_to_sim(torch.zeros(64,6,device=env.device))
    env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
    alive=torch.ones(64,dtype=torch.bool,device=env.device)
    entered=alive&False;recovered=alive&False;invalid=alive&False
    # Observe every motor-step event, including those before a macro-step ends.
    original_step=env.step
    def monitored_step(action):
        global alive,entered,recovered,invalid
        result=original_step(action)
        entered |= env.arena_recovery.entered[:,0]&alive
        recovered |= env.arena_recovery.succeeded[:,0]&alive
        nan=torch.stack([env.termination_manager.get_term(k) for k in env.termination_manager.active_terms if 'nan' in k]).any(0)
        invalid |= nan&alive
        alive &= ~(result[2]|result[3])
        return result
    env.step=monitored_step
    steps=0
    with torch.inference_mode():
        while steps<400:
            obs,_=arena.observations()
            _,_,_,_,diag=arena.step(model.actor(obs).argmax(-1))
            steps+=int(diag['motor_steps'])
    report={}
    for i,name in enumerate(('face_down','face_up','left_side','right_side')):
        mask=kinds==i
        report[name]={'trials':int(mask.sum()),'recovery_entered':int(entered[mask].sum()),
                      'stable_handoff':int(recovered[mask].sum()),'numerical_resets':int(invalid[mask].sum())}
    a.output.write_text(json.dumps(report,indent=2))
    print('RECOVERY_BATTERY',json.dumps(report),flush=True)
    assert entered.all(), 'Some falls did not trigger the recovery controller'
    assert recovered.any(), 'No physically successful get-up handoff; do not launch long run'
    assert not invalid.any(), 'Numerical failure in physical recovery battery'
finally:
    env.close()
