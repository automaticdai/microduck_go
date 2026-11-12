"""Hybrid blue motor: distance hysteresis, post-recovery settle, per-round reset."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_hybrid_motor import HybridMotor


def make(n=1):
    class Scene(dict): pass
    robot=NS(data=NS(root_link_pos_w=torch.zeros(n,3)))
    ball=NS(data=NS(root_link_pos_w=torch.zeros(n,3)))
    scene=Scene(robot=robot,ball=ball);scene.terrain=NS(env_origins=torch.zeros(n,3))
    env=NS(num_envs=n,device='cpu',step_dt=.02,scene=scene,episode_length_buf=torch.full((n,),10),
           arena_recovery=NS(active=torch.zeros(n,6,dtype=torch.bool)))
    walker=lambda obs:torch.zeros(n,14)
    football=lambda obs:torch.ones(n,14)
    return env,HybridMotor(env,'robot',walker,football,enter=.35,exit=.5,settle=1.)


def at(env,motor,distance):
    env.scene['robot'].data.root_link_pos_w[:,0]=distance
    return motor({'actor':torch.zeros(env.num_envs,61)})[0,0].item()


def test_distance_hysteresis():
    env,motor=make()
    assert at(env,motor,.45)==0          # outside enter radius: walker
    assert at(env,motor,.30)==1          # entered: football
    assert at(env,motor,.45)==1          # inside exit radius: stays football
    assert at(env,motor,.55)==0          # beyond exit: walker
    assert at(env,motor,.45)==0


def test_walker_settles_after_recovery_then_round_reset_clears():
    env,motor=make()
    env.arena_recovery.active[:,0]=True
    assert at(env,motor,.1)==0           # recovering: walker output (recovery overrides later)
    env.arena_recovery.active[:,0]=False
    for _ in range(49): assert at(env,motor,.1)==0
    assert at(env,motor,.1)==1           # 1.0 s after handoff
    env.arena_recovery.active[:,0]=True;at(env,motor,.1);env.arena_recovery.active[:,0]=False
    env.episode_length_buf[:]=0          # new round forgets both latches
    assert at(env,motor,.1)==1
    assert at(env,motor,.45)==0


def test_nearest_roles_hysteresis_and_partner_side():
    from check_hybrid_motor import nearest_roles
    class Scene(dict): pass
    pos=lambda x,y:NS(data=NS(root_link_pos_w=torch.tensor([[x,y,0.]])))
    scene=Scene(robot=pos(.30,0),blue_attacker=pos(.20,0),ball=pos(0,.1))
    scene.terrain=NS(env_origins=torch.zeros(1,3));env=NS(scene=scene)
    striker,skills=nearest_roles(env,torch.tensor([0]),'cover')
    assert striker.item()==1 and skills.tolist()==[[3,0]]          # teammate 10 cm closer
    scene['robot']=pos(.23,0)
    striker,skills=nearest_roles(env,striker,'support')
    assert striker.item()==1 and skills.tolist()==[[2,0]]          # within margin: keep; ball y>0 -> right
