"""Stop and restart gates must prevent promotion of a never-stopping motor."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_recovery_walk import evaluate


def run_battery(ignores_stop=False, fails=False):
    n=14
    d=NS(projected_gravity_b=torch.tensor([[0.,0,-1.]]*n),root_link_pos_w=torch.tensor([[0.,0,.11]]*n),
         root_link_lin_vel_b=torch.zeros(n,3),root_link_ang_vel_b=torch.zeros(n,3))
    class Scene(dict): pass
    scene=Scene(robot=NS(data=d));scene.terrain=NS(env_origins=torch.zeros(n,3))
    env=NS(num_envs=n,device='cpu',step_dt=.02,scene=scene,reset=lambda:None,
           termination_manager=NS(get_term=lambda _:torch.zeros(n,dtype=torch.bool)))
    class Actor:
        def eval(self):return self
        def __call__(self,obs):return torch.zeros(n,14)
    class Wrapped:
        step_index=0
        def get_observations(self):return {}
        def step(self,actions):
            c=env.recovery_walk_eval_command.clone()
            if 150<=self.step_index<200 and not ignores_stop:c.zero_()
            d.root_link_lin_vel_b[:,:2]=c[:,:2];d.root_link_ang_vel_b[:,2]=c[:,2]
            done=torch.zeros(n,dtype=torch.bool)
            if fails and self.step_index==10:done[2]=True
            self.step_index+=1
            return {},None,done,{}
    return evaluate(env,Wrapped(),Actor(),{'height':torch.zeros(2)})


def test_complete_stop_restart_sequence_passes():
    r=run_battery();assert r['passed'] and r['movement_gate_passed']
    assert all(c['complete_sequence']==2 for c in r['commands'].values())


def test_never_stopping_fails_despite_good_initial_motion():
    r=run_battery(ignores_stop=True)
    assert r['movement_gate_passed'] and not r['passed']
    assert r['commands']['forward']['upright_and_stopped']==0
    assert r['commands']['turn_left']['upright_and_stopped']==0


def test_reset_cannot_be_counted_as_successful_recovery_of_original_trial():
    r=run_battery(fails=True)
    assert not r['passed']
    assert r['commands']['forward']['complete_sequence']==1
