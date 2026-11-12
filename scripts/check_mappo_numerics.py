"""GPU regression: inject one bad joint state and verify isolated reset plus PPO recovery.

Pass the same --attacker/--keeper arguments as train_mappo.py. This runs five
training iterations by default; it is a deliberate-fault diagnostic, not a long run.
"""
import json
from pathlib import Path
import torch
import train_mappo

original_init=train_mappo.TacticalArena.__init__
original_step=train_mappo.TacticalArena.step
observed={'injected':False,'invalid':0,'continued':False,'directory':None}


def instrumented_init(self,env,attacker,keeper):
    original_init(self,env,attacker,keeper)
    original_compute=env.termination_manager.compute
    observed['directory']=Path(env.arena_numerical_report_dir).parent
    def inject_before_termination():
        if not observed['injected'] and env.common_step_counter == 3:
            # Mutate real MuJoCo qpos after physics, immediately before the existing
            # termination guards. Other worlds and all network weights stay untouched.
            address=env.scene['blue_attacker'].indexing.joint_q_adr[0]
            env.sim.data.qpos[0,address]=float('nan')
            observed['injected']=True
        return original_compute()
    env.termination_manager.compute=inject_before_termination


def checked_step(self,skills):
    result=original_step(self,skills)
    invalid=~self.last_valid
    if invalid.any():
        assert invalid.nonzero().flatten().tolist()==[0]
        assert bool(result[2][0])
        assert not self.previous_velocity[0].any()
        observed['invalid']+=int(invalid.sum())
    elif observed['invalid']:
        observed['continued']=True
    assert all(torch.isfinite(x).all() for x in self.observations())
    return result


if __name__=='__main__':
    train_mappo.TacticalArena.__init__=instrumented_init
    train_mappo.TacticalArena.step=checked_step
    train_mappo.main()
    assert observed['injected'] and observed['invalid']==1 and observed['continued']
    directory=observed['directory']
    events=[json.loads(line) for line in (directory/'numerics/events.jsonl').read_text().splitlines()]
    assert len(events)==1 and events[0]['entity']=='blue_attacker'
    assert events[0]['nonfinite_fields']=={'joint_pos':1}
    checkpoint=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
    assert checkpoint['iteration']==4
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert sum(row['invalid_transitions'] for row in metrics)==1
    assert sum(row['numerical_resets'] for row in metrics)==1
    print('FAULT RECOVERY PASSED: one isolated reset, one excluded transition, '
          'five finite PPO updates, rolling checkpoint 4, ONNX parity.',flush=True)
