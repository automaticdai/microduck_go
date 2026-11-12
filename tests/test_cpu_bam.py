"""Floating roots must not alias a hinge's DOF friction onto its joint ID."""
import mujoco
import numpy as np
from mjlab_microduck.actuator.cpu_bam import dof_friction_force
from mjlab_microduck.sim.football_arena import build_world
from bam.model import load_model


def test_floating_base_friction_uses_mujoco_dof_identifier():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/><geom size=".1"/>
    <body pos="0 0 .2"><joint name="hinge" frictionloss="1"/><geom size=".1"/></body>
    </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    mask=d.efc_type==mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    jid=m.joint('hinge').id;did=int(m.joint('hinge').dofadr[0])
    assert jid!=did and d.efc_id[mask].tolist()==[did]
    d.efc_force[mask]=.7
    result=dof_friction_force(d,m.nv)
    assert result[jid]==0
    np.testing.assert_allclose(result[did],.7)
    np.testing.assert_allclose(result.sum(),.7)


def test_cpu_arena_retains_bam_current_limit():
    _,bam=build_world()
    assert bam.actuator.max_current==load_model(motor_name='xl330',model='m6').actuator.max_current
