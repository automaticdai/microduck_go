"""CPU BAM adapter with friction constraints indexed by DOF, as in mjlab."""
import mujoco
import numpy as np
from bam.mujoco import MujocoController


def dof_friction_force(data, nv):
    force=np.zeros(nv)
    mask=data.efc_type==mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    # MuJoCo friction constraints identify a DOF, not a joint. Floating bases
    # make these indices differ; using joint IDs includes the wrong loads.
    np.add.at(force,data.efc_id[mask],data.efc_force[mask])
    return force


class CpuBamController(MujocoController):
    def update(self):
        super().update()
        # Recompute the budget after BAM's voltage/torque update. The upstream
        # CPU controller subtracts friction using joint_indexes; the Warp
        # implementation correctly uses dof IDs. No physics step occurs here.
        d=self.mujoco_data;ids=self.dof_indexes
        external=-d.qfrc_bias[ids]+d.qfrc_constraint[ids]-dof_friction_force(d,self.mujoco_model.nv)[ids]
        friction,damping=self.model.compute_frictions(d.qfrc_actuator[ids],external,d.qvel[ids])
        self.mujoco_model.dof_frictionloss[ids]=friction
        self.mujoco_model.dof_damping[ids]=damping
