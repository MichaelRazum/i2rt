import os
import time

import numpy as np


from i2rt.robots.utils import GripperType
from i2rt.utils.mujoco_utils import MuJoCoKDL






path = GripperType.LINEAR_4310.get_xml_path()
kdl = MuJoCoKDL(path)
print(kdl)

import mujoco
import mujoco.viewer


dt: float = 0.01
with mujoco.viewer.launch_passive(
        model=kdl.model,
        data=kdl.data,
) as viewer:
    mujoco.mjv_defaultFreeCamera(kdl.model, viewer.cam)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    while viewer.is_running():
        step_start = time.time()
        joint_pos = np.array([-1.41851682,  0.77344167,  0.72117952,  0.03986419,  0.11768521,-0.1703059,])
        kdl.data.qpos[:len(joint_pos)] = joint_pos

        # sync the kdl.model state
        mujoco.mj_kinematics(kdl.model, kdl.data)
        viewer.sync()
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)