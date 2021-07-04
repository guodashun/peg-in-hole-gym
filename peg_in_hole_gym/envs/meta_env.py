import os
import math
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class MetaEnv(object):
    action_space = spaces.Box(np.array([-1]),np.array([1]))
    observation_space = spaces.Box(np.array([-1]),np.array([1]))
    def __init__(self, client, offset, args=[]):
        self.p = client
        self.offset = np.array(offset)

        self._load_models()
        self._reset_internals()
        self.done = False
    
    def _load_models(self):
        raise NotImplementedError

    def _reset_internals(self):
        raise NotImplementedError

    def apply_action(self, action):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

    def reset(self, hard_reset=False):
        if hard_reset:
            self._load_models()
        self._reset_internals()
        self.done = False
    
        ''' 
        return obs 
        '''
    
    def render(self, mode='rgb_array'):
        raise NotImplementedError
        
    def test_mode(self, test_key, func):
        keys = p.getKeyboardEvents()
        if len(keys)>0:
            for k,v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    if (k==ord(test_key)):
                        func()

    def init_table(self, table_pos, flags):
        self.table_id=self.p.loadURDF("table/table.urdf",basePosition=table_pos, 
                                      baseOrientation=self.p.getQuaternionFromEuler([0, 0, math.pi/2]), globalScaling=2,
                                      flags=flags)

    def init_panda(self, panda_pos, panda_orn, table_pos, flags=0):
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.panda_id=self.p.loadURDF("franka_panda/panda.urdf",basePosition=panda_pos,
                                    baseOrientation=self.p.getQuaternionFromEuler([0, 0, -math.pi/2]),useFixedBase=True,
                                    flags=flags)
        for i in range(len(panda_orn)):
            self.p.resetJointState(self.panda_id,i,panda_orn[i])
        self.init_table(table_pos, flags)

    def init_ur(self, ur_pos, ur_orn, table_pos, flags=0):
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ur_id=self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/ur5.urdf"),basePosition=ur_pos,
                                    baseOrientation=[0, 0, 0, 1],useFixedBase=True,
                                    flags=flags)
        for i in range(len(ur_orn)):
            self.p.resetJointState(self.ur_id,i+1,ur_orn[i])
        self.init_table(table_pos, flags)


    def reset_panda(self, panda_orn):
        for i in range(len(panda_orn)):
            self.p.resetJointState(self.panda_id,i,panda_orn[i])

    def reset_ur(self, ur_orn):
        for i in range(len(ur_orn)):
            self.p.resetJointState(self.ur_id,i+1,ur_orn[i])

    def panda_execute(self, action, pandaEndEffectorIndex, pandaNumDofs, dv=2/240.):
        orientation=self.p.getQuaternionFromEuler([0.,-math.pi,0.])
        currentPose=self.p.getLinkState(self.panda_id,pandaEndEffectorIndex)
        currentPosition=currentPose[0]
        
        newPosition = self.vel_constraint(currentPosition, action[:3], dv)
        fingers= len(action) > 3 and action[3] or 0.
        jointPoses=self.p.calculateInverseKinematics(self.panda_id,pandaEndEffectorIndex,newPosition,orientation)[0:7]
        self.p.setJointMotorControlArray(self.panda_id,list(range(pandaNumDofs))+[9,10],self.p.POSITION_CONTROL,list(jointPoses)+2*[fingers], positionGains=[1]*9) 

    def ur_execute(self, action, urEndEffectorIndex, urNumDofs, dv=2/240.):
        pos = action[:3]
        orn = self.p.getQuaternionFromEuler(action[3:6])

        # get max force
        max_forces = []
        num_joints = self.p.getNumJoints(self.ur_id)
        for i in range(num_joints):
            max_forces.append(self.p.getJointInfo(self.ur_id, i)[10])
        jointPoses=self.p.calculateInverseKinematics(self.ur_id,urEndEffectorIndex,pos,orn)
        self.p.setJointMotorControlArray(self.ur_id,list(range(1,urNumDofs+1)),self.p.POSITION_CONTROL,list(jointPoses), positionGains=[0.03]*urNumDofs, forces=max_forces[1:7])

    def vel_constraint(cur, tar, dv):
        res = []
        for i in range(len(tar)):
            diff = tar[i] - cur[i]
            re = 0
            if abs(diff) > dv:
                re = cur[i] + (dv if diff > 0 else -dv)
            else:
                re = cur[i] + diff
            res.append(re)
        return res
