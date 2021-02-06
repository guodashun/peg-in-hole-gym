import os
import math
import time
import random
from numpy.lib.npyio import load
import numpy as np
from queue import Queue
from gym import spaces
from .utils import init_panda, panda_execute, random_pos_in_panda_space, rotate_2d
from .assets.lstm.flyer import Flyer

class RealFly(object):
    mode_list = ['Banana', 'Bottle13', 'Bottle23', 'Bag', 'Newton']
    action_space=spaces.Box(np.array([-100]*3),np.array([100]*3))
    observation_space = spaces.Box(np.array([-1]), np.array([1]))
    def __init__(self, client, offset=[0,0,0], args=None):
        assert (args[0] in self.mode_list)
        self.offset = np.array(offset)
        self.p = client
        self.data_form = args[0]
        self.flyer = Flyer(self.data_form)
        self.time_step = args[1]if len(args)>1 else 1/240.
        self.verbose = args[2] if len(args)>2 else False
        # print(self.verbose, args[2])
        self.p.setTimeStep(self.time_step)

        # panda init
        self.pandaEndEffectorIndex = 11
        self.pandaNumDofs = 7
        self.pandaUid = 0

        self.done = False
    
    def apply_action(self, action):
        pre_start = action[0]
        dv = action[1]
        distance = action[2] if len(action)>1 else None
        self.success = self._virtual_fly(pre_start, distance, dv)
        
    def get_info(self):    
        self.done = True
        info = {}
        info['success'] = self.success
        info['arm_vel'] = self.arm_vel
        if self.success:
            info['err'] = 0
        else:
            info['err'] = self.err
        return 0.,0., self.done, info
    

    def reset(self):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING,1)        
        self.gravity = [0,0,-9.8]
        self.p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
    
        # reset panda
        panda_base_pose = np.array([0.,0.,0.])+self.offset
        table_base_pose = np.array([0.0,-0.5,-1.3])+self.offset
        rest_poses=[0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08,0,0,0]
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.pandaUid, _ = init_panda(self.p, panda_base_pose, rest_poses, table_base_pose, flags)
        self.arm_vel = []
    
        # init flying object
        base_orn = self.p.getQuaternionFromEuler([random.uniform(-math.pi, math.pi) for i in range(3)])
        self.objectUid = self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/banana.urdf"),
                            basePosition=np.array([0,0,10])+self.offset, baseOrientation=base_orn,
                            globalScaling=1)
        self.p.changeDynamics(self.objectUid, -1, linearDamping=0, angularDamping=0)
        
        # done
        self.done = False

        obs = []
        return obs

    def render(self, mode='rgb_array'):
        pass

    def _init_flying_object(self, distance=None):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = cur_dir + '/assets/lstm/data/' + self.data_form + '/test/'
        data_list = os.listdir(data_dir)
        self.data_name = data_dir + random.choice(data_list)
        if self.data_form == self.mode_list[-1]:
            self.object_real_traj = np.load(self.data_name)
        else:
            self.object_real_traj = Flyer.load_data(self.data_name)
        self.object_raw_data = self.object_real_traj.copy()
        self.object_real_traj[:,[1,2,4,5,7,8]] = self.object_real_traj[:,[2,1,5,4,8,7]]
        random_pos = random_pos_in_panda_space()
        self.diff = self.object_real_traj[-1, :3] - random_pos
        if type(distance) == list:
            random_pos = self.p.getLinkState(self.pandaUid,11)[0] + np.array(distance)
            self.diff = self.object_real_traj[-1, :3] - random_pos
        if self.verbose:
            for i in range(len(self.object_real_traj)-1):
                self.p.addUserDebugLine(np.array(self.object_real_traj[i][:3]), np.array(self.object_real_traj[i+1][:3]),lineColorRGB=[0,0,255], lineWidth=3)
        self._make_in_work_space(self.object_real_traj, self.diff)
        theta = - (math.atan2((self.object_real_traj[-1][1]- self.object_real_traj[0][1]),\
                              (self.object_real_traj[-1][0] - self.object_real_traj[0][0])) - math.pi/2)
        if self.verbose:
            for i in range(len(self.object_real_traj)-1):
                self.p.addUserDebugLine(np.array(self.object_real_traj[i][:3]), np.array(self.object_real_traj[i+1][:3]),lineColorRGB=[0,255,0], lineWidth=3)
        for i in range(len(self.object_real_traj)):
            rotate_2d(self.object_real_traj[i], random_pos, theta)
        
        if self.verbose:
            for i in range(len(self.object_real_traj)-1):
                self.p.addUserDebugLine(np.array(self.object_real_traj[i][:3]), np.array(self.object_real_traj[i+1][:3]),lineColorRGB=[255,0,0], lineWidth=3)


    def _make_in_work_space(self, data, diff):
        for i in range(len(data)):
            data[i,:3] = data[i,:3] - diff


    def _get_pre_data(self, raw_data, pre_start):
        self.flyer.init_lstm_pre(raw_data[:pre_start])
        for _ in range(len(raw_data)-pre_start):
            self.flyer.get_next_pre_data()
        pre_data = self.flyer.get_whole_pre_data()
        pre_data[:,[1,2,4,5,7,8]] = pre_data[:,[2,1,5,4,8,7]]
        err = np.linalg.norm(raw_data[-1][0:3] - pre_data[-1][0:3])
        return pre_data, err


    def _virtual_fly(self, start_time, distance, dv):
        self._init_flying_object(distance)
        for i in range(start_time):
            self.p.resetBaseVelocity(self.objectUid, self.object_real_traj[i, 3:6])
            if self.data_form == self.mode_list[-1]:
                base_orn = self.p.getQuaternionFromEuler([random.uniform(-math.pi, math.pi) for i in range(3)])
                self.p.resetBasePositionAndOrientation(self.objectUid, self.object_real_traj[i, 0:3], base_orn)
            else:
                self.p.resetBasePositionAndOrientation(self.objectUid, self.object_real_traj[i, 0:3], self.object_real_traj[i, 9:13])
            self.p.stepSimulation()
            self.arm_vel.append([0,0,0])
        j = 0
        while j < (len(self.object_real_traj)-start_time):
            lstm_time = time.time()
            pre_data, self.err = self._get_pre_data(self.object_raw_data, j+start_time)
            self._make_in_work_space(pre_data, self.diff)
            cst_time = math.ceil((time.time() - lstm_time) / self.time_step)
            cst_time = 6 if cst_time >= 120 else cst_time
            cst_time = ((len(self.object_real_traj)-start_time)-j) if (j+cst_time >= (len(self.object_real_traj)-start_time)) else cst_time
            for i in range(cst_time):
                self.p.resetBaseVelocity(self.objectUid, self.object_real_traj[j+start_time, 3:6])
                if self.data_form == self.mode_list[-1]:
                    base_orn = self.p.getQuaternionFromEuler([random.uniform(-math.pi, math.pi) for i in range(3)])
                    self.p.resetBasePositionAndOrientation(self.objectUid, self.object_real_traj[j+start_time, 0:3], base_orn)
                else:
                    self.p.resetBasePositionAndOrientation(self.objectUid, self.object_real_traj[j+start_time, 0:3], self.object_real_traj[j+start_time, 9:13])
                self.p.stepSimulation()
                panda_execute(self.p, self.pandaUid, pre_data[-1][:3]-self.offset, self.pandaEndEffectorIndex, self.pandaNumDofs, dv)
                j += 1
                if self.verbose:
                    self.p.addUserDebugLine(pre_data[-1][:3]-self.offset, self.offset, [255,255,0], 3)
                self.arm_vel.append(self.p.getLinkState(self.pandaUid,self.pandaEndEffectorIndex,computeLinkVelocity=1)[6])
                if (
                    self.p.getContactPoints(self.pandaUid, self.objectUid, 8)
                ):
                    return True
        return False
