import os
import math
import time
import random
from numpy.lib.npyio import load
import numpy as np
from queue import Queue
from gym import spaces
from .utils import init_panda, panda_execute, random_pos_in_panda_space
from .assets.lstm.banana import load_data, get_lstm_pre_data

class RealFly(object):
    mode_list = ['Banana', 'Bottle13', 'Bottle23', 'Bag']
    action_space=spaces.Box(np.array([-1]*3),np.array([1]*3)) # 末端3维信息
    observation_space = spaces.Box(np.array([-1]*12), np.array([1]*12)) # [物体位置+速度 末端位置+速度]
    def __init__(self, client, offset=[0,0,0], args=None):
        assert (args[0] in self.mode_list)
        self.offset = np.array(offset)
        self.p = client
        self.data_form = args[0]
        self.time_step = args[1]if len(args)>2 else 1/240.
        self.p.setTimeStep(self.time_step)

        # panda init
        self.pandaEndEffectorIndex = 11
        self.pandaNumDofs = 7
        self.pandaUid = 0

        self.done = False
    
    def step(self, action):
        pre_start = action[0]
        dv = action[1]
        distance = action[2] if len(action)>1 else None
        observation = reward =0 
        success = self._virtual_fly(pre_start, distance, dv)
        self.done = True
        info = {}
        info['success'] = success
        return observation, reward, self.done, info
    

    def reset(self):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING,1)        
        self.gravity = [0,0,-9.8]
        self.p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
    
        # reset panda
        panda_base_pose = np.array([0.,0.,0.])+self.offset
        table_base_pose = np.array([0.0,-0.5,-1.3])+self.offset
        rest_poses=[0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.pandaUid, _ = init_panda(self.p, panda_base_pose, rest_poses, table_base_pose, flags)
    
        # init flying object
        base_orn = self.p.getQuaternionFromEuler([random.uniform(-math.pi, math.pi) for i in range(3)])
        self.objectUid = self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/Amicelli_800_tex.urdf"),
                            basePosition=np.array([0,0,10])+self.offset, baseOrientation=base_orn,
                            globalScaling=5)
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
        self.object_real_traj = load_data(self.data_name)
        self.object_raw_data = self.object_real_traj.copy()
        self.object_real_traj[:,[1,2,4,5,7,8]] = self.object_real_traj[:,[2,1,5,4,8,7]]
        random_pos = random_pos_in_panda_space()
        self.diff = self.object_real_traj[-1, :3] - random_pos
        if type(distance) == list:
            random_pos = self.p.getLinkState(self.pandaUid,11)[0] + np.array(distance)
            self.diff = self.object_real_traj[-1, :3] - random_pos
        self._make_in_work_space(self.object_real_traj, self.diff, random_pos, True)
        if type(distance) == list:
            for i in range(len(self.object_real_traj)-1):
                self.p.addUserDebugLine(np.array(self.object_real_traj[i][:3]), np.array(self.object_real_traj[i+1][:3]),lineColorRGB=[255,0,0], lineWidth=3)


    def _make_in_work_space(self, data, diff, random_pos = [], need_rotate=False):
        theta = - (math.atan2((data[-1][1]- data[0][1]), (data[-1][0] - data[0][0])) - math.pi/2)
        for i in range(len(self.object_real_traj)):
            data[i,:3] = data[i,:3] - diff
            if need_rotate:
                x = data[i][0] - random_pos[0]
                y = data[i][1] - random_pos[1]
                data[i][0] = x * math.cos(theta) - y * math.sin(theta) + random_pos[0]
                data[i][1] = x * math.sin(theta) + y * math.cos(theta) + random_pos[1]


    def _get_pre_data(self, raw_data, data_name, pre_start):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        dat_min = np.load(cur_dir + '/assets/lstm/data/' + self.data_form + '/npy/dat_min.npy')
        dat_mm = np.load(cur_dir + '/assets/lstm/data/' + self.data_form + '/npy/dat_mm.npy')
        pre_data = get_lstm_pre_data(raw_data, data_name, dat_min, dat_mm, pre_start, False)
        pre_data[:,[1,2,4,5,7,8]] = pre_data[:,[2,1,5,4,8,7]]
        return pre_data


    def _virtual_fly(self, start_time, distance, dv):
        self._init_flying_object(distance)
        for i in range(start_time):
            self.p.resetBaseVelocity(self.objectUid, self.object_real_traj[i, 3:6])
            self.p.resetBasePositionAndOrientation(self.objectUid, self.object_real_traj[i, 0:3], self.object_real_traj[i, 9:13])
            self.p.stepSimulation()
        j = 0
        while j < (len(self.object_real_traj)-start_time):
            lstm_time = time.time()
            pre_data = self._get_pre_data(self.object_raw_data, self.data_name, j+start_time)
            self._make_in_work_space(pre_data, self.diff)
            cst_time = math.ceil((time.time() - lstm_time) / self.time_step)
            cst_time = 6 if cst_time >= 120 else cst_time
            cst_time = ((len(self.object_real_traj)-start_time)-j) if (j+cst_time >= (len(self.object_real_traj)-start_time)) else cst_time
            for i in range(cst_time):
                self.p.resetBaseVelocity(self.objectUid, self.object_real_traj[j+start_time, 3:6])
                self.p.resetBasePositionAndOrientation(self.objectUid, self.object_real_traj[j+start_time, 0:3], self.object_real_traj[j+start_time, 9:13])
                self.p.stepSimulation()
                self.p.addUserDebugLine([0,0,0], pre_data[-1][:3], [0,255,0], 4, 0.1)
                panda_execute(self.p, self.pandaUid, pre_data[-1][:3]-self.offset, self.pandaEndEffectorIndex, self.pandaNumDofs, dv)
                j += 1
                if (
                    self.p.getContactPoints(self.pandaUid, self.objectUid, 8)
                ):
                    return True
        return False
