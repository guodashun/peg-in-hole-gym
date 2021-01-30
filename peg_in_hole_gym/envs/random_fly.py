import os
import math
import time
import random
import numpy as np
from queue import Queue
from gym import spaces
from .utils import data_normalize, init_panda, panda_execute

class RandomFly(object):
    action_space=spaces.Box(np.array([-1]*3),np.array([1]*3)) # 末端3维信息
    observation_space = spaces.Box(np.array([-1]*12), np.array([1]*12)) # [物体位置+速度 末端位置+速度]
    def __init__(self, client, offset=[0,0,0], args=None):
        self.offset = np.array(offset)
        self.p = client

        # panda init
        self.pandaUid = 0
        self.tableUid = 0
        self.pandaEndEffectorIndex = 11
        self.pandaNumDofs = 7
        self.panda_dv = 2/240.
        if args:
            self.panda_dv = args[0]

        self.done = False
        self.t = 0
        # self.reset()
    
    def step(self, action):
        panda_execute(self.p, self.pandaUid, action, self.pandaEndEffectorIndex, self.pandaNumDofs, self.panda_dv)
        observation, reward, success = self.random_fly()
        info = {}
        info['success'] = success
        return observation, reward, self.done, info
    

    # random fly scene
    def random_fly(self):
        obj_pos = self.p.getBasePositionAndOrientation(self.objectUid)[0]
        obj_vel, obj_w = self.p.getBaseVelocity(self.objectUid)[0:2]
        if self.obs_delay:
            self.obs_delay_queue.put([obj_pos, obj_vel])
            obj_pos, obj_vel = self.obs_delay_queue.get()
        base_info = self.p.getLinkState(self.pandaUid,self.pandaEndEffectorIndex,computeLinkVelocity=1)
        base_pos = base_info[0]
        base_vel = base_info[6]
        if (
            # np.linalg.norm(np.array(obj_pos) - np.array(self.object_pos)) < 0.01 
            # or (abs(obj_vel[0]) + abs(obj_vel[1])) < 1 
            obj_vel[2] < -15 
            or self.p.getContactPoints(self.tableUid, self.objectUid)
            # or sum([abs(x) for x in obj_vel]) < 0.5
            # or np.linalg.norm(obj_w) > 2 
           ):
            self.done = True
            # print("why reset", np.linalg.norm(np.array(obj_pos) - np.array(self.object_pos)) < 0.01 
            #         , obj_vel[2] < -15
            #         , np.linalg.norm(obj_w) > 2)
        reward = 0.0
        success = False

        pos_bias = np.linalg.norm(np.array(base_pos) - np.array(obj_pos))
        # vel_bias = np.linalg.norm(np.array(base_vel) - np.array(obj_vel))
        move_cst = np.linalg.norm(np.array(base_pos) - np.array(self.last_panda_pos))
        pos_r  = -math.log(pos_bias) * 0.7
        # vel_r  = -math.log(vel_bias) * 0.01
        move_r = 1 - math.exp(move_cst)
        use_time_r = False
        if (
            self.p.getContactPoints(self.pandaUid, self.objectUid, 8)
            or (self.done and pos_bias < 0.4)
           ):
            success = True
        if self.p.getContactPoints(self.pandaUid, self.objectUid, 8):
            cli_r = 800 + (self.time_r if use_time_r else 0.)
            self.done=True
        else:
            cli_r = -0.5
        self.time_r -= 0.01

        reward = cli_r + pos_r # + vel_r + + move_r

        self.last_panda_pos = base_pos

        # normalize
        res = list(obj_pos+ obj_vel+ base_pos+ base_vel)
        res = data_normalize(res, self.normalize_range)

        return res, reward, success

    def reset(self):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING,1)        
        self.gravity = [0,0,-9.8]
        self.p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
    
        # reset panda
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        rest_poses=[0,-0.215,-math.pi/3,-2.57,0,2.356,2.356,0.08,0.08]
        panda_base_pose = np.array([0.,0.,0.])+self.offset
        table_base_pose = np.array([0.0,-0.5,-1.3])+self.offset
        self.pandaUid, self.tableUid = init_panda(self.p, panda_base_pose, rest_poses, table_base_pose, flags)

        # normalize range
        self.normalize_range = np.array([[-6,6], [-6,6], [-3,16],               # obj_pos
                                         [-6,6], [-6,6], [-15, 5],              # obj_v
                                         [-1,1], [-1,1], [-1,1],                # base_pos
                                         [-10, 10], [-10, 10], [-10,10]])       # base_v
        # offset
        for i in range(len(self.offset)):
            self.normalize_range[i] = self.normalize_range[i] + self.offset[i]
            self.normalize_range[i+6] = self.normalize_range[i+6] + self.offset[i]

        # init flying object
        self.init_flying_object()
        self.t = 0

        base_orn = self.p.getQuaternionFromEuler([random.uniform(-math.pi, math.pi) for i in range(3)])
        self.objectUid = self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/Amicelli_800_tex.urdf"),
                                    basePosition=self.object_pos, baseOrientation=base_orn,
                                    globalScaling=5)
        self.p.changeDynamics(self.objectUid, -1, linearDamping=0, angularDamping=0)
        self.p.resetBaseVelocity(self.objectUid, self.object_vel)


        obj_pos = self.p.getBasePositionAndOrientation(self.objectUid)[0]
        obj_vel = self.p.getBaseVelocity(self.objectUid)[0]
        base_info = self.p.getLinkState(self.pandaUid,11,computeLinkVelocity=1)
        base_pos = base_info[0]
        base_vel = base_info[6]
        self.last_panda_pos = base_pos

        # add time cost
        self.time_r = 0
        
        # done
        self.done = False

        # add delay
        self.obs_delay = True if random.random() > 0.3 else False
        if self.obs_delay:
            self.obs_delay_queue = Queue(random.randint(1,3))
            i = 0
            while not self.obs_delay_queue.full():
                i += 1
                self.p.stepSimulation()
                self.obs_delay_queue.put([obj_pos, obj_vel])
            obj_pos, obj_vel = self.obs_delay_queue.get()
        obs = list(obj_pos + obj_vel + base_pos + base_vel)
        obs = data_normalize(obs, self.normalize_range)
        return obs

    def random_pos_in_panda_space(self):
        # |x|,|y| < 0.8, 0 < z < 1
        # x^2 + y^2 + (z-0.2)^2 < 0.8^2
        x = y = z = 1
        length = 0.7
        while (length*length - x*x - y*y) < 0:
            x = random.uniform(-length, length)
            y = (math.sqrt(random.uniform(0, length*length-x*x))-random.uniform(0, 0.4))*random.choice([-1,1])
        z = math.sqrt(length*length - x*x - y*y) + 0.2
        res = np.array([x,y,z]) + self.offset
        return res

    def render(self, mode='rgb_array'):
        pass

    def init_flying_object(self):
        target_pos = self.random_pos_in_panda_space()
        self.p.addUserDebugLine(self.offset, target_pos,lineColorRGB=[255,0,0], lineWidth=3)
        x  = random.uniform(4,6) * random.choice([-1., 1.]) + self.offset[0]
        vx = random.uniform(4,6) * (-1. if (x-self.offset[0])>0 else 1.)
        t  = abs((target_pos[0] - x) / vx)
        vy = random.uniform(4,6) * random.choice([-1., 1.])
        vz = random.randint(-2,5)
        y  = target_pos[1] - vy * t
        z  = target_pos[2] - (vz * t +  (self.gravity[2]) * t*t / 2)
        self.object_pos = [x, y, z]
        self.object_vel = [vx,vy,vz]
