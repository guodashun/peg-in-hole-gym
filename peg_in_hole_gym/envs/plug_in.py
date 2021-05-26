import os
import math
import time
import random
from numpy.lib.npyio import load
import numpy as np
from queue import Queue
from gym import spaces
from .utils import init_ur, ur_execute # , random_pos_in_ur_space, rotate_2d
from .assets.lstm.flyer import Flyer

class PlugIn(object):
    mode_list = ['Banana', 'Bottle', 'Bamboo', 'Green', 'Gourd', 'Paige']
    action_space=spaces.Box(np.array([-0.8,0,0,-math.pi,-math.pi,-math.pi]),np.array([0.8,0.8,0.8,math.pi,math.pi,math.pi]))
    observation_space = spaces.Box(np.array([-1]), np.array([1]))
    def __init__(self, client, offset=[0,0,0], args=['Banana', 1/120., False]):
        assert (args[0] in self.mode_list)
        self.offset = np.array(offset)
        self.p = client
        self.data_form = args[0]
        self.flyer = Flyer(self.data_form)
        self.time_step = args[1] if len(args)>1 else 1/240.
        self.verbose = args[2] if len(args)>2 else False
        self.p.setTimeStep(self.time_step)

        # ur init
        self.urEndEffectorIndex = 7 # !!!!!
        self.urNumDofs = 6
        self.urUid = 0

        self.done = False

    def apply_action(self, action):
        action = np.clip(action, self.action_space.low,self.action_space.high)
        ur_execute(self.p, self.urUid, action, self.urEndEffectorIndex, self.urNumDofs)

    def get_info(self):    
        self.done = False
        info = {}
        return 0.,0., self.done, info

    def reset(self):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING,1)        
        self.gravity = [0,0,-9.8]
        self.p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
    
        # reset ur
        ur_base_pose = np.array([0, 0.0, -0.1])+self.offset
        table_base_pose = np.array([0.0,-0.5,-1.3])+self.offset
        self.rest_poses=[3.172306597303268, -1.7963777540644157, 1.79467186617118, 0.13610846611006164, 0.14318352968043557, -0.01613721010103956]
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.p.URDF_USE_INERTIA_FROM_FILE
        self.urUid, _ = init_ur(self.p, ur_base_pose, self.rest_poses, table_base_pose, flags)
    
        # init charge board
        base_orn = self.p.getQuaternionFromEuler([0,0,0])
        self.objectUid = self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/charge_board.urdf"),
                            basePosition=np.array([0.4,0.5,0.5])+self.offset, baseOrientation=base_orn,
                            globalScaling=1)
        # self.p.changeDynamics(self.objectUid, -1, linearDamping=0, angularDamping=0)
        
        # done
        self.done = False

        obs = []
        return obs

    def render(self, mode='rgb_array'):
        pass
