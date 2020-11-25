from multiprocessing.connection import Client
from os import pardir
from .panda_env import PandaEnv
import pybullet as p
import multiprocessing as mp
# from pybullet_utils.bullet_client import BulletClient
from multiprocessing import Pipe, Process

# from .panda_env import PandaEnv

class PandaEnvMp(PandaEnv):
    def __init__(self, client, task, is_test, mp_num):
        if client == p.GUI:
            self.mp_num = 1
        else:
            self.mp_num = mp_num
        self.clients = []
        self.child_pipes = []
        self.parent_pipes = []
        for i in range(self.mp_num):    
            self.clients.append(PandaEnv(client, task, is_test))
            parent_pipe, child_pipe = Pipe()
            self.child_pipes.append(child_pipe)
            self.parent_pipes.append(parent_pipe)
        
        # super(PandaEnvMp, self).__init__(client, task, is_test)
        


    def step(self):
        pass

    
    def reset(self):
        pass


    def render(self):
        pass 
