import pybullet as p
import gym
from .base_env import BaseEnv
from .utils import MPMultiAgentActionSpace, MPMultiAgentObservationSpace
from multiprocessing import Queue, Process

class BaseEnvMp(gym.Env):

    CLOSE = 0
    RESET = 1
    STEP = 2
    RENDER = 3
    
    def __init__(self, client, task, mp_num=1, sub_num=1, offset=[0,0,0], args=None, is_test=False):
        if client == p.GUI:
            print("Warning: GUI mode is not supported in this env.")
            print("Auto switch to DIRECT mode.")
            client = p.DIRECT
        self.mp_num = mp_num
        self.task = task
        self.sub_num = sub_num
        self.clients = []
        self.msg_queues = []
        self.res_queues = []
        self.processes = []
        for i in range(self.mp_num):    
            self.clients.append(BaseEnv(client, task, self.sub_num, offset, args, is_test))
            msg_queue = Queue(1)
            res_queue = Queue(1)
            self.msg_queues.append(msg_queue)
            self.res_queues.append(res_queue)
            sub_process = Process(target=self.worker, args=(i, self.msg_queues[i], self.res_queues[i]))
            sub_process.start()
            self.processes.append(sub_process)
        self.action_space = MPMultiAgentActionSpace([self.clients[i].action_space for i in range(self.mp_num)])
        self.observation_space = MPMultiAgentObservationSpace([self.clients[i].observation_space for i in range(self.mp_num)])


    def step(self, action):
        for i in range(self.mp_num):
            if not all(self.dones[i]):
                self.msg_queues[i].put([self.STEP, action[i]])
        for i in range(self.mp_num):
            if not all(self.dones[i]):
                obs, reward, done, info = self.res_queues[i].get()
                self.observations[i] = obs
                self.rewards[i] = reward
                self.dones[i] = done
                self.infos[i] = info
        return self.observations, self.rewards, self.dones, self.infos

    
    def reset(self):
        self.observations = self.observation_space.sample()
        self.rewards = [[0. for _ in range(self.sub_num)] for _ in range(self.mp_num)]
        self.infos = [[{} for _ in range(self.sub_num)] for _ in range(self.mp_num)]
        self.dones = [[False for _ in range(self.sub_num)] for _ in range(self.mp_num)]
        for i in range(self.mp_num):
            self.msg_queues[i].put([self.RESET, None])
        for i in range(self.mp_num):
            self.observations[i] = self.res_queues[i].get()
        return self.observations


    def render(self, mode):
        for i in range(self.mp_num):
            self.msg_queues[i].put([self.RENDER, mode])

    def close(self):
        for i in range(self.mp_num):
            self.msg_queues[i].put([self.CLOSE, None])
    

    def worker(self, idx, msg_queue, res_queue):
        while True:
            msg, args = msg_queue.get()
            if msg == self.RESET:
                obs = self.clients[idx].reset()
                res_queue.put(obs)
            if msg == self.STEP:
                obs, reward, done, info = self.clients[idx].step(args)
                res_queue.put([obs, reward, done, info])
            if msg == self.RENDER:
                self.clients[idx].render(args)
            if msg == self.CLOSE:
                break
