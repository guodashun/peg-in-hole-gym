import pybullet as p
import gym
import math
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from .peg_in_hole import PegInHole
from .random_fly import RandomFly
from .real_fly import RealFly
from .utils import test_mode, MultiAgentActionSpace, MultiAgentObservationSpace

task_list = {
             'peg-in-hole':PegInHole, 
             'random-fly':RandomFly,
             'real-fly':RealFly
            }

class PandaEnv(gym.Env):
    metadata = {'render.modes':['human', 'rgb_array']}
    def __init__(self, client, task='peg-in-hole', task_num = 1, offset = [0,0,0], args=None, is_test=False):
        assert task in task_list
        assert (task_num == 1 or (task_num > 1 and offset != [0,0,0]))
        self.task = task
        self.task_num = task_num
        self.offset = offset
        self.args = args
        self.sub_env = task_list[self.task]
        self.p = BulletClient(client)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        self.sub_envs = []
        self._create_env()

        self.action_space = MultiAgentActionSpace([self.sub_env.action_space for i in range(task_num)])
        self.observation_space = MultiAgentObservationSpace([self.sub_env.observation_space for i in range(task_num)])
        
        # test_mode
        self.is_test = is_test

    def _create_env(self):
        # offset分两种情况，当只有x或y方向的offset时，只给直线方向
        # 否则平均分配xy方向的数量
        if self.offset[0] == 0 or self.offset[1] == 0:
            for i in range(self.task_num):
                offset = np.array(self.offset) * i
                self.sub_envs.append(self.sub_env(self.p, offset, self.args))
            return
        else:
            env_num = 0
            sqrt_num = int(math.ceil(math.sqrt(self.task_num)))
            # offset set
            stepX = self.offset[0]
            stepY = self.offset[1]
            for i in range(sqrt_num):
                for j in range(sqrt_num):
                    offset = np.array([stepX*i, stepY*j, self.offset[2]])
                    self.sub_envs.append(self.sub_env(self.p, offset, self.args))
                    env_num += 1
                    if env_num >= self.task_num:
                        return


    # 机械臂根据action执行动作，通过calculateInverseKinematics解算关节位置
    # 多个智能体完成任务的速度不同，当任务完成(done=True)时，不再需要step，所有信息进行保留，满足数据格式要求
    def step(self,action):
        for i in range(self.task_num):
            if not self.dones[i]:
                observation, reward, done, info = self.sub_envs[i].step(action[i])
                self.observations[i]=observation
                self.rewards[i] = reward
                self.dones[i] = done
                self.infos[i] = info
        self.p.stepSimulation()
        # quick reset for test
        if self.is_test:
            test_mode('r', self.reset)
        return self.observations, self.rewards, self.dones, self.infos


    # 神经网络输入图像信息来进行训练
    def render(self, mode='rgb_array'):
        for i in range(self.task_num):
            self.sub_envs[i].render(mode)


    def reset(self):
        self.p.resetSimulation()
        # reset task
        self.observations = self.observation_space.sample()
        self.rewards = [0. for _ in range(self.task_num)]
        self.infos = [{} for _ in range(self.task_num)]
        self.dones = [False for _ in range(self.task_num)]
        for i in range(self.task_num):
            self.observations[i] = self.sub_envs[i].reset()
        return self.observations


    def close(self):
        self.p.disconnect()


if __name__ == '__main__':
    # create_env()
    pass
