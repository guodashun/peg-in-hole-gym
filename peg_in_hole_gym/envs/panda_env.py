import pybullet as p
import gym
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from .peg_in_hole import PegInHole
from .random_fly import RandomFly
from .utils import test_mode, MultiAgentActionSpace, MultiAgentObservationSpace

task_list = {'peg-in-hole':PegInHole, 
             'random-fly':RandomFly,
            }

class PandaEnv(gym.Env):
    metadata = {'render.modes':['human', 'rgb_array']}
    def __init__(self, client, task='peg-in-hole', task_num = 1, offset = [0,0,0], is_test=False):
        assert task in task_list
        self.task = task
        self.task_num = task_num
        self.offset = offset
        self.sub_env = task_list[self.task]
        # self.client = client
        self.p = BulletClient(client)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        self.sub_envs = []
        for i in range(task_num): 
            # offset set
            offset = np.array(self.offset) * i
            self.sub_envs.append(self.sub_env(self.p, offset))


        self.action_space = MultiAgentActionSpace([self.sub_env.action_space for i in range(task_num)])
        self.observation_space = MultiAgentObservationSpace([self.sub_env.observation_space for i in range(task_num)])
        
        # test_mode
        self.is_test = is_test


    # 机械臂根据action执行动作，通过calculateInverseKinematics解算关节位置
    def step(self,action):
        observations = []
        rewards = []
        infos = []
        for i in range(self.task_num):
            observation, reward, done, info = self.sub_envs[i].step(action[i])
            observations.append(observation)
            rewards.append(reward)
            self.dones[i] = done
            infos.append(info)
        self.p.stepSimulation()
        all_done = all(self.dones)
        # quick reset for test
        if self.is_test:
            test_mode('r', self.reset)
        return observations, rewards, all_done, infos


    # 神经网络输入图像信息来进行训练
    def render(self,mode='rgb_array'):
        panda_position =self.p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0]
        # view_matrix=self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[panda_position[0],
        #                                                                       panda_position[1],
        #                                                                       panda_position[2]-0.5],
        #                                                 distance=.7,
        #                                                 yaw=0,
        #                                                 pitch=80,
        #                                                 roll=0,upAxisIndex=2)
        view_matrix = self.p.computeViewMatrix(cameraEyePosition=[panda_position[0],
                                                             panda_position[1],
                                                             panda_position[2]],
                                          cameraTargetPosition=[panda_position[0],
                                                                panda_position[1],
                                                                panda_position[2]-10],
                                          cameraUpVector=[0,1,0],          
        )
        proj_matrix=self.p.computeProjectionMatrixFOV(fov=60,aspect=float(self.input_rgb_shape[1]/self.input_rgb_shape[0]),
                                                 nearVal=0.001,
                                                 farVal=1000.0)
        (_,_,r,d,_)=self.p.getCameraImage(width=self.input_rgb_shape[1],height=self.input_rgb_shape[0],
                                      viewMatrix=view_matrix,
                                      projectionMatrix=proj_matrix,
                                      renderer=self.p.ER_TINY_RENDERER)
        rgb_array = np.array(r,dtype=np.uint8)

        dpt_array = np.array(d) #, dtype=np.uint8 # need fix
        # # raw camera shape is (720, 960, 4)
        # # center crop img
        # rgb_array = rgb_array[int((720 - self.input_rgb_shape[0])/2):int(self.input_rgb_shape[0]+(720 - self.input_rgb_shape[0])/2),
        #                       int((960 - self.input_rgb_shape[1])/2):int(self.input_rgb_shape[1]+(960 - self.input_rgb_shape[1])/2),
        #                       :self.input_rgb_shape[2]]
        # dpt_array = dpt_array[int((720 - self.input_rgb_shape[0])/2):int(self.input_rgb_shape[0]+(720 - self.input_rgb_shape[0])/2),
        #                       int((960 - self.input_rgb_shape[1])/2):int(self.input_rgb_shape[1]+(960 - self.input_rgb_shape[1])/2)]
        # reshape img
        rgb_array = rgb_array[:,:,:3]
        res =   np.concatenate(
                    (dpt_array.reshape((dpt_array.shape[0],dpt_array.shape[1],1)),
                    rgb_array),
                    2
                )
        return res


    def reset(self):
        self.p.resetSimulation()

        # reset task
        self.dones = [False for _ in range(self.task_num)]
        observations = []
        for i in range(self.task_num):
            observations.append(self.sub_envs[i].reset())
          
        return observations


    def close(self):
        self.p.disconnect()


if __name__ == '__main__':
    # create_env()
    pass
