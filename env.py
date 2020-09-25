import os
import pybullet as p
import pybullet_data
import gym
import math
import random
import numpy as np
from gym import error,spaces,utils
from gym.utils import seeding


class PandaEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self):
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,\
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space=spaces.Box(np.array([-1]*8),np.array([1]*8)) # 末端7维信息+手指1维
        self.observation_space=spaces.Box(np.array([-1]*9),np.array([1]*9)) # [夹爪1值 夹爪2值 末端位置x y z 四元数]

    # 机械臂根据action执行动作，通过calculateInverseKinematics解算关节位置
    # observation,info分别表示机械臂、目标物体的位置
    def step(self,action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        dv=0.005
        dw=0.005

        dx=action[0]*dv
        dy=action[1]*dv
        dz=action[2]*dv
        orientation=[i*dw for i in action[3:7]]
        fingers=action[7]

        currentPose=p.getLinkState(self.pandaUid,11)
        currentPosition=currentPose[0]
        newPosition=[currentPosition[0]+dx,
                     currentPosition[1]+dy,
                     currentPosition[2]+dz]
        jointPoses=p.calculateInverseKinematics(self.pandaUid,11,newPosition,orientation)[0:7]
        p.setJointMotorControlArray(self.pandaUid,list(range(7))+[9,10],p.POSITION_CONTROL,list(jointPoses)+2*[fingers])
        p.stepSimulation()

        state_object,_=p.getBasePositionAndOrientation(self.objectUid)
        state_robot=p.getLinkState(self.pandaUid,11)[0]
        state_fingers=(p.getJointState(self.pandaUid,9)[0],p.getJointState(self.pandaUid,10)[0])

        # need test
        reward = 0
        done = False

        info=state_object
        observation=state_robot+state_fingers
        return observation,reward,done,info

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.8)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeUid=p.loadURDF("plane.urdf",basePosition=[0,0,-0.65])
        rest_poses=[0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid=p.loadURDF("franka_panda/panda.urdf",useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i,rest_poses[i])
        tableUid=p.loadURDF("table/table.urdf",basePosition=[0.5,0,-0.65])

        # soft object add here
        state_object=[random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid=p.loadURDF("random_urdfs/000/000.urdf",basePosition=state_object)

        state_robot=p.getLinkState(self.pandaUid,11)[0]
        state_fingers=(p.getJointState(self.pandaUid,9)[0],p.getJointState(self.pandaUid,10)[0])
        observation=state_robot+state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return observation


    # 神经网络输入图像信息来进行训练
    def render(self,mode='human'):
        view_matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                        distance=.7,
                                                        yaw=90,
                                                        pitch=-70,
                                                        roll=0,upAxisIndex=2)
        proj_matrix=p.computeProjectionMatrixFOV(fov=60,aspect=float(960)/720,
                                                 nearVal=0.1,
                                                 farVal=100.0)
        (_,_,px,_,_)=p.getCameraImage(width=960,height=720,
                                      viewMatrix=view_matrix,
                                      projectionMatrix=proj_matrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array=np.array(px,dtype=np.uint8)
        rgb_array=np.reshape(rgb_array,(720,960,4))
        rgb_array=rgb_array[:,:,:3]
        return rgb_array

    def close(self):
        p.disconnect()


if __name__ == '__main__':
    # create_env()
    pass
