import os
from posix import POSIX_FADV_WILLNEED
import posix
import time
from numpy.core.fromnumeric import trace
import pybullet as p
import pybullet_data
import gym
import math
import random
import numpy as np
from gym import spaces
from gym.utils import seeding


class PandaEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, client):
        p.connect(client)
        p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space=spaces.Box(np.array([-1]*4),np.array([1]*4)) # 末端3维信息+手指1维 (默认朝下)
        self.observation_space=spaces.Box(np.array([-1]*5),np.array([1]*5)) # [夹爪1值 夹爪2值 末端位置x y z]
        
        # panda init
        self.pandaUid = 0
        self.pandaEndEffectorIndex = 11
        self.pandaNumDofs = 7
        
        # pipe init
        self.objectUid = 1
        self.object_joints_num = 1

        # hole init
        self.holeUid = 2
        self.hole_state = [0,0,0]

        # grasp state init
        self.dv = 0.05
        self.t = 0.
        self.timeStep = 1./240
        self.cur_state = 0
        self.state_t = 0
        self.stateDurations = [0.25,3,2,1,1.5,1.5,0.5,0.25,0.25,0.25]
        self.done = False
        self.reset()

    # 机械臂根据action执行动作，通过calculateInverseKinematics解算关节位置
    # observation,info分别表示机械臂、目标物体的位置
    def step(self,action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation=p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv=0.3
        dx=action[0]*dv
        dy=action[1]*dv
        dz=action[2]*dv
        fingers=action[3]

        currentPose=p.getLinkState(self.pandaUid,11)
        currentPosition=currentPose[0]
        newPosition=[currentPosition[0]+dx,
                     currentPosition[1]+dy,
                     currentPosition[2]+dz]
        jointPoses=p.calculateInverseKinematics(self.pandaUid,self.pandaEndEffectorIndex,newPosition,orientation)[0:7]
        p.setJointMotorControlArray(self.pandaUid,list(range(self.pandaNumDofs))+[9,10],p.POSITION_CONTROL,list(jointPoses)+2*[fingers])
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

    # 随机抓取软管的关节，并返回抓取的位置及可行性
    def random_grasp(self):
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # calculate target pos and dir
        # grasp_joint_idx = random.randint(0, self.object_joints_num)
        grasp_joint_idx = random.choice([0,23])
        random_vector = [1, 0, 0] # random.uniform(-0.5, 0.5)
        # start grasping
        while(self.done==False): # 
            time.sleep(1./240)
            p.stepSimulation()
            self.render()
            # p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
            #                     cameraPitch=-40,cameraTargetPosition=p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0])
            # switch state
            self.update_state()
            targetPos, targetOrn = p.getLinkState(self.objectUid, grasp_joint_idx)[0:2]
            # origin_pos = targetPos
            # targetPos = [targetPos[0] + random_vector[0], 
            #              targetPos[1] + random_vector[1],
            #              targetPos[2] + random_vector[2]]
            # # rotate_orn = p.getEulerFromQuaternion(targetOrn)
            # # rotate_orn = p.getQuaternionFromEuler([rotate_orn[0],
            # #                                        rotate_orn[1],
            # #                                        rotate_orn[2]])
            # rotate_vector = self.rotate_vector(random_vector, targetOrn)
            # targetPos = [targetPos[0] + rotate_vector[0], 
            #              targetPos[1] + rotate_vector[1],
            #              targetPos[2] + rotate_vector[2]]
            # jointPos = p.getJointState(self.objectUid, grasp_joint_idx)[0:3]
            # print("fuck", jointPos)

            # p.addUserDebugLine(jointPos, targetPos,lineColorRGB=[255,0,0], lineWidth=0.3, lifeTime=1.0)
            self.grasp_process(self.cur_state, targetPos, targetOrn)
            # self.render()

        self.done = False
        pos, orn = p.getLinkState(self.objectUid, grasp_joint_idx)[0:2]
        q = 1. if 1 else 0.

        return pos, orn, 0.2

    def grasp_process(self, state, targetPos, targetOrn):
        currentPos = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0]
        
        targetPos = self.smooth_vel(currentPos, targetPos)
        targetOrn = p.getEulerFromQuaternion(targetOrn)

        # gripper width init
        if state == 0:
            for i in [9,10]:
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, 0.02 ,force= 20)

        # gripper moving to grasping-target x,y 
        if state == 1:
            jointPoses = p.calculateInverseKinematics(
                    self.pandaUid, self.pandaEndEffectorIndex, 
                    targetPos+np.array([0,0,0.05]), p.getQuaternionFromEuler([0,-math.pi,math.pi/2.+targetOrn[2]])
                )
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240.)

        # gripper moving to grasping-target z
        if state == 2:
            jointPoses = p.calculateInverseKinematics(
                    self.pandaUid, self.pandaEndEffectorIndex, 
                    targetPos+np.array([0,0,0]), p.getQuaternionFromEuler([0,-math.pi,math.pi/2.+targetOrn[2]])
                )
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
        
        # grasping
        if state == 3:
            for i in [9,10]:
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, 0.007 ,force= 2000)
        
        # lift
        if state == 4:
            targetPos = [self.hole_state[0] - 0.2, self.hole_state[1], self.hole_state[2]]
            targetPos = self.smooth_vel(currentPos, targetPos)
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, targetPos, 
                p.getQuaternionFromEuler([0.,-math.pi,0]))
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240)


        # gripper moving to manipulation-target (y,z)
        if state == 5:
            targetPos = [self.hole_state[0] - 0.04, self.hole_state[1], self.hole_state[2]]
            targetPos = self.smooth_vel(currentPos, targetPos)
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, p.getQuaternionFromEuler([0.,-math.pi,0]))
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
            

        # gripper moving to manipulating-target z
        if state == 6:
            targetPos = self.hole_state
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, p.getQuaternionFromEuler([0.,-math.pi,0]))
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240.)

        # loosing
        if state == 7:
            for i in [9,10]:
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, 0.02 ,force= 20)


        # completion
        if state == 8:
            targetPos = [0.2, -0.6, 0.4]
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.]))
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240.)

        # reset
        if state == 9:
            self.done = True


    def update_state(self):
        self.state_t += self.timeStep
        if self.state_t > self.stateDurations[self.cur_state]:
            self.cur_state += 1
            self.state_t = 0
            if self.cur_state >= len(self.stateDurations):
                self.cur_state = 0

    def smooth_vel(self, cur, tar):
        res = []
        for i in range(len(tar)):
            diff = tar[i] - cur[i]
            re = 0
            if abs(diff) > self.dv:
                re = cur[i] + (self.dv if diff > 0 else -self.dv)
            else:
                re = cur[i] + diff
            res.append(re)
        return res

    def rotate_vector(self, vec, qua):
        if type(vec) == 'list':
            vec = np.array(vec)
        x, y, z, w = qua
        rotate_matrix = np.array([1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w,
                                  2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w,
                                  2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]).reshape(3,3)
        vec = rotate_matrix @ vec @ np.linalg.inv(rotate_matrix)
        # vec =  np.linalg.inv(rotate_matrix) @ vec

        return vec.tolist()
    
    def reset(self):
        p.resetSimulation()
        self.done = False
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.8)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # planeUid=p.loadURDF("plane.urdf",basePosition=[0,0,-0.65])
        rest_poses=[0,-0.215,-math.pi/3,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid=p.loadURDF("franka_panda/panda.urdf",baseOrientation=p.getQuaternionFromEuler([0, 0, -math.pi/2]),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i,rest_poses[i])
        tableUid=p.loadURDF("table/table.urdf",basePosition=[0.0,-0.5,-1.3], baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi/2]), globalScaling=2)

        # soft pipe init
        state_object=[random.uniform(-0.2, 0.2), random.uniform(-0.4, -0.6), 0.11]# xyz
        self.objectUid=p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/pipe.urdf"),
                                  basePosition=state_object, baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
			                      useFixedBase=0, flags=p.URDF_USE_SELF_COLLISION, globalScaling=0.01)
        self.object_joints_num = p.getNumJoints(self.objectUid)
        for i in random.sample(range(self.object_joints_num), random.randint(5, self.object_joints_num)):
            p.resetJointState(self.objectUid, i, random.uniform(0, math.pi / 3))

        # hole init
        self.hole_state = [0.5, -0.2, 0.2]
        self.holeUid = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/hole.urdf"), 
                                  basePosition=self.hole_state, baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi/2]),
                                  useFixedBase=1, globalScaling=0.013)

        state_robot=p.getLinkState(self.pandaUid,11)[0]
        state_fingers=(p.getJointState(self.pandaUid,9)[0],p.getJointState(self.pandaUid,10)[0])
        observation=state_robot+state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return observation


    # 神经网络输入图像信息来进行训练
    def render(self,mode='human'):
        panda_position =p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0]
        view_matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[panda_position[0],
                                                                              panda_position[1],
                                                                              panda_position[2]-0.5],
                                                        distance=.7,
                                                        yaw=0,
                                                        pitch=-70,
                                                        roll=0,upAxisIndex=2)
        proj_matrix=p.computeProjectionMatrixFOV(fov=60,aspect=float(960)/720,
                                                 nearVal=0.01,
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
