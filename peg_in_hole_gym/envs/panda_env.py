import os
import time
import pybullet as p
import pybullet_data
import gym
import math
import random
import numpy as np
from gym import spaces
from queue import Queue


class PandaEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, client, task='peg-in-hole', is_test=False):
        assert task in ['peg-in-hole', 'random-fly']
        self.task = task
        self.client = client
        p.connect(self.client)
        p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space=spaces.Box(np.array([-1,-1,-1]),np.array([1,1,1])) # 末端3维信息+手指1维 (默认朝下)
        # self.observation_space=spaces.Box(np.array([-1]*5),np.array([1]*5)) # [夹爪1值 夹爪2值 末端位置x y z]
        self.observation_space = spaces.Box(np.array([-30]*12), np.array([30]*12))
        
        # test_mode
        self.is_test = is_test

        # panda init
        self.pandaUid = 0
        self.pandaEndEffectorIndex = 11
        self.pandaNumDofs = 7
        
        self.reset()


    # 机械臂根据action执行动作，通过calculateInverseKinematics解算关节位置
    def step(self,action):
        # excute
        if self.task != 'peg-in-hole':
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            orientation=p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
            dv=0.3
            dx=action[0]*dv
            dy=action[1]*dv
            dz=action[2]*dv
            # fingers=action[3]

            currentPose=p.getLinkState(self.pandaUid,11)
            currentPosition=currentPose[0]
            newPosition=[currentPosition[0]+dx,
                        currentPosition[1]+dy,
                        currentPosition[2]+dz]
            jointPoses=p.calculateInverseKinematics(self.pandaUid,self.pandaEndEffectorIndex,newPosition,orientation)[0:7]
            # p.setJointMotorControlArray(self.pandaUid,list(range(self.pandaNumDofs))+[9,10],p.POSITION_CONTROL,list(jointPoses)+2*[fingers])
            p.setJointMotorControlArray(self.pandaUid,list(range(self.pandaNumDofs)),p.POSITION_CONTROL,list(jointPoses))
            p.stepSimulation()

        # observation
        observation = []
        reward = 0
        done = self.done
        info = {}
        if self.task == 'peg-in-hole':
            observation = self.grasp_img
            info, reward = self.random_grasp()
        if self.task == 'random-fly':
            observation, reward = self.random_fly()
        
        # quick reset for test
        self.test_mode()
        return observation, reward, done, info


    # peg-in-hole scene
    def random_grasp(self):
        # start grasping
        while(self.done==False): 
            # switch state
            self.update_state()

            # calculate random grasp pos
            rawPos, targetOrn = p.getLinkState(self.objectUid, self.grasp_joint_idx)[0:2]
            rotate_vector = self.rotate_vector(self.random_vector, targetOrn)
            targetPos = [rawPos[0] + rotate_vector[0], 
                         rawPos[1] + rotate_vector[1],
                         rawPos[2] + rotate_vector[2]]
            
            if self.last_state != self.cur_state:
                # capture the grasp img
                if self.cur_state == 2:
                    self.grasp_img = self.render()

            # p.addUserDebugLine(rawPos, targetPos,lineColorRGB=[255,0,0], lineWidth=3, lifeTime=1.0)
            self.grasp_process(self.cur_state, targetPos, targetOrn)
            p.stepSimulation()
            if self.client == p.GUI:
                time.sleep(1./240)
            self.last_state = self.cur_state

        self.done = False
        # projection on x-y plain
        pos = p.getLinkState(self.objectUid, self.grasp_joint_idx)[0]
        pos = [pos[0], pos[1]]
        cos_o = pos[0]/math.sqrt(pos[0]*pos[0] + pos[1]*pos[1])
        sin_o = pos[1]/math.sqrt(pos[0]*pos[0] + pos[1]*pos[1])
        width = 0.2
        threshold = 0.05 # test result
        q = 1. if np.linalg.norm(np.array(p.getLinkState(self.objectUid, self.grasp_joint_idx)[0])-
                                 np.array(p.getBasePositionAndOrientation(self.holeUid)[0])) < threshold \
               else 0.
        if self.is_test:
            print("rawPos", p.getLinkState(self.objectUid, self.grasp_joint_idx)[0], "holePos", p.getBasePositionAndOrientation(self.holeUid)[0],
                "result",np.linalg.norm(np.array(p.getLinkState(self.objectUid, self.grasp_joint_idx)[0])-
                                    np.array(p.getBasePositionAndOrientation(self.holeUid)[0])),
                "q", q)
        return [pos, sin_o, cos_o, width], q

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
                    targetPos+np.array([0,0,-0.01]), p.getQuaternionFromEuler([0,-math.pi,math.pi/2.+targetOrn[2]])
                )
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
        
        # grasping
        if state == 3:
            for i in [9,10]:
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, 0.006 ,force= 20000)
        
        # lift
        if state == 4:
            targetPos = [self.hole_state[0] - 0.2, self.hole_state[1], self.hole_state[2]]
            targetPos = self.smooth_vel(currentPos, targetPos)
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, targetPos, 
                p.getQuaternionFromEuler([0.,-math.pi,-math.pi]))
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240)


        # gripper moving to manipulation-target (y,z)
        if state == 5:
            targetPos = [self.hole_state[0] - 0.04, self.hole_state[1], self.hole_state[2]]
            targetPos = self.smooth_vel(currentPos, targetPos)
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, p.getQuaternionFromEuler([0.,-math.pi,-math.pi]))
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
            

        # gripper moving to manipulating-target z
        if state == 6:
            targetPos = self.hole_state
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, p.getQuaternionFromEuler([0.,-math.pi,-math.pi]))
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
        
        self.test_mode()

    def test_mode(self):    
        if self.is_test:
            keys = p.getKeyboardEvents()
            if len(keys)>0:
                for k,v in keys.items():
                    if v & p.KEY_WAS_TRIGGERED:
                        if (k==ord('r')):
                            self.reset()

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
        vec = rotate_matrix @ vec # @ np.linalg.inv(rotate_matrix)
        # vec =  np.linalg.inv(rotate_matrix) @ vec

        return vec.tolist()

    def reset_peg_in_hole(self):
        # soft pipe init
        p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
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
                                  useFixedBase=1, globalScaling=0.016)
        
        # grasp state init
        self.dv = 0.05
        self.done = False
        self.grasp_img = [0,0,0]
        self.stateDurations = [0.25,2,2,1,1.5,1.5,0.5,0.25,0.25,0.25]

        # calculate target pos and dir
        self.grasp_joint_idx = random.choice([0,23])
        self.random_vector = [0, random.uniform(-0.03, 0.03), 0]


    # random fly scene
    def random_fly(self):
        obj_pos = p.getBasePositionAndOrientation(self.objectUid)[0]
        obj_vel, obj_w = p.getBaseVelocity(self.objectUid)[0:2]
        if self.obs_delay:
            self.obs_delay_queue.put([obj_pos, obj_vel])
            obj_pos, obj_vel = self.obs_delay_queue.get()
        base_info = p.getLinkState(self.pandaUid,self.pandaEndEffectorIndex,computeLinkVelocity=1)
        base_pos = base_info[0]
        base_vel = base_info[6]
        # print("fuck", np.linalg.norm(np.array(obj_pos) - np.array(self.object_pos)) < 0.01, (abs(obj_vel[0]) + abs(obj_vel[1])) < 1, obj_vel[2] < -10)
        if np.linalg.norm(np.array(obj_pos) - np.array(self.object_pos)) < 0.01 or \
           (abs(obj_vel[0]) + abs(obj_vel[1])) < 1 or \
           obj_vel[2] < -15 or \
           np.linalg.norm(obj_w) > 0.1:
            self.done = True
        # print("ahaha",base_pos)
        reward = 0.0
        pos_bias = np.linalg.norm(np.array(base_pos) - np.array(obj_pos))
        vel_bias = np.linalg.norm(np.array(base_vel) - np.array(obj_vel))
        pos_r  = -math.log(pos_bias) * 0.7
        vel_r  = -math.log(vel_bias) * 0.01
        cli_r  = 10 if p.getContactPoints(self.pandaUid, self.objectUid, 8) else 0
        self.time_r -= 0.01
        r_sum  = pos_r + vel_r + cli_r + self.time_r
        reward = pos_r + vel_r + cli_r + self.time_r
        print("reward now", r_sum, pos_r, vel_r, self.time_r, cli_r) 
        # if self.is_test:
        #     print("collision points:",cli_r) #, self.pandaEndEffectorIndex
        return list(obj_pos+ obj_vel+ base_pos+ base_vel), reward

    def reset_fly(self):
        p.resetDebugVisualizerCamera(cameraDistance=10,cameraYaw=0,
                                     cameraPitch=-0,cameraTargetPosition=[0,0,0])
        
        # init flying object
        base_pos = self.random_pos_in_panda_space()
        x  = random.uniform(4,6) * random.choice([-1, 1])
        vx = random.uniform(4,6) * (-1. if x>0 else 1.)
        t  = abs((base_pos[0] - x) / vx)
        vy = random.uniform(4,6) * random.choice([-1, 1])
        vz = random.randint(-2,5)
        y  = base_pos[1] - vy * t
        z  = base_pos[2] - (vz * t +  (self.gravity[2]) * t*t / 2)
        self.object_pos = [x, y, z]
        base_orn = p.getQuaternionFromEuler([random.uniform(-math.pi, math.pi) for i in range(3)])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.objectUid = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/Amicelli_800_tex.urdf"),
                                    basePosition=self.object_pos, baseOrientation=base_orn,
                                    globalScaling=5)
        p.changeDynamics(self.objectUid, -1, linearDamping=0, angularDamping=0)
        p.resetBaseVelocity(self.objectUid, [vx,vy,vz])

        p.addUserDebugLine([0,0,0], base_pos,lineColorRGB=[255,0,0], lineWidth=3)

        obj_pos = p.getBasePositionAndOrientation(self.objectUid)[0]
        obj_vel = p.getBaseVelocity(self.objectUid)[0]
        base_info = p.getLinkState(self.pandaUid,11,computeLinkVelocity=1)
        base_pos = base_info[0]
        base_vel = base_info[6]

        # add time cost
        self.time_r = 2
        
        # add delay
        self.obs_delay = True if random.random() > 0.3 else False
        self.obs_delay_queue = Queue(random.randint(1,3))
        print("queue maxsize", self.obs_delay_queue.maxsize)
        i = 0
        while not self.obs_delay_queue.full():
            i += 1
            p.stepSimulation()
            self.obs_delay_queue.put([obj_pos, obj_vel])
            print("haha, delay is work", i)
        obj_pos, obj_vel = self.obs_delay_queue.get()
        obs = list(obj_pos + obj_vel + base_pos + base_vel)
        return obs

    def random_pos_in_panda_space(self):
        # |x|,|y| < 0.8, 0 < z < 1
        # x^2 + y^2 + (z-0.2)^2 < 0.8^2
        x = y = z = 1
        length = 0.8
        while (length*length - x*x - y*y) < 0:
            x = random.uniform(-length, length)
            y = (math.sqrt(random.uniform(0, length*length-x*x))-random.uniform(0, 0.4))*random.choice([-1,1])
        z = math.sqrt(length*length - x*x - y*y) + 0.2
        return [x,y,z]


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


    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        self.gravity = [0,0,-9.8]
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        rest_poses=[0,-0.215,-math.pi/3,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid=p.loadURDF("franka_panda/panda.urdf",baseOrientation=p.getQuaternionFromEuler([0, 0, -math.pi/2]),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i,rest_poses[i])
        tableUid=p.loadURDF("table/table.urdf",basePosition=[0.0,-0.5,-1.3], baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi/2]), globalScaling=2)

        self.timeStep = 1./240
        self.cur_state = 0
        self.last_state = 0
        self.state_t = 0
        self.stateDurations = [0.25]

        state_robot=p.getLinkState(self.pandaUid,11)[0]
        state_fingers=(p.getJointState(self.pandaUid,9)[0],p.getJointState(self.pandaUid,10)[0])
        observation=state_robot+state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        # reset task
        self.done = False
        if self.task == 'peg-in-hole':
            self.reset_peg_in_hole()
            observation = self.render()
        if self.task == 'random-fly':
            observation = self.reset_fly()
            
        return observation


    def close(self):
        p.disconnect()


if __name__ == '__main__':
    # create_env()
    pass
