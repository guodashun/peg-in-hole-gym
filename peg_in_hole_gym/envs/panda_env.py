import os
import time
import pybullet as p
import pybullet_data
import gym
import math
import random
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from gym import spaces
from queue import Queue
from PIL import Image, ImageDraw
from skimage.draw import polygon
from sklearn.preprocessing import MinMaxScaler


class PandaEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, client, task='peg-in-hole', is_test=False):
        assert task in ['peg-in-hole', 'random-fly']
        self.task = task
        self.client = client
        self.p = BulletClient(client)
        # self.p.connect(self.client)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        if self.task == "random-fly":
            self.action_space=spaces.Box(np.array([-1]*3),np.array([1]*3)) # 末端3维信息
            self.observation_space = spaces.Box(np.array([-1]*12), np.array([1]*12)) # [物体位置+速度 末端位置+速度]
        else:
            self.action_space=spaces.Box(np.array([-1]*4),np.array([1]*4)) # 末端3维信息+手指1维 (默认朝下)
            self.observation_space=spaces.Box(np.array([-1]*5),np.array([1]*5)) # [夹爪1值 夹爪2值 末端位置x y z]
        
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
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_SINGLE_STEP_RENDERING)
            orientation=self.p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
            dv=0.3
            dx=action[0]*dv
            dy=action[1]*dv
            dz=action[2]*dv
            fingers=[] if self.task == 'random-fly' else action[3]

            currentPose=self.p.getLinkState(self.pandaUid,11)
            currentPosition=currentPose[0]
            newPosition=[currentPosition[0]+dx,
                         currentPosition[1]+dy,
                         currentPosition[2]+dz]
            jointPoses=self.p.calculateInverseKinematics(self.pandaUid,self.pandaEndEffectorIndex,newPosition,orientation)[0:7]
            if self.task == 'random-fly':
                self.p.setJointMotorControlArray(self.pandaUid,list(range(self.pandaNumDofs)),self.p.POSITION_CONTROL,list(jointPoses))
            else:
                self.p.setJointMotorControlArray(self.pandaUid,list(range(self.pandaNumDofs))+[9,10],self.p.POSITION_CONTROL,list(jointPoses)+2*[fingers])
            self.p.stepSimulation()

        # observation
        observation = []
        reward = 0
        done = self.done
        info = {}
        if self.task == 'peg-in-hole':
            info, reward = self.random_grasp()
            observation = self.grasp_img
        if self.task == 'random-fly':
            observation, reward, success = self.random_fly()
            info['success'] = success
        # quick reset for test
        self.test_mode()
        return observation, reward, done, info


    # peg-in-hole scene
    def random_grasp(self):
        # init img
        pos_img = np.zeros(self.output_shape)
        ang_img = np.zeros(self.output_shape)
        wid_img = np.zeros(self.output_shape)
        sin_img = np.sin(2*ang_img)
        cos_img = np.cos(2*ang_img)
        x = 0.0
        y = 0.0
        angle = 0.0
        width = 0.0
        length = 0.0

        # start grasping
        while(self.done==False): 
            # switch state
            self.update_state()

            # calculate random grasp pos
            rawPos, targetOrn = self.p.getLinkState(self.objectUid, self.grasp_joint_idx)[0:2]
            rotate_vector = self.rotate_vector(self.random_vector, targetOrn)
            targetPos = [rawPos[0] + rotate_vector[0], 
                         rawPos[1] + rotate_vector[1],
                         rawPos[2] + rotate_vector[2]]
            
            if self.last_state != self.cur_state:
                # capture the grasp img
                if self.cur_state == 2:
                    self.grasp_img = self.render()
                    # projection on x-y plain
                    # pos = self.p.getLinkState(self.objectUid, self.grasp_joint_idx)[0]
                    # pos = [pos[0], pos[1]]
                    camera_pos = self.p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0]
                    camera_pos = [camera_pos[0], camera_pos[1]]
                    relative_pos = [(targetPos[0] - camera_pos[0])*self.input_rgb_shape[0], (targetPos[1] - camera_pos[1])*self.input_rgb_shape[1]]
                    # pos is (0, 0)
                    angle = math.atan2(rotate_vector[1], rotate_vector[0])
                    length = 0.1 # random
                    width = 0.2
                    # img = Image.fromarray(pos_img)
                    # draw = ImageDraw.Draw(img)
                    # draw.line([(0,0), ((0.5+relative_pos[0])*self.input_rgb_shape[0], (0.5+relative_pos[1])*self.input_rgb_shape[1])], fill=(255,0,0))
                    a = ( (1. + length * math.cos(angle) + width * math.sin(angle)) / 2 * self.output_shape[0], (1. - length * math.sin(angle) + width * math.cos(angle)) / 2 * self.output_shape[1])
                    b = ( (1. - length * math.cos(angle) - width * math.sin(angle)) / 2 * self.output_shape[0], (1. + length * math.sin(angle) - width * math.cos(angle)) / 2 * self.output_shape[1])
                    c = ( (1. - length * math.cos(angle) + width * math.sin(angle)) / 2 * self.output_shape[0], (1. + length * math.sin(angle) + width * math.cos(angle)) / 2 * self.output_shape[1])
                    d = ( (1. + length * math.cos(angle) - width * math.sin(angle)) / 2 * self.output_shape[0], (1. - length * math.sin(angle) - width * math.cos(angle)) / 2 * self.output_shape[1])
                    R = [a[0], b[0], c[0], d[0]]
                    C = [a[1], b[1], c[1], d[0]]
                    rr, cc = polygon(R, C, shape=self.output_shape)
                    pos_img[rr,cc] = 1.0
                    ang_img[rr,cc] = angle
                    wid_img[rr,cc] = width * self.output_shape[0]
                    # draw.polygon([(100,100), (100,200), (200, 200), (200,100)],outline=(255,0,0))
                    # draw.polygon([a,c,b,d], outline=25)
                    # img.show()
                    # cos_o = targetPos[0]/math.sqrt(targetPos[0]*targetPos[0] + targetPos[1]*targetPos[1])
                    # sin_o = targetPos[1]/math.sqrt(targetPos[0]*targetPos[0] + targetPos[1]*targetPos[1])
                    sin_img = np.sin(2*ang_img)
                    cos_img = np.cos(2*ang_img)
                if self.cur_state == 4:
                    self.constraintUid = self.p.createConstraint(self.pandaUid, self.pandaEndEffectorIndex, self.objectUid, self.grasp_joint_idx, self.p.JOINT_GEAR, 
                                                            [0,0,0], [0,0,0], self.random_vector, childFrameOrientation=self.p.getQuaternionFromEuler([0,-math.pi,math.pi/2.+targetOrn[2]])
                                                           )
                if self.cur_state == 7:
                    self.p.removeConstraint(self.constraintUid)

            # self.p.addUserDebugLine(rawPos, targetPos,lineColorRGB=[255,0,0], lineWidth=3, lifeTime=1.0)
            self.grasp_process(self.cur_state, targetPos, targetOrn)
            self.p.stepSimulation()
            if self.client == p.GUI:
                time.sleep(1./240)
            self.last_state = self.cur_state

        self.done = False
        threshold = 0.05 # test result
        q = 1. if np.linalg.norm(np.array(self.p.getLinkState(self.objectUid, self.grasp_joint_idx)[0])-
                                 np.array(self.p.getBasePositionAndOrientation(self.holeUid)[0])) < threshold \
               else 0.
        # if self.is_test:
        #     print("rawPos", self.p.getLinkState(self.objectUid, self.grasp_joint_idx)[0], "holePos", self.p.getBasePositionAndOrientation(self.holeUid)[0],
        #         "result",np.linalg.norm(np.array(self.p.getLinkState(self.objectUid, self.grasp_joint_idx)[0])-
        #                             np.array(self.p.getBasePositionAndOrientation(self.holeUid)[0])),
        #         "q", q)
        return [[pos_img, sin_img, cos_img, wid_img],[x,y,angle/math.pi*180.,width,length]], q

    def grasp_process(self, state, targetPos, targetOrn):
        currentPos = self.p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0]
        
        targetPos = self.smooth_vel(currentPos, targetPos)
        targetOrn = self.p.getEulerFromQuaternion(targetOrn)

        # gripper width init
        if state == 0:
            for i in [9,10]:
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, 0.02 ,force= 20)

        # gripper moving to grasping-target x,y 
        if state == 1:
            jointPoses = self.p.calculateInverseKinematics(
                    self.pandaUid, self.pandaEndEffectorIndex, 
                    targetPos+np.array([0,0,0.05]), self.p.getQuaternionFromEuler([0,-math.pi,math.pi/2.+targetOrn[2]])
                )
            for i in range(self.pandaNumDofs):
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, jointPoses[i], force=5.*240.)

        # gripper moving to grasping-target z
        if state == 2:
            jointPoses = self.p.calculateInverseKinematics(
                    self.pandaUid, self.pandaEndEffectorIndex, 
                    targetPos+np.array([0,0,-0.01]), self.p.getQuaternionFromEuler([0,-math.pi,math.pi/2.+targetOrn[2]])
                )
            for i in range(self.pandaNumDofs):
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
        
        # grasping
        if state == 3:
            for i in [9,10]:
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, 0.006 ,force= 20000)
        
        # lift
        if state == 4:
            targetPos = [self.hole_state[0] - 0.2, self.hole_state[1], self.hole_state[2]]
            targetPos = self.smooth_vel(currentPos, targetPos)
            jointPoses = self.p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, targetPos, 
                self.p.getQuaternionFromEuler([0.,-math.pi,-math.pi]))
            for i in range(self.pandaNumDofs):
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, jointPoses[i], force=5.*240)


        # gripper moving to manipulation-target (y,z)
        if state == 5:
            targetPos = [self.hole_state[0] - 0.04, self.hole_state[1], self.hole_state[2]]
            targetPos = self.smooth_vel(currentPos, targetPos)
            jointPoses = self.p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, self.p.getQuaternionFromEuler([0.,-math.pi,-math.pi]))
            for i in range(self.pandaNumDofs):
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
            

        # gripper moving to manipulating-target z
        if state == 6:
            targetPos = self.hole_state
            jointPoses = self.p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, self.p.getQuaternionFromEuler([0.,-math.pi,-math.pi]))
            for i in range(self.pandaNumDofs):
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, jointPoses[i], force=5.*240.)

        # loosing
        if state == 7:
            for i in [9,10]:
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, 0.02 ,force= 20)


        # completion
        if state == 8:
            targetPos = [0.2, -0.6, 0.4]
            jointPoses = self.p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex, 
                targetPos, self.p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.]))
            for i in range(self.pandaNumDofs):
                self.p.setJointMotorControl2(self.pandaUid, i, self.p.POSITION_CONTROL, jointPoses[i], force=5.*240.)

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
            keys = self.p.getKeyboardEvents()
            if len(keys)>0:
                for k,v in keys.items():
                    if v & self.p.KEY_WAS_TRIGGERED:
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
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5,cameraYaw=0,
                                     cameraPitch=-40,cameraTargetPosition=[0.55,-0.35,0.2])
        state_object=[random.uniform(-0.2, 0.2), random.uniform(-0.4, -0.6), 0.11]# xyz
        self.objectUid=self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/pipe.urdf"),
                                  basePosition=state_object, baseOrientation=self.p.getQuaternionFromEuler([0, 0, 0]),
			                      useFixedBase=0, flags=self.p.URDF_USE_SELF_COLLISION, globalScaling=0.01)
        self.object_joints_num = self.p.getNumJoints(self.objectUid)
        for i in random.sample(range(self.object_joints_num), random.randint(5, self.object_joints_num)):
            self.p.resetJointState(self.objectUid, i, random.uniform(0, math.pi / 3))

        # hole init
        self.hole_state = [0.5, -0.2, 0.2]
        self.holeUid = self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/hole.urdf"), 
                                  basePosition=self.hole_state, baseOrientation=self.p.getQuaternionFromEuler([0, 0, math.pi/2]),
                                  useFixedBase=1, globalScaling=0.016)
        
        # grasp state init
        self.dv = 0.05
        self.done = False
        self.grasp_img = [0,0,0]
        self.stateDurations = [0.25,2,2,1,1.5,1.5,0.5,0.25,0.25,0.25]

        # calculate target pos and dir
        self.grasp_joint_idx = random.choice([0,23])
        self.random_vector = [0, random.uniform(-0.03, 0.03), 0]
        
        # set image shape
        self.input_rgb_shape = (300, 300, 3)
        self.input_dpt_shape = (300, 300)
        self.output_shape    = (300, 300)

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
            or np.linalg.norm(obj_w) > 2 
           ):
            self.done = True
            # print("why reset", np.linalg.norm(np.array(obj_pos) - np.array(self.object_pos)) < 0.01 
            #         , obj_vel[2] < -15
            #         , np.linalg.norm(obj_w) > 2)
        reward = 0.0
        success = False

        pos_bias = np.linalg.norm(np.array(base_pos) - np.array(obj_pos))
        vel_bias = np.linalg.norm(np.array(base_vel) - np.array(obj_vel))
        move_cst = np.linalg.norm(np.array(base_pos) - np.array(self.last_panda_pos))
        pos_r  = -math.log(pos_bias) * 0.7
        vel_r  = -math.log(vel_bias) * 0.01
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
        res = self.data_normalize(res)

        return res, reward, success

    def reset_fly(self):
        self.p.resetDebugVisualizerCamera(cameraDistance=5,cameraYaw=0,
                                     cameraPitch=-0,cameraTargetPosition=[0,0,0])
        # normalize range
        self.normalize_range = [(-6,6), (-6,6), (-3,16),
                                (-6,6), (-6,6), (-15, 5),
                                (-1,1), (-1,1), (-1,1),
                                (-10, 10), (-10, 10), (-10,10)]

        # init flying object
        target_pos = self.random_pos_in_panda_space()
        x  = random.uniform(4,6) * random.choice([-1., 1.])
        vx = random.uniform(4,6) * (-1. if x>0 else 1.)
        t  = abs((target_pos[0] - x) / vx)
        vy = random.uniform(4,6) * random.choice([-1., 1.])
        vz = random.randint(-2,5)
        y  = target_pos[1] - vy * t
        z  = target_pos[2] - (vz * t +  (self.gravity[2]) * t*t / 2)
        self.object_pos = [x, y, z]
        base_orn = self.p.getQuaternionFromEuler([random.uniform(-math.pi, math.pi) for i in range(3)])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.objectUid = self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/Amicelli_800_tex.urdf"),
                                    basePosition=self.object_pos, baseOrientation=base_orn,
                                    globalScaling=5)
        self.p.changeDynamics(self.objectUid, -1, linearDamping=0, angularDamping=0)
        self.p.resetBaseVelocity(self.objectUid, [vx,vy,vz])

        self.p.addUserDebugLine([0,0,0], target_pos,lineColorRGB=[255,0,0], lineWidth=3)

        obj_pos = self.p.getBasePositionAndOrientation(self.objectUid)[0]
        obj_vel = self.p.getBaseVelocity(self.objectUid)[0]
        base_info = self.p.getLinkState(self.pandaUid,11,computeLinkVelocity=1)
        base_pos = base_info[0]
        base_vel = base_info[6]
        self.last_panda_pos = base_pos

        # add time cost
        self.time_r = 0
        
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
        obs = self.data_normalize(obs)
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

    def data_normalize(self, data):
        for i in range(len(data)):
            data[i] = (data[i] - self.normalize_range[i][0]) / (self.normalize_range[i][1] - self.normalize_range[i][0])
        return data


    # 神经网络输入图像信息来进行训练
    def render(self,mode='human'):
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
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING,0)
        self.gravity = [0,0,-9.8]
        self.p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        rest_poses=[0,-0.215,-math.pi/3,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid=self.p.loadURDF("franka_panda/panda.urdf",baseOrientation=self.p.getQuaternionFromEuler([0, 0, -math.pi/2]),useFixedBase=True)
        for i in range(7):
            self.p.resetJointState(self.pandaUid,i,rest_poses[i])
        tableUid=self.p.loadURDF("table/table.urdf",basePosition=[0.0,-0.5,-1.3], baseOrientation=self.p.getQuaternionFromEuler([0, 0, math.pi/2]), globalScaling=2)

        self.timeStep = 1./240
        self.cur_state = 0
        self.last_state = 0
        self.state_t = 0
        self.stateDurations = [0.25]

        state_robot=self.p.getLinkState(self.pandaUid,11)[0]
        state_fingers=(self.p.getJointState(self.pandaUid,9)[0],self.p.getJointState(self.pandaUid,10)[0])
        observation=state_robot+state_fingers
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING,1)

        # reset task
        self.done = False
        if self.task == 'peg-in-hole':
            self.reset_peg_in_hole()
            observation = self.render()
        if self.task == 'random-fly':
            observation = self.reset_fly()
            
        return observation


    def close(self):
        self.p.disconnect()


if __name__ == '__main__':
    # create_env()
    pass
