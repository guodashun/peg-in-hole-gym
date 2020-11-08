import os
import math
import time
import random
import pybullet as p
import numpy as np
from gym import spaces
from skimage.draw import polygon
from pybullet_utils.bullet_client import BulletClient

class PegInHole(object):
    def __init__(self, offset, client):
        self.offset = np.array(offset)
        self.client = client
        self.p = BulletClient(client)
        self.action_space=spaces.Box(np.array([-1]*4),np.array([1]*4)) # 末端3维信息+手指1维 (默认朝下)
        self.observation_space=spaces.Box(np.array([-1]*5),np.array([1]*5)) # [夹爪1值 夹爪2值 末端位置x y z]

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
                                     cameraPitch=-40,cameraTargetPosition=[0.55+self.offset[0],-0.35+self.offset[1],0.2+self.offset[2]])
        state_object=[random.uniform(-0.2, 0.2)+self.offset[0], random.uniform(-0.4, -0.6)+self.offset[1], 0.11]# xyz
        self.objectUid=self.p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),"assets/urdf/pipe.urdf"),
                                  basePosition=state_object, baseOrientation=self.p.getQuaternionFromEuler([0, 0, 0]),
			                      useFixedBase=0, flags=self.p.URDF_USE_SELF_COLLISION, globalScaling=0.01)
        self.object_joints_num = self.p.getNumJoints(self.objectUid)
        for i in random.sample(range(self.object_joints_num), random.randint(5, self.object_joints_num)):
            self.p.resetJointState(self.objectUid, i, random.uniform(0, math.pi / 3))

        # hole init
        self.hole_state = [0.5+self.offset[0], -0.2+self.offset[1], 0.2+self.offset[2]]
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
