import pybullet as p
import numpy as np
import pybullet_data
import math
import gym
import random


def test_mode(test_key, func):
    keys = p.getKeyboardEvents()
    if len(keys)>0:
        for k,v in keys.items():
            if v & p.KEY_WAS_TRIGGERED:
                if (k==ord(test_key)):
                    func()

def data_normalize(data, normalize_range):
    for i in range(len(data)):
        data[i] = (data[i] - normalize_range[i][0]) / (normalize_range[i][1] - normalize_range[i][0])
    return data

def init_panda(client, panda_pos, panda_orn, table_pos, flags=0):
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    panda_id=client.loadURDF("franka_panda/panda.urdf",basePosition=panda_pos,
                                baseOrientation=client.getQuaternionFromEuler([0, 0, -math.pi/2]),useFixedBase=True,
                                flags=flags)
    for i in range(len(panda_orn)):
        client.resetJointState(panda_id,i,panda_orn[i])
    table_id=client.loadURDF("table/table.urdf",basePosition=table_pos, 
                            baseOrientation=client.getQuaternionFromEuler([0, 0, math.pi/2]), globalScaling=2,
                            flags=flags)
    return panda_id, table_id

def reset_panda(client, panda_id, panda_orn):
    for i in range(7):
        client.resetJointState(panda_id,i,panda_orn[i])


def panda_execute(client, panda_id, action, pandaEndEffectorIndex, pandaNumDofs, dv=2/240.):
    client.configureDebugVisualizer(client.COV_ENABLE_SINGLE_STEP_RENDERING)
    orientation=client.getQuaternionFromEuler([0.,-math.pi,0.])
    currentPose=client.getLinkState(panda_id,pandaEndEffectorIndex)
    currentPosition=currentPose[0]
    
    newPosition = vel_constraint(currentPosition, action[:3], dv)
    fingers= len(action) > 3 and action[3] or 0.
    jointPoses=client.calculateInverseKinematics(panda_id,pandaEndEffectorIndex,newPosition,orientation)[0:7]
    client.setJointMotorControlArray(panda_id,list(range(pandaNumDofs))+[9,10],client.POSITION_CONTROL,list(jointPoses)+2*[fingers], positionGains=[1]*9) 


def vel_constraint(cur, tar, dv):
    res = []
    for i in range(len(tar)):
        diff = tar[i] - cur[i]
        re = 0
        if abs(diff) > dv:
            re = cur[i] + (dv if diff > 0 else - dv)
        else:
            re = cur[i] + diff
        res.append(re)
    return res

def random_pos_in_panda_space():
    # |x|,|y| < 0.8, 0 < z < 1
    # x^2 + y^2 + (z-0.2)^2 < 0.8^2
    x = y = z = 1
    length = 0.7
    while (length*length - x*x - y*y) < 0:
        x = random.uniform(-length, length)
        y = (math.sqrt(random.uniform(0, length*length-x*x))-random.uniform(0, 0.4))*random.choice([-1,1])
    z = math.sqrt(length*length - x*x - y*y) + 0.2
    res = np.array([x,y,z])
    return res


def translate(data, diff):
    for i in range(len(data)):
        data[i] = data[i] - diff


def rotate_2d(point, center, theta):
    x = point[0] - center[0]
    y = point[1] - center[1]
    new_x = x * math.cos(theta) - y * math.sin(theta) 
    new_y = x * math.sin(theta) + y * math.cos(theta)
    point[0] = new_x + center[0]
    point[1] = new_y + center[1]


def rotate_3d(vec, qua):
    x, y, z, w = qua
    rotate_matrix = np.array([1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w,
                                2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w,
                                2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]).reshape(3,3)
    vec = rotate_matrix @ vec 


class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space
        self.shape = agents_observation_space[0].shape
        self.high = agents_observation_space[0].high
        self.low = agents_observation_space[0].low

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space
        self.shape = agents_action_space[0].shape
        self.high = agents_action_space[0].high
        self.low = agents_action_space[0].low

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]


class MPMultiAgentObservationSpace(MultiAgentObservationSpace):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, MultiAgentObservationSpace)

        # super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space
        self.shape = agents_observation_space[0].shape
        self.high = agents_observation_space[0].high
        self.low = agents_observation_space[0].low

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True


class MPMultiAgentActionSpace(MultiAgentActionSpace):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, MultiAgentActionSpace)

        # super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space
        self.shape = agents_action_space[0].shape
        self.high = agents_action_space[0].high
        self.low = agents_action_space[0].low

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]
