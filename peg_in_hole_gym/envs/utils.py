import pybullet as p
import numpy as np
import pybullet_data
import math
import gym


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

def reset_panda(client, panda_pos, panda_orn, table_pos, flags):
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    # rest_poses=[0,-0.215,-math.pi/3,-2.57,0,2.356,2.356,0.08,0.08]
    panda_id=client.loadURDF("franka_panda/panda.urdf",basePosition=panda_pos,
                                    baseOrientation=client.getQuaternionFromEuler([0, 0, -math.pi/2]),useFixedBase=True,
                                    flags=flags)
    for i in range(7):
        client.resetJointState(panda_id,i,panda_orn[i])
    table_id=client.loadURDF("table/table.urdf",basePosition=table_pos, 
                                baseOrientation=client.getQuaternionFromEuler([0, 0, math.pi/2]), globalScaling=2,
                                flags=flags)
    return panda_id, table_id

def panda_execute(client, panda_id, action, pandaEndEffectorIndex, pandaNumDofs):
    client.configureDebugVisualizer(client.COV_ENABLE_SINGLE_STEP_RENDERING)
    orientation=client.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
    dv=0.1
    dx=action[0]*dv
    dy=action[1]*dv
    dz=action[2]*dv
    fingers= len(action) > 3 and action[3] or 0.

    currentPose=client.getLinkState(panda_id,pandaEndEffectorIndex)
    currentPosition=currentPose[0]
    newPosition=[currentPosition[0]+dx,
                 currentPosition[1]+dy,
                 currentPosition[2]+dz]
    jointPoses=client.calculateInverseKinematics(panda_id,pandaEndEffectorIndex,newPosition,orientation)[0:7]
    client.setJointMotorControlArray(panda_id,list(range(pandaNumDofs))+[9,10],client.POSITION_CONTROL,list(jointPoses)+2*[fingers])

        
class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

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

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]


class MPMultiAgentObservationSpace(MultiAgentObservationSpace):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, MultiAgentObservationSpace)

        # super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

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

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]
