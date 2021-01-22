import os, inspect
import pkg_resources
import pybullet
import numpy as np
import math

currentdir = pkg_resources.resource_filename("myGym", "envs")
repodir = pkg_resources.resource_filename("myGym", "")


class HackRobot:
    """
    Robot class for control of robot environment interaction


    """
    def __init__(self,
                 position=[-0.1, 0, 0.07], orientation=[0, 0, 0],
                 dimension_velocity = 0.5,
                 max_velocity = 10.,
                 max_force = 500.,
                 pybullet_client=None):

        self.p = pybullet_client

        self.robot_path = pkg_resources.resource_filename("myGym", "envs/objects/assembly/urdf/cube_holes.urdf")
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        self.orientation = self.p.getQuaternionFromEuler(self.orientation)

        self.max_velocity = max_velocity
        self.max_force = max_force


        self._load_robot()
        self.reset()

    def _load_robot(self):
        """
        Load SDF or URDF model of specified robot and place it in the environment to specified position and orientation
        """
        if self.robot_path[-3:] == 'sdf':
            objects = self.p.loadSDF(
                pkg_resources.resource_filename("myGym",
                                                self.robot_path))
            self.robot_uid = objects[0]
            self.p.resetBasePositionAndOrientation(self.robot_uid, self.position,
                                              self.orientation)
        else:
            self.robot_uid = self.p.loadURDF(
                pkg_resources.resource_filename("myGym",
                                                self.robot_path),
                self.position, self.orientation, useFixedBase=True, flags=(self.p.URDF_USE_SELF_COLLISION))


    def reset(self, random_robot=False):
        """
        Reset joint motors

        Parameters:
            :param random_robot: (bool) Whether the joint positions after reset should be randomized or equal to initial values.
        """
        pass

    def get_action_dimension(self):
        """
        Get dimension of action data, based on robot control mechanism

        Returns:
            :return dimension: (int) The dimension of action data
        """
        if self.robot_action == "joints":
            return len(self.motor_indices)
        else:
            return 3

    def get_observation_dimension(self):
        """
        Get dimension of robot part of observation data, based on robot task and rewatd type

        Returns:
            :return dimension: (int) The dimension of observation data
        """
        return len(self.get_observation())

    def get_observation(self):
        """
        Get robot part of observation data

        Returns:
            :return observation: (list) Position of end-effector link (center of mass)
        """
        observation = []
        state = self.p.getLinkState(self.robot_uid, self.end_effector_index)
        pos = state[0]
        orn = self.p.getEulerFromQuaternion(state[1])

        observation.extend(list(pos))
        return observation

    def get_position(self):
        """
        Get position of robot's end-effector link

        Returns:
            :return position: (list) Position of end-effector link (center of mass)
        """
        return self.p.getLinkState(self.robot_uid, self.end_effector_index)[0]


    def apply_action(self, action):
        """
        Apply action command to robot in simulated environment

        Parameters:
            :param action: (list) Desired action data
        """
        pass
