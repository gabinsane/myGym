from myGym.envs.base_env import CameraEnv
import pybullet
import time
import numpy as np
import math
from gym import spaces
import os
import inspect
import random
import pkg_resources
currentdir = pkg_resources.resource_filename("myGym", "envs")
repodir = pkg_resources.resource_filename("myGym", "")
print("current_dir=" + currentdir)


class HackEnv(CameraEnv):
    def __init__(self,
                 action_repeat=1,
                 active_cameras=None,
                 obs_space=None,
                 visualize=0,
                 visgym=1,
                 logdir=None,
                 **kwargs
                 ):

        self.task = None
        self.reward = None

        self.obs_space = obs_space
        self.visualize = visualize
        self.visgym = visgym
        self.logdir = logdir
        super(HackEnv, self).__init__(active_cameras=active_cameras, **kwargs)

    def _setup_scene(self):
        """
        Set-up environment scene. Load static objects, apply textures. Load robot.
        """
        pass

    def _set_observation_space(self):
        """
        Set observation space type, dimensions and range
        """
        pass

    def _set_action_space(self):
        """
        Set action space dimensions and range
        """
        pass

    def reset(self, hard=False):
        """
        Environment reset called at the beginning of an episode. Reset state of objects, robot, task and reward.

        Parameters:
            :param hard: (bool) Whether to do hard reset (resets whole pybullet scene)
        Returns:
            :return self._observation: (list) Observation data of the environment
        """
        super().reset(hard=hard)

        pass

    def _set_cameras(self):
        """
        Add cameras to the environment
        """
        camera_args = self.workspace_dict[self.workspace]['camera']
        for cam_idx in range(len(camera_args['position'])):
            self.add_camera(position=camera_args['position'][cam_idx], target_position=camera_args['target'][cam_idx], distance=0.001, is_absolute_position=True)

    def get_observation(self):
        """
        Get observation data from the environment

        Returns:
            :return observation: (array) Represented position of task relevant objects
        """
        pass

    def step(self, action):
        """
        Environment step in simulation

        Parameters:
            :param actions: (list) Action data to send to robot to perform movement in this step
        Returns:
            :return self._observation: (list) Observation data from the environment after this step
            :return reward: (float) Reward value assigned to this step
            :return done: (bool) Whether this stop is episode's final
            :return info: (dict) Additional information about step
        """
        pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        #@TODO: Reward computation for HER, argument for .compute()
        reward = self.reward.compute(np.append(achieved_goal, desired_goal))
        return reward

    def _apply_action_robot(self, action):
        """
        Apply desired action to robot in simulation

        Parameters:
            :param action: (list) Action data returned by trained model
        """
        pass
