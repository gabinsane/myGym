from myGym.envs.gym_env import GymEnv
from myGym.envs.rewards import HackReward
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


class HackEnv(GymEnv):
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
        self.reward = HackReward(self.env, self.task)

        self.obs_space = obs_space
        self.visualize = visualize
        self.visgym = visgym
        self.logdir = logdir
        self.time_counter = 0
        self.parcels_done = 0
        super(HackEnv, self).__init__(active_cameras=active_cameras, **kwargs)

    def _setup_scene(self):
        """
        Set-up environment scene. Load static objects, apply textures. Load robot.
        """
        self._add_scene_object_uid(self.p.loadURDF(pkg_resources.resource_filename("myGym", "/envs/rooms/hack_plane.urdf"),
                                        np.add([0,0,0], self.global_shift), [0,0,0,1], useFixedBase=True, useMaximalCoordinates=True), "floor")
        self._add_scene_object_uid(self.p.loadURDF(
                pkg_resources.resource_filename("myGym", "/envs/rooms/hack_room.urdf"),
                                            [0,0,0],[0,0,0,1],useFixedBase=True, useMaximalCoordinates=True), "gym")
        self._add_scene_object_uid(self.p.loadURDF(
            pkg_resources.resource_filename("myGym", "envs/objects/assembly/urdf/sphere_holes.urdf"),
                                        [0,0,0],[0,0,0,1],useFixedBase=True, useMaximalCoordinates=True), "cube1")
        self._add_scene_object_uid(self.p.loadURDF(
            pkg_resources.resource_filename("myGym", "envs/objects/assembly/urdf/sphere_holes.urdf"),
                                        [0.75,0.75,0],[0,0,0,1],useFixedBase=True, useMaximalCoordinates=True), "cube2")

    def _set_observation_space(self):
        """
        Set observation space type, dimensions and range
        """
        observationDim = self.task.obsdim
        observation_high = np.array([100] * observationDim)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)

    def _set_action_space(self):
        """
        Set action space dimensions and range
        """
        action_dim = 3
        self.action_low = np.array([-1] * action_dim)
        self.action_high = np.array([1] * action_dim)
        self.action_space = spaces.Box(np.array([-1]*action_dim), np.array([1]*action_dim))

    def reset(self, hard=False):
        """
        Environment reset called at the beginning of an episode. Reset state of objects, robot, task and reward.

        Parameters:
            :param hard: (bool) Whether to do hard reset (resets whole pybullet scene)
        Returns:
            :return self._observation: (list) Observation data of the environment
        """
        super().reset(hard=hard)
        self.robot.reset()
        self.task.reset_task()
        self.reward.reset()
        self.p.stepSimulation()
        self.parcels_done = 0
        self.time_counter = 0
        self._observation = self.get_observation()
        return self._observation

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

    def step(self, actions):
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
        for robot_idx, action in enumerate(actions):
            self._apply_action_robot(action, robot_idx)
        self._observation = self.get_observation()
        reward = self.compute_reward(observation=self._observation)
        self.episode_reward += reward
        #info = {'d': self.task.last_distance / self.task.init_distance,
        #        'p': int(self.parcels_done)}  ## @TODO can we log number of sorted parcels?
        self.time_counter += 0.25
        return self._observation, reward

    def compute_reward(self, observation):
        reward = self.reward.compute(observation)
        if self.reward.goal_reached:
            self.parcels_done += 1
            print("Success! Parcels sorted: {}".format(self.parcels_done))
        return reward

    def _apply_action_robot(self, action, robot_idx):
        """
        Apply desired action to robot in simulation

        Parameters:
            :param action: (list) Action data returned by trained model
        """
        pass
