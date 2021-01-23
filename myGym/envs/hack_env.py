from myGym.envs.base_env import CameraEnv
from myGym.envs.rewards import HackReward
from myGym.envs.hack_robot import HackRobot
from myGym.envs.hack_task import TaskModule
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
                 active_cameras=None,
                 obs_space=None,
                 visgym=1,
                 logdir=None,
                 num_robots=5,
                 render_on=0,
                 visualize=0,
                 robot_action=None,
                 robot_init_joint_poses=None,
                 max_steps=1000,
                 gui_on=0
                 ):

        self.num_robots = num_robots
        self.robots = []
        self.holes = [[10,10],[15,15]]  # list of coordinates of all holes
        self.humans = [[30,0],[20,0]] # list of coordinates of all humans
        self.task = TaskModule(logdir=logdir, env=self)
        self.reward = HackReward(self, self.task, num_robots=num_robots)
        self.obs_space = obs_space
        self.visualize = visualize
        self.visgym = visgym
        self.logdir = logdir
        self.global_shift = [30,30,0]
        self.time_counter = 0
        self.parcels_done = 0
        self.episode_steps = 0
        self.robots_states = [0] * self.num_robots  # 0 for unloaded, 1 for loaded
        self.robots_waits = [0] * self.num_robots  # num steps to wait (loading, unloading)
        self.timestep = 0.125 #sec


        super(HackEnv, self).__init__(active_cameras=active_cameras, render_on=render_on, gui_on=gui_on)

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

        for robot_id in range(self.num_robots):

            self.robots.append(HackRobot(position = [0+robot_id*0.75, 0, 0], pybullet_client=self.p))

    def _set_observation_space(self):
        """
        Set observation space type, dimensions and range
        """
        observationDim = self.task.obsdim
        observation_high = np.full(observationDim, 100)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)

    def _set_action_space(self):
        """
        Set action space dimensions and range
        """
        action_dim = 2
        self.action_low = np.tile([-np.pi, -1], self.num_robots)
        self.action_high = np.tile([np.pi, 1], self.num_robots)
        self.action_space = spaces.Box(self.action_low, self.action_high)




    def reset(self, hard=False):
        """
        Environment reset called at the beginning of an episode. Reset state of objects, robot, task and reward.

        Parameters:
            :param hard: (bool) Whether to do hard reset (resets whole pybullet scene)
        Returns:
            :return self._observation: (list) Observation data of the environment
        """
        super().reset(hard=hard)
        #self.robot.reset()
        #self.task.reset_task()
        self.reward.reset()
        self.p.stepSimulation()
        self.parcels_done = 0
        self.time_counter = 0
        self.episode_steps = 0
        self._observation = self.get_observation()
        return self._observation

    def _set_cameras(self):
        """
        Add cameras to the environment
        """
        camera_args = {'position': [[-0.0, 2.1, 1.0], [0.0, -1.7, 1.2], [3.5, -0.6, 1.0], [-3.5, -0.7, 1.0], [-0.0, 2.0, 4.9]],
                       'target': [[0.0, 0.0, 0.7], [-0.0, 1.3, 0.2], [3.05, -0.2, 0.9], [-2.9, -0.2, 0.9], [-0.0, 2.1, 3.6]]}
        for cam_idx in range(len(camera_args['position'])):
            self.add_camera(position=camera_args['position'][cam_idx], target_position=camera_args['target'][cam_idx], distance=0.001, is_absolute_position=True)

    def get_observation(self):
        """
        Get observation data from the environment

        Returns:
            :return observation: (array) Represented position of task relevant objects
        """
        #return np.zeros((self.num_robots, 6), np.float) #DEBUG
        return self.task.get_observation()

    def check_collision(self):
        obs_after_actions = self.get_observation()
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if i == j:
                    continue   
                vector1 = obs_after_actions[i][0:2] - obs_after_actions[j][0:2]
                ss = np.linalg.norm(vector1) #dist between two bots
                if ss < 0.5: # bots in collision
                    print('collision')

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
        actions = actions.reshape(-1,2)
        for robot_idx, action in enumerate(actions):
            if self.robots_waits[robot_idx] > 0: #check if bot is loading/unloading
                self.robots_waits[robot_idx] -= self.timestep #if waiting, sub step time
            else:
                self._apply_action_robot(action, robot_idx) #if not waiting, apply action
        self.check_collision()
        self._observation = self.get_observation()
        reward = self.compute_reward(observation=self._observation)
        self.episode_reward += np.mean(reward)  # not sure where this is used
        #info = {'d': self.task.last_distance / self.task.init_distance,
        #        'p': int(self.parcels_done)}  ## @TODO can we log number of sorted parcels?
        self.time_counter += self.timestep
        self.episode_steps += 1
        return self._observation, reward, None, None

    def compute_reward(self, observation):
        reward = self.reward.compute(observation)
        for ix, goal in enumerate(self.reward.goals_reached):
            if goal == 1 and self.robots_states[ix] == 1:
                self.parcels_done += 1
                print("Robot {} succeeded! Overall parcels sorted: {}".format(ix, self.parcels_done))
                self.reward.goals_reached[ix] = 0
            elif goal == 1 and self.robots_states[ix] == 0:
                print("Robot {} picked up parcel".format(ix))
                self.reward.goals_reached[ix] = 0
        return reward

    def _apply_action_robot(self, action, robot_idx):
        """
        Apply desired action to robot in simulation

        Parameters:
            :param action: (list) Action data returned by trained model
        """
        self.robots[robot_idx].apply_action(np.append(action, 2))
