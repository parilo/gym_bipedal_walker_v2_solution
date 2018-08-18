#!/usr/bin/env python

import gym
import numpy as np

class BipedalWalker:

    def __init__(self, reward_scale=0.1, frame_skip=1, visualize=False, hardcore=False):
        self.visualize = visualize
        if hardcore:
            self.env = gym.make("BipedalWalkerHardcore-v2")
        else:
            self.env = gym.make("BipedalWalker-v2")
        self.reward_scale = reward_scale
        self.frame_skip = frame_skip
        self.observation_shapes = [(24,)]
        self.action_size = 4

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        self.init_action = np.round(np.random.uniform(-1.0, 1.0, size=self.action_size))
        return self.env.reset()

    def step(self, action):
        for i in range(self.frame_skip):
            observation, r, done, info = self.env.step(action)
            # using shaped reward for training
            reward = r * self.reward_scale

            if self.visualize:
                self.env.render()

            if done: break
        self.total_reward += reward
        self.time_step += 1
        return observation, reward, done, info

    def get_total_reward(self):
        # return not shaped reward for episode
        return self.total_reward / self.reward_scale

    def get_random_action(self, resample=True):
        if self.time_step % 10 == 0:
            self.init_action = np.round(np.random.uniform(-1.0, 1.0, size=self.action_size))
        return self.init_action
