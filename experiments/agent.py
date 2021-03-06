#!/usr/bin/env python

import sys
sys.path.append("../")

import os
import time
import argparse
import random
import numpy as np

from rl_server.server.rl_client import RLClient
from agent_replay_buffer import AgentBuffer
from misc.defaults import default_parse_fn
from tensorboardX import SummaryWriter
from misc.defaults import create_if_need
from datetime import datetime


random.seed(3)
np.random.seed(3)

############################# parse arguments #############################
parser = argparse.ArgumentParser(
    description="Run RL agent on RL server")
parser.add_argument(
    "--id",
    dest="id",
    type=int,
    default=0)
parser.add_argument(
    "--visualize",
    dest="visualize",
    action="store_true",
    default=False)
parser.add_argument(
    "--validation",
    dest="validation",
    action="store_true",
    default=False)
parser.add_argument(
    "--hparams",
    type=str, required=True)
parser.add_argument(
    "--logdir",
    type=str, required=True)
parser.add_argument(
    "--exploration",
    dest="exploration",
    type=str,
    default="0.1")
args = parser.parse_args()
args, hparams = default_parse_fn(args, [])

from envs.bipedal_walker import BipedalWalker
env = BipedalWalker(
    reward_scale=0.1,
    frame_skip=hparams["env"]["frame_skip"],
    visualize=args.visualize,
    hardcore=False)

observation_shapes = env.observation_shapes
action_size = env.action_size
path_to_rewards_train = f"{args.logdir}/rewards-train.txt"
path_to_rewards_test = f"{args.logdir}/rewards-test.txt"

current_date = datetime.now().strftime('%y-%m-%d-%H-%M-%S-%M-%f')
if args.validation:
    path_to_rewards = path_to_rewards_test
    logpath = f"{args.logdir}/agent-valid-{args.id}-{current_date}"
else:
    path_to_rewards = path_to_rewards_train
    logpath = f"{args.logdir}/agent-train-{args.id}-{current_date}"
create_if_need(logpath)
logger = SummaryWriter(logpath)
history_len = hparams["server"]["history_length"]

################################ run agent ################################
buf_capacity = hparams["env"]["agent_buffer_size"]

rl_client = RLClient(
    port=hparams["server"]["init_port"] + args.id)
agent_buffer = AgentBuffer(
    buf_capacity, observation_shapes, action_size)
agent_buffer.push_init_observation([env.reset()])

start_time = time.time()
n_steps = 0
episode_index = 0

while True:

    # obtain current state from the buffer
    state = agent_buffer.get_current_state(
        history_len=history_len)[0].ravel()

    if args.validation:
        action = rl_client.act([state])
    else:
        action = rl_client.act([state]) + np.random.normal(
            scale=np.float(args.exploration), size=action_size)
            
    # clip action to be in range [-1, 1]
    action = np.clip(action, -1., 1.)

    next_obs, reward, done, info = env.step(action)
    transition = [[next_obs], action, reward, done]
    agent_buffer.push_transition(transition)
    next_state = agent_buffer.get_current_state(
        history_len=history_len)[0].ravel()
    n_steps += 1

    if done:
        elapsed_time = time.time() - start_time
        episode = agent_buffer.get_complete_episode()
        rl_client.store_episode(episode)
        print("--- episode ended {} {} {}".format(
            episode_index, env.time_step, env.get_total_reward()))

        logger.add_scalar("steps", n_steps, episode_index)
        logger.add_scalar(
            "reward", env.get_total_reward(), episode_index)
        logger.add_scalar(
            "episode per minute",
            episode_index / elapsed_time * 60,
            episode_index)
        logger.add_scalar(
            "steps per second",
            n_steps / elapsed_time,
            episode_index)

        with open(path_to_rewards, "a") as f:
            f.write(
                str(args.id) + " " +
                str(episode_index) + " " +
                str(env.get_total_reward()) + "\n")

        episode_index += 1
        agent_buffer = AgentBuffer(
            buf_capacity, observation_shapes, action_size)
        agent_buffer.push_init_observation([env.reset()])
        n_steps = 0
        start_time = time.time()
