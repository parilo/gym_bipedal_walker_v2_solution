#!/usr/bin/env python

import sys
sys.path.append("../")

import argparse
import random

import numpy as np

from rl_server.tensorflow.rl_server import RLServer
from rl_server.tensorflow.algo.categorical_ddpg import CategoricalDDPG
from rl_server.tensorflow.networks.actor_networks import *
from rl_server.tensorflow.networks.critic_networks_new import CriticNetwork
from misc.defaults import default_parse_fn, create_if_need, set_global_seeds

set_global_seeds(44)

############################# parse arguments #############################
parser = argparse.ArgumentParser(
    description="Run RL algorithm on RL server")
parser.add_argument(
    "--hparams",
    type=str, 
    required=True)
parser.add_argument(
    "--logdir",
    type=str, 
    required=True)
args, unknown_args = parser.parse_known_args()
create_if_need(args.logdir)
args, hparams = default_parse_fn(args, unknown_args)
observation_shapes = [(hparams["env"]["obs_size"],)]
history_len = hparams["server"]["history_length"]
state_shapes = [(history_len, hparams["env"]["obs_size"],)]
action_size = hparams["env"]["action_size"]

############################# define algorithm ############################
actor = ActorNetwork(
    state_shape=state_shapes[0],
    action_size=action_size,
    **hparams["actor"],
    scope="actor")

critic = CriticNetwork(
    state_shape=state_shapes[0],
    action_size=action_size,
    **hparams["critic"],
    num_atoms=101,
    v=(-15., 30.),
    scope="critic")

agent_algorithm = CategoricalDDPG(
    state_shapes=state_shapes,
    action_size=action_size,
    actor=actor,
    critic=critic,
    actor_optimizer=tf.train.AdamOptimizer(
        learning_rate=hparams["actor_optim"]["lr"]),
    critic_optimizer=tf.train.AdamOptimizer(
        learning_rate=hparams["critic_optim"]["lr"]),
    **hparams["algorithm"])

############################## run rl server ##############################
rl_server = RLServer(
    action_size=action_size,
    observation_shapes=observation_shapes,
    state_shapes=state_shapes,
    agent_algorithm=agent_algorithm,
    ckpt_path=args.logdir,
    **hparams["server"])
rl_server.start()
