#!/usr/bin/env python

import sys

sys.path.append("../")

import subprocess
import atexit
import time
import argparse

parser = argparse.ArgumentParser(
    description="Run several RL agents on RL server")
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

ps = []

agent_id = 0
for i in range(1):
    ps.append(subprocess.Popen([
        "python", "agent.py", "--visualize", "--id", str(agent_id),
        "--hparams", args.hparams, "--logdir", args.logdir, 
        "--exploration", args.exploration]))
    agent_id += 1

for i in range(1):
    ps.append(subprocess.Popen([
        "python", "agent.py", "--visualize", "--validation",
        "--id", str(agent_id),
        "--hparams", args.hparams, "--logdir", args.logdir, 
        "--exploration", args.exploration]))
    agent_id += 1

for i in range(0):
    ps.append(subprocess.Popen([
        "python", "agent.py", "--validation", "--id", str(agent_id),
        "--hparams", args.hparams, "--logdir", args.logdir, 
        "--exploration", args.exploration]))
    agent_id += 1

for i in range(6):
    ps.append(subprocess.Popen([
        "python", "agent.py", "--id", str(agent_id),
        "--hparams", args.hparams, "--logdir", args.logdir, 
        "--exploration", args.exploration]))
    agent_id += 1


def on_exit():
    for p in ps:
        p.kill()


atexit.register(on_exit)

while True:
    time.sleep(60)
