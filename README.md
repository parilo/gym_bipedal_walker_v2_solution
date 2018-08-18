OpenAI Gym Bipedal Walker v2 solution using
A Distributional Perspective on Reinforcement Learning (https://arxiv.org/abs/1707.06887)

```
git clone repo
cd experiments
python run_server_tf.py --hparams config_bipedal_walker.yml --logdir logs/bipedal_walker
# see errors and solve dependences
# run it again
# here you have output of RL server app

# get new therminal
cd experiments
python run_many_agents.py --hparams config_bipedal_walker.yml --logdir logs/asd
# again solve dependences
# here you have output of 8 agents
```

It runs 8 agents in parallel. 2 of them will be visible. One with exploration (added normal noise of 0.1) another for validation (without exploration noise).
Near to 200K server train ops you probably get working bipedal walker
