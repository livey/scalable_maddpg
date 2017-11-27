#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.new_policy import InteractivePolicy
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='ev.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario(3)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        observation=obs_n[0]
        length=observation.shape[1]
        for i,policy in enumerate(policies):
            if i<length-1:
                obs=observation[i,:]
                act_n.append(policy.action(obs))
            if i==length-1:
                act_n.append(policy.action(observation))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)

        # render all agent views
        env.render()

        #print action
        print(act_n)
        #get observation/state
        env_obs = obs_n[0]
        print(env_obs)
        #get reward
        env_reward=[]
        for agent in env.world.agents:
            env_reward.append(env._get_reward(agent))
        print(env_reward)