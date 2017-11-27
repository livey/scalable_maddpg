import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math

from multiagent.environment import MultiAgentEnv
from multiagent.new_policy import InteractivePolicy
import multiagent.scenarios as scenarios


class Environment_output():
    def __init__(self, env):
        self.env = env

    def action_transfer(self,action):
        u=np.zeros(5)
        movable=action[0]
        angle=action[1]
        if self.env.discrete_action_input:
            if movable<=0:
                u=0
            else:
                direction_x = math.cos(angle * math.pi)
                direction_y = math.sin(angle * math.pi)
                if abs(direction_x)>=abs(direction_y ):
                    if direction_x>0:
                        u=1
                    if direction_x<0:
                        u=2
                else:
                    if direction_y>0:
                        u=3
                    if direction_y<0:
                        u=4
        else:
            if movable<=0:
                u[0] +=1
            if movable>0:
                direction_x=math.cos(angle*math.pi)
                direction_y=math.sin(angle*math.pi)
                if direction_x>0:
                    u[1] +=1
                if direction_x<0:
                    u[2] +=1
                if direction_y>0:
                    u[3] +=1
                if direction_y<0:
                    u[4] +=1
        return u


    def environment_output(self,obs):
        policies = [InteractivePolicy(self.env, i) for i in range(self.env.n)]
        act_n =[]
        observation=obs
        length = observation.shape[1]
        for i, policy in enumerate(policies):
            if i < length - 1:
                obs = observation[i, :]
                act_n.append(policy.action(obs))
            if i == length - 1:
                act_n.append(policy.action(observation))
        obs_n, reward_n, done_n, _ = self.env.step(act_n)
        env_obs = obs_n[0]
        env_reward = []
        for agent in self.env.world.agents:
            env_reward.append(self.env._get_reward(agent))
        return env_obs,env_reward,done_n





