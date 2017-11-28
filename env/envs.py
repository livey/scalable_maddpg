import os, sys
import math
import numpy as np
from pray_policy import Pray
sys.path.insert(1,os.path.join(sys.path[0],'..'))

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


class Environ:
    def __init__(self,num_agents):
        self.num_agents = num_agents

        self.env, \
        self.current_obs \
            = self.create_env(num_agents)

        self.pray = Pray()

    def re_create_env(self, num_agents):
        self.num_agents = num_agents
        self.env, \
        self.current_obs \
            = self.create_env(num_agents)

    def create_env(self,num_agents):
        scenario = scenarios.load('ev.py')
        world = scenarios.make_world()
        env = MultiAgentEnv(world)
        obs_n = env.reset()
        return env, obs_n[0]

    def render(self):
        self.env.render()

    def step(self,act_n):
        agent_actions = self.action_transfer(act_n)
        pray_action = self.pray.action(self.current_obs)
        actions = np.vstack((agent_actions,pray_action))
        obs_n, reward_n, done_n, _ = self.env.step(actions)
        agents_obs = obs_n[0][:self.num_agents, :]
        self.current_obs = obs_n[0]
        return agents_obs, np.squeeze(reward_n[:self.num_agents]), done_n[:self.num_agents]

    def reset(self):
        self.env.reset()

    def make_new_env(self,num_agents):
        self.num_agents = num_agents
        self.env = self.create_env(num_agents)


    def action_transfer(self, action):
        for i in range(self.num_agents):
            u = np.zeros((self.num_agents, 5))
            movable = action[i, 0]
            angle = action[i, 1]

            if movable <= 0:
                u[i,0] += 1
            if movable > 0:
                direction_x = math.cos(angle * math.pi)
                direction_y = math.sin(angle * math.pi)
                if direction_x > 0:
                    u[i, 1] += 1
                if direction_x < 0:
                    u[i, 2] += 1
                if direction_y > 0:
                    u[i, 3] += 1
                if direction_y < 0:
                    u[i, 4] += 1
        return u