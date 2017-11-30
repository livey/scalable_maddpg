import os, sys
import math
import numpy as np

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from pray_policy import Pray

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


max_edge=0.2

class Environ:
    def __init__(self,num_agents):
        self.num_agents = num_agents

        self.env, \
        self.current_obs \
            = self.create_env(num_agents)

        self.pray = Pray()

    def re_create_env(self, num_agents=4):
        self.num_agents = num_agents

        self.env, \
        self.current_obs \
            = self.create_env(num_agents)

    def create_env(self, num_agents=3):
        scenario = scenarios.load('ev.py').Scenario(num_agents)
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                            shared_viewer=False)
        obs_n = env.reset()
        return env, obs_n[0]

    def render(self):
        self.env.render()

    def step(self,act_n):
        agent_actions = self.action_transfer(act_n,self.current_obs)
        pray_action = self.pray.action(self.current_obs)
        actions = np.vstack((agent_actions,pray_action))

        #print(actions)
        obs_n, reward_n, done_n, _ = self.env.step(actions)
        agents_obs = obs_n[0][:self.num_agents, :]
        self.current_obs = obs_n[0]
        done = False
        for ii in range(self.num_agents):
            x_dis=self.current_obs[ii,0]-self.current_obs[ii,2]
            y_dis=self.current_obs[ii,1]-self.current_obs[ii,3]
            dis=np.sqrt(np.square(x_dis)+np.square(y_dis))
            if dis<=0.075+0.05:
                done= True
        return agents_obs, np.squeeze(reward_n[:self.num_agents]), done

    def reset(self):
        return self.env.reset()[0]


    def action_transfer(self, action, current_obs):

        u = np.zeros((self.num_agents, 5))
        for i in range(self.num_agents):
            movable = 1
            #angle = action[i, 1]
            angle=action[i]
            if movable <= -1:
                u[i,0] += 1
            if movable > -1:
                direction_x = math.cos(angle * math.pi)
                direction_y = math.sin(angle * math.pi)
                x=current_obs[i,0]
                y = current_obs[i, 1]

                if direction_x > 0:
                    u[i, 1] += direction_x
                if direction_x < 0:
                    u[i, 2] += -direction_x
                if direction_y > 0:
                    u[i, 3] += direction_y
                if direction_y < 0:
                    u[i, 4] += -direction_y
                if u[i,1] > 0 and x >= max_edge:
                    u[i,1] = 0
                if u[i,2] > 0 and x <= -max_edge:
                    u[i,2] = 0
                if u[i,3] > 0 and y >= max_edge:
                    u[i,3] = 0
                if u[i,4] > 0 and y <= -max_edge:
                    u[i,4] = 0

        return u