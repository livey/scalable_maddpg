import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'env/')
from env import envs
from maddpg import MaDDPG
from maddpg1 import MADDPG1

state_dim = 5
action_dim = 1
max_edge= 1
num_agents = 3
maddpg = MaDDPG(num_agents,state_dim, action_dim)


preyAct_dim = 1
preyState_dim = (num_agents + 1) * 2 + num_agents
prey_network = MADDPG1(1,preyState_dim ,preyAct_dim)



def learn_action(self, obs):
    action = maddpg.noise_action(current_state)


Env = envs.Environ(num_agents,max_edge)

max_episode = 1000000
done_epoch = 0
#print(current_state)
max_epoch = 1000

catch_time = []

for episode in range(max_episode):
    print('episode',episode)
    #while (True):
        #Env.re_create_env(num_agents)
    agent_current_state, prey_current_state = Env.reset()
        #action = maddpg.noise_action(current_state)
        #next_state, reward, done = Env.step(action)
        #print(reward)
       # if not done:
       #    current_state = next_state
       #      break

    for epoch in range(max_epoch):
        #print('epoch',epoch)
        #Env.render()
        #action calculate
        agent_action = maddpg.noise_action(agent_current_state)
        prey_action = prey_network.noise_action(prey_current_state)

        #implement actions
        agent_next_state,agent_reward,prey_next_state,prey_reward,done = Env.step(agent_action,prey_action)

        #update network
        maddpg.perceive(agent_current_state,agent_action,agent_reward,agent_next_state,done)
        prey_network.perceive(prey_current_state,prey_action,prey_reward,prey_next_state,done)

        agent_current_state=agent_next_state
        prey_current_state=prey_next_state

        if done:
            print('Done!!!!!!!!!!!! at epoch{} , reward:{}'.format(epoch,reward))
            # add summary for each episode
            maddpg.summary(episode)
            break
    if epoch ==max_epoch-1:
        print('Time up >>>>>>>>>>>>>>')

