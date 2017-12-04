import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'env/')
from env import envs
from maddpg import MaDDPG

state_dim = 5
action_dim = 1
max_edge= 1

num_agents = 3
maddpg = MaDDPG(num_agents,state_dim, action_dim)

Env = envs.Environ(num_agents,max_edge)
obs = Env.reset()
current_state = obs

max_episode = 1000000
done_epoch = 0
#print(current_state)
max_epoch = 1000

catch_time = []

for episode in range(max_episode):
    print('episode',episode)
    #while (True):
        #Env.re_create_env(num_agents)
    current_state = Env.reset()
        #action = maddpg.noise_action(current_state)
        #next_state, reward, done = Env.step(action)
        #print(reward)
       # if not done:
       #    current_state = next_state
       #      break

    for epoch in range(max_epoch):
        #print('epoch',epoch)
        #Env.render()
        action = maddpg.noise_action(current_state)
        #print(action)
        next_state, reward, done = Env.step(action)
        maddpg.perceive(current_state,action,reward,next_state,done)
        current_state = next_state
        if done:
            print('Done!!!!!!!!!!!! at epoch{} , reward:{}'.format(epoch,reward))
            # add summary for each episode
            maddpg.summary()
            break
    if epoch ==max_epoch-1:
        print('Time up >>>>>>>>>>>>>>')

