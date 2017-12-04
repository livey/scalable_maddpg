import numpy as np
import time
import sys
sys.path.insert(1,'env/')
from env import envs
from maddpg import MaDDPG

# load the pre-trained network and plays the video slowly
state_dim = 5
action_dim = 1
max_edge=1

num_agents = 3
maddpg = MaDDPG(num_agents,state_dim, action_dim)

# load saved network
maddpg.load_network()

Env = envs.Environ(num_agents,max_edge)
obs = Env.reset()
current_state = obs
max_time = 10000
#print(current_state)
for epoch in range(max_time):
    print('epoch',epoch)
    action = maddpg.action(current_state)
    #print(action)
    next_state, reward, done = Env.step(action)
    #print(reward)
    #maddpg.perceive(current_state,action,reward,next_state,done)
    current_state = next_state
    if done:
        print('done!!!!, reward : {}'.format(reward))
        #Env.re_create_env(num_agents)
        current_state = Env.reset()

    Env.render()
    time.sleep(.2)

maddpg.close_session()

