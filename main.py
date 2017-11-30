import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'env/')
from env import envs
from maddpg import MaDDPG

state_dim = 4
action_dim = 1

num_agents = 4
maddpg = MaDDPG(num_agents,state_dim, action_dim)

Env = envs.Environ(num_agents)
obs = Env.reset()
current_state = obs
max_time = 100001
done_epoch =0
#print(current_state)
catch_time = []
for epoch in range(max_time):
    #print('epoch',epoch)
    action = maddpg.noise_action(current_state)
    #print(action)
    next_state, reward, done = Env.step(action)
    #print(reward)
    maddpg.perceive(current_state,action,reward,next_state,done)
    current_state = next_state
    if done:
        print('done at epoch: {}'.format(epoch))
        inter = epoch - done_epoch
        catch_time.append(inter)
        print('episode time',  inter)
        done_epoch = epoch
        Env.re_create_env(num_agents)
        current_state = Env.reset()

    #if epoch % 1000==1 or epoch% 1000 == 1 or epoch%1000==2 or epoch%1000==3:
    #Env.render()


plt.plot(catch_time)
plt.show()