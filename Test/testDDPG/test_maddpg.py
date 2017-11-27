from maddpg import MaDDPG
import numpy as np
state_dim = 3
action_dim =2
num_agents = 2
states = np.ones((4,state_dim))
states_batch = np.ones((2,4,state_dim))
maddpg = MaDDPG(num_agents,state_dim,action_dim)
maddpg.add_agents(2)
print(maddpg.action(states))
actions = maddpg.target_actions(states_batch)
noise_action = maddpg.noise_action(states)
print(noise_action)
maddpg.close_session()
#print(maddpg.num_agents)