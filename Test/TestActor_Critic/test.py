import tensorflow as tf
import  numpy as np
from criticnetwork import CriticNetwork
from actor_network import ActorNetwork
state_dim = 2
action_dim =3
batch_size = 4
GAMMA = .9
num_agents = 5

nets =None
agents=[]
sess = tf.InteractiveSession()
for ii in range(num_agents):
    agent_name = 'agent'+str(ii)
    agents.append(ActorNetwork(sess,state_dim,action_dim,agent_name,nets))
    nets = agents[-1].nets

critic = CriticNetwork(sess,state_dim,action_dim)

# take action
current_states = np.random.rand(1,num_agents,state_dim)
current_action = np.zeros((1,num_agents,action_dim))
for ii in range(num_agents):
    current_action[0,ii,:] = agents[ii].actions(np.reshape(current_states[0,ii,:],[-1,state_dim]))

Rt = np.random.rand(1,num_agents)
next_state = np.random.rand(1,num_agents,state_dim)
next_action = np.zeros((1,num_agents,action_dim))
for ii in range(num_agents):
    next_action[0,ii,:] = agents[ii].target_actions(np.reshape(next_state[0,ii,:],[1,state_dim]))

qt = critic.target_q(next_state,next_action)
Gt = Rt+ GAMMA*qt
critic.train(Gt,current_states,current_action)
gradients = critic.gradients(current_states,current_action)

for ii in range(num_agents):
    agents[ii].train(np.reshape(gradients[0,ii,:],[1,action_dim]),np.reshape(current_states[0,ii,:],[1,state_dim]))

critic.update_target()
for ii in range(num_agents):
    agents[ii].update_target()

### add on agent, repeat the whole process
num_agents = 6
agents.append(ActorNetwork(sess, state_dim, action_dim, 'agent6',agents[-1].nets))

# take action
current_states = np.random.rand(1,num_agents,state_dim)
current_action = np.zeros((1,num_agents,action_dim))
for ii in range(num_agents):
    current_action[0,ii,:] = agents[ii].actions(np.reshape(current_states[0,ii,:],[-1,state_dim]))

Rt = np.random.rand(1,num_agents)
next_state = np.random.rand(1,num_agents,state_dim)
next_action = np.zeros((1,num_agents,action_dim))
for ii in range(num_agents):
    next_action[0,ii,:] = agents[ii].target_actions(np.reshape(next_state[0,ii,:],[1,state_dim]))

qt = critic.target_q(next_state,next_action)
Gt = Rt+ GAMMA*qt
critic.train(Gt,current_states,current_action)
gradients = critic.gradients(current_states,current_action)

for ii in range(num_agents):
    agents[ii].train(np.reshape(gradients[0,ii,:],[1,action_dim]),np.reshape(current_states[0,ii,:],[1,state_dim]))

critic.update_target()
for ii in range(num_agents):
    agents[ii].update_target()


sess.close()