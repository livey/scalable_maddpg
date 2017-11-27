import tensorflow as tf
import numpy as np
from actor_network import ActorNetwork
state_dim = 3
action_dim = 2
batch_size = 5
agent_name = 'agent1'

y_grad = np.random.rand(batch_size,state_dim)
state_batch  = np.random.rand(batch_size, state_dim)
# with tf.Session() as sess:
#     actor = ActorNetwork(sess,state_dim,action_dim,agent_name,1)
#     print(actor.actions(state_batch))
#     actor.update_target()
#     print('\n')
#     print(actor.target_actions(state_batch))
#
#     actor.train(y_grad,state_batch)
#     actor.update_target()
#     print(actor.target_actions(state_batch))

# test create multiple agents
# agents = []
# with tf.Session() as sess:
#     for ii in range(10):
#         agent_name = 'agent'+str(ii)
#         print(agent_name)
#         agents.append(ActorNetwork(sess, state_dim, action_dim, agent_name))
#
#     print(agents)

# test the copy works
with tf.Session() as sess:
    agent1 = ActorNetwork(sess,state_dim,action_dim,'agent1')
    agent1.train(y_grad,state_batch)

    agent2 = ActorNetwork(sess, state_dim, action_dim, 'agent2', agent1.nets)
    print('agent 1', agent1.actions(state_batch))
    print('agent 2', agent2.actions(state_batch))
