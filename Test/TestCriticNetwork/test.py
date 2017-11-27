import tensorflow as tf
import numpy as np
from criticnetwork import CriticNetwork

actionDimension = 2
stateDimension = 3
numOfAgents = 4
batchSize = 2
stateInputs = np.random.rand(batchSize,numOfAgents,stateDimension)
actionInputs = np.random.rand(batchSize,numOfAgents,actionDimension)
Rt = np.random.rand(batchSize,numOfAgents)
#stateInputs = np.ones((batchSize,numOfAgents,stateDimension))
#actionInputs = np.ones((batchSize,numOfAgents,actionDimension))

sess = tf.InteractiveSession()
Critic = CriticNetwork(sess,stateDimension,actionDimension)
#Critic.printnets()
qvalue = Critic.q_value(stateInputs,actionInputs)
Critic.update_target()
print('qvalue', qvalue)
#print(qvalue.shape)
#ops = Critic.class_test()
#print('stop')

aa = Critic.target_q(stateInputs, actionInputs)
print('target value', aa)

# test train
Critic.train(Rt,stateInputs,actionInputs)
Critic.update_target()
bb = Critic.target_q(stateInputs, actionInputs)
print('before initialize', bb)


#gg = Critic.gradients(stateInputs, actionInputs)
#print(gg.shape)


# test the behaviour of the global_varialbes_initilaizer
# whether initialize existing initialized variables
sess.run(tf.global_variables_initializer())
bb = Critic.q_value(stateInputs, actionInputs)
print('after second initialize q value ', bb)

sess.close()