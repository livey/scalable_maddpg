import tensorflow as tf
import numpy as np
import math

#
# def variable(shape, f):
#     return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
#
# ## test the slicing of the input
# actionDimension = 2
# outputDimension = 2
# inputs = tf.placeholder(tf.float32, [None, actionDimension])
# outputs = tf.placeholder(tf.float32, [None, 2])
# W1 = tf.get_variable('W1', shape=[2,2],
#                      initializer=tf.contrib.layers.xavier_initializer())
#
# W = variable([1,actionDimension],actionDimension)
#y = tf.matmul(inputs, W)
#
''' Tensorflow slicing and index 
https://stackoverflow.com/questions/34002591/tensorflow-slicing-based-on-variable'''

'''tf.split 

   tf.expand_dims
'''

# y = tf.matmul(tf.expand_dims(inputs[:, 0], 1), tf.eye(1))
# '''use concatenate instead of stack'''
# for ii in range(1,actionDimension):
#     y = tf.concat([y, tf.matmul(tf.expand_dims(inputs[:, ii], 1), tf.eye(1))], 1)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# feedin = np.random.rand(10,2)
# #print(sess.run(tf.expand_dims(inputs[:, 1]), feed_dict={inputs:feedin}))
# print(sess.run(y, feed_dict={inputs: feedin}))
# g = tf.gradients(y,inputs)
# print(sess.run(g,feed_dict={inputs:feedin}))
# print('hello')
#y = tf.matmul(inputs, W)+b

# for ii in range(outputDimension):
#     y[ii] = tf.matmul(inputs[:, outputDimension-ii], W)+b

# loss = tf.reduce_mean(tf.square(outputs-y))



def preLstmNetwork(actions,states,actionDimension,stateDimension):
    layer1_size = 10
    layer2_size = 20

    ''' xavier initializer 
    W = tf.get_variable("W", shape=[784, 256],
               initializer=tf.contrib.layers.xavier_initializer())
               https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    '''
    with tf.variable_scope('PreLstm'):
        W1 = tf.get_variable('W1', shape=[stateDimension,layer1_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 =tf.get_variable('b1', shape =[layer1_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable('W2', shape=[layer1_size,layer2_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape=[layer2_size],
                             initializer=tf.contrib.layers.xavier_initializer())
        W2_action = tf.get_variable('W2_action', shape=[actionDimension,layer2_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable('W3',shape=[layer2_size,1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3',shape=[1],
                             initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(states, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(actions, W2_action) + b2)
        preLstmOutputs = tf.identity(tf.matmul(layer2, W3) + b3)
        return preLstmOutputs

'''
collect trainable variables in name scope
https://www.tensorflow.org/api_docs/python/tf/get_collection
https://stackoverflow.com/questions/36533723/tensorflow-get-all-variables-in-scope
'''
def creat_q_network(actionDimension,stateDimension,numOfAgents):
    '''
    share the Pre-LSTM neuron networks
    https://www.tensorflow.org/versions/r0.12/how_tos/variable_scope/'''
    state_input = tf.placeholder('float', [None, stateDimension*numOfAgents])
    action_input = tf.placeholder('float', [None, actionDimension*numOfAgents])
    with tf.variable_scope('q_network') as scope:
        preOutPut = preLstmNetwork(action_input[:,0:actionDimension],state_input[:,0:stateDimension],actionDimension,stateDimension)
        scope.reuse_variables()
        for ii in range(0,numOfAgents-1):
            preOutPut = tf.concat([preOutPut,preLstmNetwork(
                action_input[:, ii*actionDimension:(ii+1)*actionDimension],
                state_input[:, ii*stateDimension:(ii+1)*stateDimension],
                actionDimension, stateDimension)],1)

        '''Tutorial on LSTM http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        How to use http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
        Tutorial : https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714
        How LSTM was trained 
        https://stackoverflow.com/questions/38441589/is-rnn-initial-state-reset-for-subsequent-mini-batches'''

        cell_f = tf.nn.rnn_cell.LSTMCell(num_units=numOfAgents)
        cell_b = tf.nn.rnn_cell.LSTMCell(num_units=numOfAgents)
        lstmOutputs = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_f,
                                                          cell_bw=cell_b,
                                                          dtype = tf.float32,
                                                          #sequence_length=[10,numOfAgents],
                                                          parallel_iterations=128, # this parameter is crutial, since we do not need temporal dependence
                                                          inputs=preOutPut,
                                                          scope=scope)
    return state_input, action_input, lstmOutputs








# state_input, action_input, preOutPut = creat_q_network(2,2,3)
# n = 10
# actionDimension = 2
# stateDimension = 2
# actions = tf.placeholder(tf.float32, [None, actionDimension])
# states  = tf.placeholder(tf.float32, [None, stateDimension])
#
# output1= preLstmNetwork(actions,states,actionDimension,stateDimension)
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# ot1 = sess.run(output1,feed_dict={actions:np.random.rand(n,actionDimension),states:
#                                  np.random.rand(n,stateDimension)})
#
# inputs = np.ones((n,2*3))
# ot2 = sess.run(preOutPut,feed_dict={state_input:inputs,action_input:inputs})
# print(ot2)

''' test whether is possible to use changable agents variable'''
#agents = tf.placeholder(tf.int8, [1])
actionDimension =3
batchSize =2
inputs = tf.placeholder(tf.float32, [None, None, actionDimension])

#outputs = tf.reshape(inputs,shape=[-1,tf.shape(inputs)[1]])
inputsW = tf.reshape(inputs, [-1, actionDimension])
W = tf.Variable(tf.random_uniform([actionDimension,1],
                                  -1 / math.sqrt(actionDimension),
                                  1 / math.sqrt(actionDimension)))

outputs1 = tf.matmul(inputsW, W)
outputs = tf.reshape(outputs1,[-1,tf.shape(inputs)[1]])
gd = tf.gradients(outputs,inputs)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

b = sess.run(outputs, feed_dict={ inputs:np.random.rand(batchSize,5,actionDimension)})
print(b)

print(sess.run(tf.shape(b)))
gd0 = sess.run(gd,feed_dict={inputs:np.random.rand(batchSize,5,actionDimension)})
print(np.shape(gd0))
#sizes = sess.run(tf.shape(gd0))
#print(sizes)