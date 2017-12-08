import tensorflow as tf
import numpy as np

PRE_LSTM_LAYER_SIZE = 10
CELL_UNITS = 5
LAYER1_SIZE =10
TAU = 0.001
BATCH_SIZE = 64

class PreyActorNetwork:
    def __init__(self, sess, prey_state_dim, prey_action_dim, agent_state_dim):
        self.sess = sess
        self.prey_state_dim = prey_action_dim
        self.prey_action_dim = prey_action_dim
        self.agent_state_dim = agent_state_dim
        self.pre_state_inputs,\
            self.agents_state_inputs,\
            self.nets = self.create_network(prey_state_dim, prey_action_dim, agent_state_dim)

    def create_network(self, prey_state_dim, prey_action_dim, agent_state_dim):
        with tf.variable_scope('prey_actor') as scope:
            prey_state_input = tf.placeholder('float', [None, prey_state_dim])
            agents_state_inputs = tf.placeholder('float',[None, None, agent_state_dim])
            batch_size = tf.shape(agents_state_inputs)[0]
            num_agents = tf.shape(agents_state_inputs)[1]
            pre_W = tf.get_variable('pre_W',[PRE_LSTM_LAYER_SIZE, CELL_UNITS],
                                    initializer=tf.contrib.layers.xavier_initializer())
            pre_b = tf.get_variable('pre_b',[CELL_UNITS],
                                    initializer=tf.contrib.layers.xavier_initializer())
            agents_states = tf.reshape(agents_state_inputs,[-1,agent_state_dim])
            pre_layer_output = tf.nn.relu(tf.matmul(agents_state_inputs,pre_W)+pre_b)
            lstm_input = tf.reshape(pre_layer_output,[-1,num_agents,CELL_UNITS])

            with tf.variable_scope('forward_lstm'):
                lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_UNITS)
            with tf.variable_scope('backward_lstm'):
                lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_UNITS)

            (outputs, output_state) = tf.nn.bidirectional_dynamic_rnn(
                lstm_forward_cell,
                lstm_backward_cell,
                lstm_input,
                dtype='float',
                # initial_state_fw=initial_lstm_state_forward_input,
                # initial_state_bw=initial_lstm_state_backward_input,
                # sequence_length=step_size,
                time_major=False,
                scope=scope)

            suf_W00 = tf.get_variable('suf_W00',[CELL_UNITS, LAYER1_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
            suf_W01 = tf.get_variable('suf_W01',[CELL_UNITS, LAYER1_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
            suf_W10 = tf.get_variable('suf_W10', [CELL_UNITS, LAYER1_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
            suf_W11 = tf.get_variable('suf_W11',[CELL_UNITS, LAYER1_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
            suf_b = tf.get_variable('suf_b', [LAYER1_SIZE],
                                    initializer=tf.contrib.layers.xavier_initializer())
            state_W = tf.get_variable('state_W',[prey_state_dim, LAYER1_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())

            layer_ouput = tf.nn.relu(tf.matmul(outputs[0][0],suf_W00)+
                                     tf.matmul(outputs[0][-1],suf_W01)+
                                     tf.matmul(outputs[1][0],suf_W10)+
                                     tf.matmul(outputs[1][-1],suf_W11)+
                                     tf.matmul(prey_state_input, state_W)+
                                     suf_b)
            layer2_W = tf.get_variable('layer2_W',[LAYER1_SIZE,prey_action_dim],
                                       initializer=tf.contrib.layers.xavier_initializer())
            layer2_b = tf.get_variable('layer2_b',[prey_action_dim],
                                       initializer=tf.contrib.layers.xavier_initializer())
            action_output = tf.tanh(tf.matmul(layer_ouput,layer2_W)+layer2_b)

            nets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prey_actor')














