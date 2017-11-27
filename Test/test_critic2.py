
import tensorflow as tf
import numpy as np
import math

from constants import LAYER_ENCODER_SIZE

from constants import LEARNING_RATE
from constants import TAU
from constants import L2
from constants import max_time_step


class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,action_dim,user_num):
        self.time_step = 0
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.user_num = user_num
        self.max_time_step = max_time_step
        self.fc_layer_size = LAYER_ENCODER_SIZE

        # create q network
        self.lstm_layer_input,\
        self.action_input, \
        self.q_value_output,\
        self.net,\
        self.output_state,\
        self.initial_lstm_state_forward,\
        self.initial_lstm_state_backward,\
        self.step_size = self.create_q_network()

        # create target q network (the same structure with q network)
        self.target_lstm_layer_input,\
        self.target_action_input, \
        self.target_q_value_output,\
        self.target_net,\
        self.target_output_state,\
        self.target_initial_lstm_state_forward,\
        self.target_initial_lstm_state_backward,\
        self.target_time_step = self.create_target_q_network()

        self.create_training_method()

        # initialization
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,self.user_num,1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay

        opt = tf.train.AdamOptimizer(LEARNING_RATE)
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in self.net]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        self.parameters_gradients = opt.compute_gradients(self.cost, self.net)

        self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.parameters_gradients)]

        self.optimizer = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(self.parameters_gradients)])


        self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

    def create_q_network(self):

       with tf.variable_scope("critic_net") as scope_pi:
           lstm_layer_input = tf.placeholder("float", [None, self.user_num, self.state_dim])
           action_input = tf.placeholder("float", [None, self.user_num, self.state_dim])

           step_size = tf.placeholder("float", [1])

           initial_lstm_state_forward = tf.placeholder("float", [2, None, self.fc_layer_size])

           initial_lstm_state_forward_list  = tf.unpack(initial_lstm_state_forward, axis=0)

           initial_lstm_state_forward_input = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state_forward_list[0],
                                                                    initial_lstm_state_forward_list[1])

           initial_lstm_state_backward = tf.placeholder("float", [2, None, self.fc_layer_size])

           initial_lstm_state_backward_list = tf.unpack(initial_lstm_state_forward, axis=0)

           initial_lstm_state_backward_input = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state_backward_list[0],
                                                                            initial_lstm_state_backward_list[1])


           input_s = tf.reshape(lstm_layer_input, [-1, self.state_dim])
           input_a = tf.reshape(action_input, [-1, self.action_dim])

           # encoder layer parameters
           W1_s = tf.get_variable("W1_s", [self.state_dim, self.fc_layer_size],
                                  initializer=tf.random_uniform([self.state_dim, self.fc_layer_size],
                                                                -1 / math.sqrt(self.state_dim),
                                                                1 / math.sqrt(self.state_dim)))

           W1_a = tf.get_variable("W1_a", [self.action_dim, self.fc_layer_size],
                                  initializer=tf.random_uniform([self.action_dim, self.fc_layer_size],
                                                                -1 / math.sqrt(self.action_dim),
                                                                1 / math.sqrt(self.action_dim)))

           b1 = tf.get_variable("b1", [self.fc_layer_size],
                                initializer=tf.random_uniform([self.fc_layer_size], -1 / math.sqrt(self.state_dim),
                                                              1 / math.sqrt(self.state_dim)))
           W2_fw = tf.get_variable("W2_fw", [self.fc_layer_size, 1],
                                   initializer=tf.random_uniform([self.fc_layer_size, self.action_dim],
                                                                 -1 / math.sqrt(self.fc_layer_size),
                                                                 1 / math.sqrt(self.fc_layer_size)))
           W2_bw = tf.get_variable("W2_bw", [self.fc_layer_size, 1],
                                   initializer=tf.random_uniform([self.fc_layer_size, self.action_dim],
                                                                 -1 / math.sqrt(self.fc_layer_size),
                                                                 1 / math.sqrt(self.fc_layer_size)))
           b2 = tf.get_variable("b1", [1],
                                initializer=tf.random_uniform([self.action_dim], -1 / math.sqrt(self.fc_layer_size),
                                                              1 / math.sqrt(self.fc_layer_size)))

           h_fc = tf.nn.relu(tf.matmul(input_s, W1_s) + tf.matmul(input_a, W1_a) + b1)
           h_fc1 = tf.reshape(h_fc, [-1, self.user_num, self.fc_layer_size])
           with tf.variable_scope('forward'):
               lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.fc_layer_size, state_is_tuple=False)
           with tf.variable_scope('backward'):
               lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.fc_layer_size, state_is_tuple=False)

           # "outputs" is a tuple (outputs_forward, outputs_backward).
           # We set "time_major=True" and [num_user, batch_size, fc_layer_size]

           (outputs, output_state) = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell,
                                                                     lstm_backward_cell,
                                                                     h_fc1,
                                                                     initial_state_fw=initial_lstm_state_forward_input ,
                                                                     initial_state_bw=initial_lstm_state_backward_input ,
                                                                     sequence_length=step_size,
                                                                     time_major=False,
                                                                     scope=scope_pi)

           output_fw = tf.reshape(outputs[0], [-1, self.fc_layer_size])
           output_bw = tf.reshape(outputs[1], [-1, self.fc_layer_size])
           # output layer
           q_value_output = tf.reshape(tf.tanh(tf.matmul(output_fw, W2_fw) + tf.matmul(output_bw, W2_bw) + b2),
                                       [-1,self.user_num,  1])

           scope_pi.reuse_variables()
           W_lstm = tf.get_variable("BasicLSTMCell/Linear/Matrix")
           b_lstm = tf.get_variable("BasicLSTMCell/Linear/Bias")


           return  lstm_layer_input, action_input, q_value_output, [W1_s,W1_a,b1,W2_fw,W2_bw,b2,W_lstm,b_lstm],output_state, initial_lstm_state_forward,initial_lstm_state_backward,step_size


    def create_target_q_network(self):

        with tf.variable_scope("target_critic_net") as scope_pi:
            lstm_layer_input = tf.placeholder("float", [None, self.user_num, self.state_dim])
            action_input = tf.placeholder("float", [None, self.user_num, self.state_dim])

            step_size = tf.placeholder("float", [1])

            initial_lstm_state_forward = tf.placeholder("float", [2, None, self.fc_layer_size])

            initial_lstm_state_forward_list = tf.unpack(initial_lstm_state_forward, axis=0)

            initial_lstm_state_forward_input = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state_forward_list[0],
                                                                             initial_lstm_state_forward_list[1])

            initial_lstm_state_backward = tf.placeholder("float", [2, None, self.fc_layer_size])

            initial_lstm_state_backward_list = tf.unpack(initial_lstm_state_forward, axis=0)

            initial_lstm_state_backward_input = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state_backward_list[0],
                                                                              initial_lstm_state_backward_list[1])

            input_s = tf.reshape(lstm_layer_input, [-1, self.state_dim])
            input_a = tf.reshape(action_input, [-1, self.action_dim])

            # encoder layer parameters
            W1_s = tf.get_variable("W1_s", [self.state_dim, self.fc_layer_size],
                                   initializer=tf.random_uniform([self.state_dim, self.fc_layer_size],
                                                                 -1 / math.sqrt(self.state_dim),
                                                                 1 / math.sqrt(self.state_dim)))

            W1_a = tf.get_variable("W1_a", [self.action_dim, self.fc_layer_size],
                                   initializer=tf.random_uniform([self.action_dim, self.fc_layer_size],
                                                                 -1 / math.sqrt(self.action_dim),
                                                                 1 / math.sqrt(self.action_dim)))

            b1 = tf.get_variable("b1", [self.fc_layer_size],
                                 initializer=tf.random_uniform([self.fc_layer_size], -1 / math.sqrt(self.state_dim),
                                                               1 / math.sqrt(self.state_dim)))
            W2_fw = tf.get_variable("W2_fw", [self.fc_layer_size, 1],
                                    initializer=tf.random_uniform([self.fc_layer_size, self.action_dim],
                                                                  -1 / math.sqrt(self.fc_layer_size),
                                                                  1 / math.sqrt(self.fc_layer_size)))
            W2_bw = tf.get_variable("W2_bw", [self.fc_layer_size, 1],
                                    initializer=tf.random_uniform([self.fc_layer_size, self.action_dim],
                                                                  -1 / math.sqrt(self.fc_layer_size),
                                                                  1 / math.sqrt(self.fc_layer_size)))
            b2 = tf.get_variable("b1", [1],
                                 initializer=tf.random_uniform([self.action_dim], -1 / math.sqrt(self.fc_layer_size),
                                                               1 / math.sqrt(self.fc_layer_size)))

            h_fc = tf.nn.relu(tf.matmul(input_s, W1_s) + tf.matmul(input_a, W1_a) + b1)
            h_fc1 = tf.reshape(h_fc, [-1, self.user_num, self.fc_layer_size])

            with tf.variable_scope('forward'):
               lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.fc_layer_size, state_is_tuple=False)
            with tf.variable_scope('backward'):
               lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.fc_layer_size, state_is_tuple=False)

            # "outputs" is a tuple (outputs_forward, outputs_backward).
            # We set "time_major=True" and [num_user, batch_size, fc_layer_size]

            (outputs, output_state) = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell,
                                                                      lstm_backward_cell,
                                                                      h_fc1,
                                                                      initial_state_fw=initial_lstm_state_forward_input,
                                                                      initial_state_bw=initial_lstm_state_backward_input,
                                                                      sequence_length=step_size,
                                                                      time_major=False,
                                                                      scope=scope_pi)

            output_fw = tf.reshape(outputs[0], [-1, self.fc_layer_size])
            output_bw = tf.reshape(outputs[1], [-1, self.fc_layer_size])
            # output layer
            q_value_output = tf.reshape(tf.tanh(tf.matmul(output_fw, W2_fw) + tf.matmul(output_bw, W2_bw) + b2),
                                        [-1, self.user_num, 1])

            scope_pi.reuse_variables()
            W_lstm = tf.get_variable("BasicLSTMCell/Linear/Matrix")
            b_lstm = tf.get_variable("BasicLSTMCell/Linear/Bias")

            return  lstm_layer_input, action_input, q_value_output, [W1_s,W1_a,b1,W2_fw,W2_bw,b2,W_lstm,b_lstm],output_state, initial_lstm_state_forward,initial_lstm_state_backward,step_size


    def reset_state(self, time_step):
        self.output_state_fw_feed = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([time_step, self.fc_layer_size]),
                                                            np.zeros([time_step, self.fc_layer_size]))
        self.output_state_bw_feed = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([time_step, self.fc_layer_size]),
                                                             np.zeros([time_step, self.fc_layer_size]))
        self.target_output_state_fw_feed = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([time_step, self.fc_layer_size]),
                                                             np.zeros([time_step, self.fc_layer_size]))
        self.target_output_state_bw_feed = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([time_step, self.fc_layer_size]),
                                                             np.zeros([time_step, self.fc_layer_size]))

    def update_target(self):
        self.sess.run([
            self.target_net[0].assign(TAU * self.net[0] + (1 - TAU) * self.target_net[0]),
            self.target_net[1].assign(TAU * self.net[1] + (1 - TAU) * self.target_net[1]),
            self.target_net[2].assign(TAU * self.net[2] + (1 - TAU) * self.target_net[2]),
            self.target_net[3].assign(TAU * self.net[3] + (1 - TAU) * self.target_net[3]),
            self.target_net[4].assign(TAU * self.net[4] + (1 - TAU) * self.target_net[4]),
            self.target_net[5].assign(TAU * self.net[5] + (1 - TAU) * self.target_net[5]),
            self.target_net[6].assign(TAU * self.net[6] + (1 - TAU) * self.target_net[6]),
            self.target_net[7].assign(TAU * self.net[7] + (1 - TAU) * self.target_net[6]),
        ])

    def train(self,y_batch,state_batch,action_batch,time_step):
        self.time_step += 1
        self.sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.lstm_layer_input:state_batch,
            self.action_input:action_batch,
            self.initial_lstm_state_forward:self.output_state_fw_feed,
            self.initial_lstm_state_backward: self.output_state_bw_feed,
            self.step_size : time_step})

    def gradients(self,state_batch,action_batch,time_step):
        return self.sess.run(self.action_gradients,feed_dict={
            self.lstm_layer_input:state_batch,
            self.action_input:action_batch,
            self.initial_lstm_state_forward: self.output_state_fw_feed,
            self.initial_lstm_state_backward: self.output_state_bw_feed,
            self.step_size: time_step})[0]

    def target_q(self,state_batch,action_batch, time_step):
        target_q_value = self.sess.run(self.target_q_value_output,feed_dict={
            self.lstm_layer_input:state_batch,
            self.target_action_input:action_batch,
            self.initial_lstm_state_forward: self.output_state_fw_feed,
            self.initial_lstm_state_backward: self.output_state_bw_feed,
            self.step_size : time_step})
        return target_q_value

    def q_value(self,state_batch,action_batch,time_step):
        return self.sess.run(self.q_value_output,feed_dict={
            self.lstm_layer_input:state_batch,
            self.action_input:action_batch,
            self.target_initial_lstm_state_forward: self.output_state_fw_feed,
            self.target_initial_lstm_state_backward: self.output_state_bw_feed,
            self.step_size:time_step
        })


'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
