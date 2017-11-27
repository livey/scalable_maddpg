import tensorflow as tf
import numpy as np

# how to syncronize the parameters in different device
# https://stackoverflow.com/questions/37801137/duplicate-a-tensorflow-graph
# relationship between Session and Graph one session per graph, one graph can be used in multiple sessions.
# https://www.tensorflow.org/versions/r0.12/api_docs/python/client/session_management

# create multiple instances
# https://stackoverflow.com/questions/41709207/python-create-n-number-of-class-instances

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

# when adding new agents, initialize the
class ActorNetwork:
    def __init__(self,sess, state_dim, action_dim, agent_name,pre_nets = None):
        self.sess = sess
        self.agent_name = agent_name
        self.state_dim = state_dim
        self.action_dim = action_dim

        if pre_nets ==None:
            print('create new agent')
            self.state_input, self.action_output, self.nets = \
                self.create_new_network(state_dim, action_dim)
        else:
            print('create new agent use previous weights')
            self.state_input, self.action_output, self.nets = \
                self.create_copy_network(state_dim, action_dim,pre_nets)


        self.target_update, \
        self.target_action_output= self.create_target_network(
            self.action_output, self.nets)

        self.create_training_method()

        self.init_new_variables()
        self.update_target()



    def create_new_network(self,state_dim,action_dim):
        layer1_size =  LAYER1_SIZE
        layer2_size =  LAYER2_SIZE
        with tf.variable_scope(self.agent_name) as scope:
            state_input = tf.placeholder('float',[None,state_dim])
            W1 = tf.get_variable('W1',[state_dim,layer1_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('b1',[layer1_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable('W2',[layer1_size,layer2_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable('b2',[layer2_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            W3 = tf.get_variable('W3',[layer2_size,action_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable('b3',initializer=tf.random_uniform([action_dim],-3e-3,3e-3))
            layer1 = tf.nn.relu(tf.matmul(state_input,W1)+b1)
            layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
            action_ouput = tf.tanh(tf.matmul(layer2,W3)+b3)

        nets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.agent_name)
        return state_input, action_ouput,nets

    def create_copy_network(self,state_dim, action_dim, pre_nets):
        with tf.variable_scope(self.agent_name) as scope:
            state_input = tf.placeholder('float',[None,state_dim])
            W1 = tf.get_variable('W1',initializer=self.sess.run(pre_nets[0]))
            b1 = tf.get_variable('b1',initializer=self.sess.run(pre_nets[1]))
            W2 = tf.get_variable('W2',initializer=self.sess.run(pre_nets[2]))
            b2 = tf.get_variable('b2',initializer=self.sess.run(pre_nets[3]))
            W3 = tf.get_variable('W3',initializer=self.sess.run(pre_nets[4]))
            b3 = tf.get_variable('b3',initializer=self.sess.run(pre_nets[5]))
            layer1 = tf.nn.relu(tf.matmul(state_input,W1)+b1)
            layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
            action_output = tf.tanh(tf.matmul(layer2,W3)+b3)

        nets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.agent_name)
        return state_input, action_output,nets


    def create_target_network(self,action_output,nets):
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU, zero_debias=True)
        target_update = ema.apply(nets)
        replace_ts ={}
        for tt in nets:
            temp_ts = ema.average(tt)
            replace_ts.update({tt.value(): temp_ts.value()})

        target_action_output =tf.contrib.graph_editor.graph_replace(action_output,replace_ts)
        return target_update, target_action_output


    def create_training_method(self):
        self.q_gradient_input = tf.placeholder('float',[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,
                                                  self.nets,-self.q_gradient_input)

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, self.nets))

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input: state_batch
        })

    def action(self,state):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input: [state]
        })
    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input: state_batch
        })

    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.state_input: state_batch
        })


    def init_new_variables(self):
        '''init the new add variables, instead of all the variables
           it is convenient to add new agents
           https://asyoulook.com/computers%20&%20internet/tensorflow-how-to-get-the-list-of-uninitialized-variables-from-tf-report-uninitialized-variables/1730337
           https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
        '''
        list_of_variables = tf.global_variables()
        # this method returns b'strings' , so decode to string for comparison
        uninit_names = set(self.sess.run(tf.report_uninitialized_variables()))
        # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
        uninit_names = [v.decode('utf-8') for v in uninit_names]
        uninit_variables = [v for v in list_of_variables if
                            v.name.split(':')[0] in uninit_names]
        ss = tf.variables_initializer(uninit_variables)
        self.sess.run(ss)








