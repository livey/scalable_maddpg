import tensorflow as tf
import numpy as np
import math

# target updating rate
TAU = .001
L2 = .0001
LEARNING_RATE = 1e-3
preLayer1Size = 20
preLayer2Size = 20
sufLayerSize = 20
SUMMARY_DIR ='summaries/'

class CriticNetwork:

    ''''for critic network,
    the input is the (states,actions) for every agents,
    output is the Q(s,a) value for each agents'''
    def __init__(self,sess,stateDimension,actionDimension):
        self.time_step = 0
        self.sess = sess
        self.actionDimension = actionDimension
        self.stateDimension  = stateDimension

        # create critic network
        self.stateInputs,\
        self.actionInputs,\
        self.q_value_outputs,\
        self.nets = self.createQNetwork(stateDimension,actionDimension)

        # construct target q network
        self.target_q_value_outputs, \
        self.target_update = self.create_target_network(self.q_value_outputs, self.nets)

        # create training methods
        self.create_training_method()

        # merge all the summaries

        self.summaries_writer,\
            self.merge_summaries = self.collect_summaries()

        self.init_new_variables()

        self.update_target()

    def createQNetwork(self,stateDimension,actionDimension):
        cell_units = preLayer2Size
        with tf.variable_scope('criticNetwork') as scope:
            # the input state training data  is batchSize*numOfAgents*stateDimension
            stateInputs = tf.placeholder('float',[None,None,stateDimension])
            # the input action training data is batchSize*numOfAgents*stateDimension
            actionInputs = tf.placeholder('float',[None,None,actionDimension])
            # get the batch size, and numOfAgents
            batchSize = tf.shape(stateInputs)[0]
            numOfAgents = tf.shape(stateInputs)[1]

            # construct the input DNN for bidirectional LSTM
            # reshape the input data with size (batchSize*NumOfAgents)*featureDimension
            inputDNNstate  = tf.reshape(stateInputs,[-1,stateDimension])
            inputDNNaction = tf.reshape(actionInputs,[-1,actionDimension])
            preW1S = tf.get_variable('preW1S',[stateDimension,preLayer1Size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            preB1S = tf.get_variable('preB1S',[preLayer1Size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            preW2S = tf.get_variable('preW2S',[preLayer1Size,preLayer2Size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            preW2A = tf.get_variable('preW2A',[actionDimension,preLayer2Size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            preB2  = tf.get_variable('preB2',[preLayer2Size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            preLayer1 = tf.nn.relu(tf.matmul(inputDNNstate,preW1S)+preB1S)
            preLayer2 = tf.nn.relu(tf.matmul(preLayer1,preW2S)
                                   +tf.matmul(inputDNNaction,preW2A)
                                   +preB2)
            lstmInputs = tf.reshape(preLayer2,[batchSize,numOfAgents,preLayer2Size])

            # construct the bidirectional LSTM
            # make sure each epoch the init sate is set to be zero
            #https://stackoverflow.com/questions/38441589/is-rnn-initial-state-reset-for-subsequent-mini-batches/41239965#41239965
            with tf.variable_scope('forward_lstm'):
                lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_units)
            with tf.variable_scope('backward_lstm'):
                lstm_backward_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_units)

            (outputs, output_state) = tf.nn.bidirectional_dynamic_rnn(
                lstm_forward_cell,
                lstm_backward_cell,
                lstmInputs,
                dtype='float',
                #initial_state_fw=initial_lstm_state_forward_input,
                #initial_state_bw=initial_lstm_state_backward_input,
                #sequence_length=step_size,
                time_major=False,
                scope=scope)
            first_layer_output = tf.reshape(outputs[0],[-1,cell_units])
            second_layer_output = tf.reshape(outputs[1],[-1,cell_units])
            suf_w1 = tf.get_variable('suf_w1',[cell_units,1],
                                     initializer=tf.contrib.layers.xavier_initializer())
            suf_w2 = tf.get_variable('suf_w2',[cell_units,1],
                                     initializer=tf.contrib.layers.xavier_initializer())
            suf_b  = tf.get_variable('suf_b',initializer=tf.random_uniform([1],-3e-3,3e-3))

            q_value1 = tf.identity(tf.matmul(first_layer_output,suf_w1)
                                  +tf.matmul(second_layer_output,suf_w2)
                                  +suf_b)
            q_value = tf.reshape(q_value1,[batchSize,-1])

        nets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='criticNetwork')
        return stateInputs, actionInputs, q_value, nets


    def create_target_network(self,q_output, nets):
        #state_input = tf.placeholder('float', [None,None,stateDimension])
        #action_input = tf.placeholder('float', [None,None,actionDimension])
        ##https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        ## how to use https://stackoverflow.com/questions/45206910/tensorflow-exponential-moving-average
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU,zero_debias=True)
        target_update = ema.apply(nets)
        # reference using
        #http://web.stanford.edu/class/cs20si/lectures/slides_14.pdf
        # on page 22
        # get the after averaged weights
        # copy this the Q network, but with the target network weights

        # the difference between operation and the result of that operation
        # Variable has the function value()
        replace_ts = {}
        for tt in nets:
            temp_ts = ema.average(tt)
            replace_ts.update({tt.value(): temp_ts.value()}) # Tensor to Tensor
        # graph_replace
        # https://www.tensorflow.org/api_docs/python/tf/contrib/graph_editor/graph_replace
        target_q_value = tf.contrib.graph_editor.graph_replace(q_output, replace_ts)

        return target_q_value, target_update

    def create_training_method(self):
        # the expected size of Rt is batch_size* agents
        self.Rt = tf.placeholder('float', [None, None])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.nets])
        self.cost = tf.reduce_mean(tf.square(self.Rt - self.q_value_outputs)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        mean_rewards = tf.reduce_mean(self.q_value_outputs)
        tf.summary.scalar('mean_Q_value', mean_rewards)
        self.action_gradients = tf.gradients(mean_rewards, self.actionInputs)

    def train(self,Rt,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(
            self.optimizer,feed_dict={
                self.Rt : Rt,
                self.stateInputs: state_batch,
                self.actionInputs: action_batch
            }
        )

    def target_q(self,state_batch,action_batch):
        return self.sess.run(
            self.target_q_value_outputs, feed_dict={
            self.stateInputs: state_batch,
            self.actionInputs: action_batch})

    def printnets(self):
        for nn in self.nets:
            print(nn)

    def q_value(self, stateInputs, actionInputs):
        return self.sess.run(self.q_value_outputs,feed_dict={
            self.stateInputs: stateInputs, self.actionInputs: actionInputs})

    def update_target(self):
        self.sess.run(self.target_update)

    def gradients(self,state_batch,action_batch):
        return self.sess.run(
            self.action_gradients,feed_dict={
                self.stateInputs: state_batch,
                self.actionInputs: action_batch
            }
        )[0]

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_outputs,feed_dict={
            self.stateInputs: state_batch,
            self.actionInputs: action_batch
        })

    def collect_summaries(self):
        summaries = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        return summary_writer, summaries

    def write_summaries(self,state_batch, action_batch, record_num):
        summ = self.sess.run(self.merge_summaries, feed_dict={self.stateInputs: state_batch,
                                                              self.actionInputs: action_batch})
        self.summaries_writer.add_summary(summ, record_num)

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
        uninit_variables =  [v for v in list_of_variables if
             v.name.split(':')[0] in uninit_names]
        ss = tf.variables_initializer(uninit_variables)
        self.sess.run(ss)

    # def load_network(self):
    #     checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
    #     if checkpoint and checkpoint.model_checkpoint_path:
    #         self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
    #         print("Successfully loaded:", checkpoint.model_checkpoint_path)
    #     else:
    #         print('Could not find old network weights')
    #
    # def save_network(self,time_step):
    #     print('save critic-network...',time_step)
    #     self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step=time_step)
