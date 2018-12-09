import tensorflow as tf
from final.helpers import Helpers
import numpy as np

class Model:
    """ 
    Model predicting n_angles dihedral angles in radians by first constructing an
    intermmediary representation using a cnn or an lstm and then converting the output
    to angles through a dense layer.

    See readme.md for more information on possible configurations.
    """
    def __init__(self, n_angles, output_mask, 
                 model_type='cnn_big', prediction_mode='regression_angles', loss_mode='angular_mae',
                 dropout_rate=0., ang_mode='undefined', regularize_vectors=True, n_clusters=None):
        """ 
        n_angles: 2 - only predict phi and psi, or 3 - predict phi, psi and omega
        model_type: decides on core architecture
        prediction_mode: decides how to convert core output into angles
        ang_mode: decides on how to angularize regression output if normal regression is chosen
        regularize_vectors: True if vectors should be kept on a unit circle using a (1-x^2-y^2)^2 loss

        See readme.md for more information on possible configurations.
        """
        if prediction_mode=='alphabet' and n_clusters is None:
            raise self.WrongModelConfiguration('In \'alphabet\' mode number of clusters (n_clusters) has to be specified')
        if n_clusters is not None:
            assert n_clusters > 0 and n_clusters < 500

        self.output_mask = output_mask

        self.n_angles = { # restricting options to 2 and 3
            2: 2,
            3: 3
        }[n_angles] 

        self.n_clusters = n_clusters
        self.prediction_mode = prediction_mode
        self.loss_mode = loss_mode

        self.core_out_model = {
            'cnn_big': lambda input_: self.CoreModels.resnet1d_big(input_, dropout_rate),
            'cnn_small': lambda input_: self.CoreModels.resnet1d_small(input_, dropout_rate),
            'bilstm': lambda input_: self.CoreModels.bidirectional_lstm(input_, num_layers=2, 
                rnn_size=300, dropout=dropout_rate, lengths=None)
        }[model_type]
        
        if self.prediction_mode == 'regression_angles':
            self.angularize = {
                'tanh': lambda input_: Helpers.angularize(input_, mode='tanh'),
                'cos': lambda input_: Helpers.angularize(input_, mode='cos'),
                'undefined': lambda: self.raise_wrong_model_configuration('ang_mode has to be defined for regression_angles prediction mode')
            }[ang_mode]

        self.rad_pred_model = {
            'regression_angles': lambda input_: self.PredictionModels.regression_angles(input_, self.n_angles, self.output_mask,
                                                                                        angularize=self.angularize),
            'regression_vectors': lambda input_: self.PredictionModels.regression_vectors(input_, self.n_angles, 
                                                                                          self.output_mask, 
                                                                                          regularize_vectors=regularize_vectors),
            'alphabet_vectors': lambda input_: self.PredictionModels.alphabet_vectors(input_, self.n_angles, self.n_clusters,
                                                                              self.output_mask
                                                                              ),
            'alphabet_angles': lambda input_: self.PredictionModels.alphabet_angles(input_, self.n_angles, self.n_clusters,
                                                                              self.output_mask, regularize_vectors=regularize_vectors,
                                                                              ),  
        }[self.prediction_mode]

        self.regularization_losses = []
    
    def build_model(self, input_data):
        core_out = self.core_out_model(input_data)
        pred_masked, reg_losses = self.rad_pred_model(core_out)
        self.regularization_losses += reg_losses
        return pred_masked

    def calculate_loss(self, true_dihedrals_masked, rad_pred_masked,
                       true_vectors_masked=None, vec_pred_masked=None):

        if self.prediction_mode == 'regression_vectors' or self.prediction_mode == 'alphabet_vectors':
            if true_vectors_masked is None or vec_pred_masked is None:
                raise WrongModelConfiguration('Vectors have to be passed to loss calculation if prediction mode involves vectors')
            # vectors are returned together with angles thus mae can be calculated directly on vectors
            if self.loss_mode == 'angular_mae':
                loss_vec = Helpers.loss360(true_dihedrals_masked, rad_pred_masked)
                loss = tf.reduce_mean(loss_vec)
            elif self.loss_mode == 'mae':
                loss_vec = Helpers.mae(true_vectors_masked, vec_pred_masked)
                loss = tf.reduce_mean(loss_vec)
            elif self.loss_mode == 'angular_mae_and_mae':
                loss_vec = tf.add(Helpers.loss360(true_dihedrals_masked, rad_pred_masked), 
                                Helpers.mae(true_vectors_masked, vec_pred_masked))
                loss = tf.reduce_mean(loss_vec)
        
        elif self.prediction_mode == 'regression_angles' or self.prediction_mode == 'alphabet_angles':
            if vec_pred_masked:
                raise WrongModelConfiguration('Predicted vectors were passed to loss calculation even though the prediction mode involves only angles')
            # no vectors are predicted, thus 'mae' is calculated on angles
            if self.loss_mode == 'angular_mae':
                loss_vec = Helpers.loss360(true_dihedrals_masked, rad_pred_masked)
                loss = tf.reduce_mean(loss_vec)
            elif self.loss_mode == 'mae':
                loss_vec = Helpers.mae(true_dihedrals_masked, rad_pred_masked)
                loss = tf.reduce_mean(loss_vec)

        loss = tf.add_n([loss] + self.regularization_losses)

        return loss, loss_vec

    def mask_other(self, inputs):
        return Helpers.mask_all(inputs, self.output_mask)

    class PredictionModels:
        """ Definitions of how LSTM or CNN output is converted into angles
        using a fully connected layer """
        @staticmethod
        def regression_angles(core_out, n_angles, output_mask, angularize):
            rad_pred_cont = tf.layers.dense(core_out, n_angles) # continuous value before angualrization
            rad_pred = angularize(rad_pred_cont) # chosen in Model constructor with ang_mode parameter
            rad_pred_masked = tf.boolean_mask(rad_pred, output_mask)
            return (rad_pred_masked, None), []
        
        @staticmethod
        def regression_vectors(core_out, n_angles, output_mask, regularize_vectors):
            if regularize_vectors: # have to squeeze values with tanh if regularization is used
                activation='tanh'
            else:
                activation='linear'
            
            vec_pred = tf.layers.dense(core_out, n_angles * 2, activation=activation) # angles logits

            vec_pred_r = tf.reshape(vec_pred, shape=(-1, tf.shape(vec_pred)[1], n_angles, 2)) # reshape to 2 numbers (vector) per angle

            vec_pred_masked_r = tf.boolean_mask(vec_pred_r, output_mask)

            losses_to_return = []
            if regularize_vectors:
                losses_to_return = [tf.reduce_mean(tf.square(1 - (vec_pred_masked_r[:,:,0] ** 2 + vec_pred_masked_r[:,:,1]**2)))] # regularization loss that keeps vectors close to 1
        #     vec_pred = vec_pred / (tf.expand_dims(tf.sqrt(vec_pred[:,:,:,0]**2 + vec_pred[:,:,:,1]**2), 3)+1) # normalize to length 1 (alternative to above - doesn't work)
            rad_pred_masked = tf.atan2(vec_pred_masked_r[:,:,1], vec_pred_masked_r[:,:,0]) # convert vector to angle
            return (rad_pred_masked, vec_pred_masked_r), losses_to_return

        @staticmethod
        def alphabet_vectors(core_out, n_angles, n_clusters, output_mask):

            clusters = Helpers.ang_to_vec(np.asarray(np.random.uniform(low=-np.pi, high=np.pi, size=(n_clusters, n_angles)), dtype=np.float32))
            clusters_tf = tf.Variable(initial_value=clusters, dtype=np.float32, trainable=True)
            clusters_tf = tf.clip_by_value(clusters_tf, -1, 1)

            logits = tf.layers.dense(core_out, n_clusters) # cluster logits
            logits = tf.layers.dropout(logits, rate=0.05)
            y_pred = tf.nn.softmax(logits)
            y_pred_masked = tf.boolean_mask(y_pred, output_mask)

            vec_pred_masked = tf.einsum('ij,bi->bj', clusters_tf, y_pred_masked)
            vec_pred_masked_r = tf.reshape(vec_pred_masked, shape=(-1,n_angles,2))

            rad_pred_masked = tf.atan2(vec_pred_masked_r[:,:,1], vec_pred_masked_r[:,:,0])
            return (rad_pred_masked, vec_pred_masked_r), []

        @staticmethod
        def alphabet_angles(core_out, n_angles, n_clusters, output_mask, regularize_vectors):

            clusters = np.asarray(np.random.uniform(low=-np.pi, high=np.pi, size=(n_clusters, n_angles)), dtype=np.float32)
            clusters_tf = tf.Variable(initial_value=clusters, dtype=np.float32, trainable=True)
            clusters_tf = tf.clip_by_value(clusters_tf, -np.pi, np.pi)

            logits = tf.layers.dense(core_out, n_clusters) # cluster logits
            logits = tf.layers.dropout(logits, rate=0.05)
            y_pred = tf.nn.softmax(logits)
            y_pred_masked = tf.boolean_mask(y_pred, output_mask)

            rad_pred_masked = tf.einsum('ij,bi->bj', clusters_tf, y_pred_masked)
            return (rad_pred_masked, None), []

    class CoreModels:
        """ Definitions of the core model. I.e.: how protein sequence and evolutionary profile are
        converted into intermediary representation """ 
        @staticmethod
        def bidirectional_lstm(input_data, num_layers, rnn_size, dropout, lengths):
            outputs, _ = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, rnn_size, input_mode='linear_input', 
                                                        direction='bidirectional', dropout=dropout
                                                        )(input_data)
            return outputs

        @staticmethod
        def resnet1d_big(input_data, dropout_rate):
            # helper for setting up a resnet cnn
            conv1a = Model.conv_layer(input_data, 32, 5)
            conv1b = Model.conv_layer(conv1a, 32, 5)
            conv1b = tf.layers.dropout(conv1b, rate=dropout_rate)

            conv2a = Model.conv_layer(conv1b, 64, 5)
            conv2b = Model.conv_layer(conv2a, 64, 5)
            conv2b = tf.layers.dropout(conv2b, rate=dropout_rate)

            conv3a = Model.conv_layer([conv1b, conv2b], 128, 5) # residual connection is automated in the conv_layer helper
            conv3b = Model.conv_layer(conv3a, 128, 5)
            conv3b = tf.layers.dropout(conv3b, rate=dropout_rate)

            conv4a = Model.conv_layer([conv2b, conv3b], 256, 5)
            conv4b = Model.conv_layer(conv4a, 256, 5)
            conv4b = tf.layers.dropout(conv4b, rate=dropout_rate)
            return conv4b
        
        @staticmethod
        def resnet1d_small(input_data, dropout_rate):
            # helper for setting up a resnet cnn
            conv1a = Model.conv_layer(input_data, 32, 5)
            conv1b = Model.conv_layer(conv1a, 32, 5)
            conv1b = tf.layers.dropout(conv1b, rate=dropout_rate)

            conv2a = Model.conv_layer(conv1b, 64, 5)
            conv2b = Model.conv_layer(conv2a, 64, 5)
            conv2b = tf.layers.dropout(conv2b, rate=dropout_rate)
            
            conv3a = Model.conv_layer([conv1b, conv2b], 128, 5) # residual connection is automated in the conv_layer helper
            conv3b = Model.conv_layer(conv3a, 128, 5)
            conv3b = tf.layers.dropout(conv3b, rate=dropout_rate)
            return conv3b

    @staticmethod
    def conv_layer(in_, neurons, filter_size, input_combine_mode='concat'):
        if type(in_) == list:
            in_ = {
                'concat': lambda in_: tf.concat(in_, axis=2),
                'add': lambda in_: tf.add_n(in_, axis=2)
            }[input_combine_mode](in_)
        out = tf.layers.conv1d(in_, neurons, filter_size, activation=tf.nn.relu, padding='same')
        out = tf.layers.batch_normalization(out)
        return out

    @staticmethod
    def raise_wrong_model_configuration(message):
        raise WrongModelConfiguration(message)

    class WrongModelConfiguration(Exception):
        pass