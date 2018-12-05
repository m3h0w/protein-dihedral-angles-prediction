import tensorflow as tf
from final.helpers import Helpers

class Model:
    def __init__(self, model_type='cnn_big', ang_mode='cos'):
        
        self.core = {
            'cnn_big': lambda input_data: self.Models.resnet1d_big(input_data),
            'cnn_small': lambda input_data: self.Models.resnet1d_small(input_data),
            'bilstm': lambda input_data: self.Models.bidirectional_lstm(input_data, num_layers=1, rnn_size=128, dropout=0.01, lengths=None)
        }[model_type]
        
        self.angularize = {
            'tanh': lambda input_data: Helpers.angularize(input_data, mode='tanh'),
            'cos': lambda input_data: Helpers.angularize(input_data, mode='cos')
        }[ang_mode]
    
    def build_model(self, input_data):
        
        core_out = self.core(input_data)
        
        # squeezing the output into tanh with 3 outputs
        pred = tf.layers.dense(core_out, 3, use_bias=False)
        
        # rescaling the output to match the scale of the angles (-180, 180)
        # pred = tf.multiply(pred, tf.constant(180.))
        pred = self.angularize(pred)
        
        return pred

    class Models:  
        @staticmethod
        def bidirectional_lstm(input_data, num_layers, rnn_size, dropout, lengths):
            outputs, _ = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, rnn_size, input_mode='linear_input', 
                                                                         direction='bidirectional', dropout=dropout
                                                                         )(input_data)
            return outputs

        @staticmethod
        def resnet1d_big(input_data):
            # helper for setting up a resnet cnn
            conv1a = Helpers.conv_layer(input_data, 32, 5)
            conv1b = Helpers.conv_layer(conv1a, 32, 5)
            conv2a = Helpers.conv_layer(conv1b, 64, 5)
            conv2b = Helpers.conv_layer(conv2a, 64, 5)
            conv3a = Helpers.conv_layer([conv1b, conv2b], 128, 5) # residual connection is automated in the conv_layer helper
            conv3b = Helpers.conv_layer(conv3a, 128, 5)
            conv4a = Helpers.conv_layer([conv2b, conv3b], 256, 5)
            conv4b = Helpers.conv_layer(conv4a, 256, 5)
            return tf.layers.dropout(conv4b, rate=0.1)
        
        @staticmethod
        def resnet1d_small(input_data):
            # helper for setting up a resnet cnn
            conv1a = Helpers.conv_layer(input_data, 32, 5)
            conv1b = Helpers.conv_layer(conv1a, 32, 5)
            conv2a = Helpers.conv_layer(conv1b, 64, 5)
            conv2b = Helpers.conv_layer(conv2a, 64, 5)
            conv3a = Helpers.conv_layer([conv1b, conv2b], 128, 5) # residual connection is automated in the conv_layer helper
            return tf.layers.dropout(conv3a, rate=0.1)