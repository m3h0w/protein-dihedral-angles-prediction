import numpy as np
import scipy
import tensorflow as tf

class Helpers:
    @staticmethod
    def get_phi_psi_omega(dihedrals):
        dih = np.array(dihedrals)
        return dih[:,:,0], dih[:,:,1], dih[:,:,2]

    @staticmethod
    def tf_rad2deg(rad):
        pi_on_180 = 0.017453292519943295
        return rad / pi_on_180

    @staticmethod
    @tf.custom_gradient
    def clip_grad_layer(x, eps=1e-10):
        def grad(dy):
            return tf.clip_by_value(dy, eps, eps)
        return tf.identity(x), grad
    
    @staticmethod
    def input_placeholders(max_len, input_columns):
        X = tf.placeholder(tf.float32, [None, max_len, input_columns], name="X")
        input_mask = tf.placeholder(tf.bool, [None, max_len], name="input_mask")
        return X, input_mask

    @staticmethod
    def labels_placeholders(max_len, n_clusters, n_angles):
        y = tf.placeholder(tf.float32, [None, max_len, n_clusters], name='y')
        y_angles = tf.placeholder(tf.float32, [None, max_len, n_angles], name='y_angles')
        return y, y_angles

    @staticmethod
    def clusters(init_clusters, trainable=False, clip_gradient_eps=None):
        clusters_tf = tf.Variable(initial_value=init_clusters, dtype=np.float32, trainable=trainable)
        clusters_tf = tf.clip_by_value(clusters_tf, -np.pi, np.pi)
        if clip_gradient_eps:
            clusters_tf = Helpers.clip_grad_layer(clusters_tf, eps=clip_gradient_eps)
        return clusters_tf

    @staticmethod
    def conv_layer(in_, neurons, filter_size):
        if type(in_) == list:
            in_ = tf.concat(in_, axis=2)
        return tf.layers.conv1d(in_, neurons, filter_size, activation=tf.nn.relu, padding='same')

    @staticmethod
    def mask_all(tensors_list, mask, axis=None):
        res = []
        for tensor in tensors_list:
            if axis:
                res.append(tf.boolean_mask(tensor, mask, axis=axis))
            else:
                res.append(tf.boolean_mask(tensor, mask))
        return res
    
    @staticmethod
    def angularize(input_tensor, mode='cos', name=None):
        """ Restricts real-valued tensors to the interval [-pi, pi] by feeding them through a cosine. """

        with tf.name_scope(name, 'angularize', [input_tensor]) as scope:
            input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')
            
            if mode=='cos':
                return tf.multiply(np.pi, tf.cos(input_tensor + (np.pi / 2)), name=scope)
            elif mode=='tanh':
                return tf.multiply(np.pi, tf.tanh(input_tensor), name=scope)
            else:
                raise KeyError(mode, 'is an invalid angularization mode')

    @staticmethod
    def vec_to_angle(input_tensor):
        return tf.math.atan2(input_tensor[:,:,1], input_tensor[:,:,0])
    
    @staticmethod
    def mae(y_true, y_pred):
        if tf.shape(y_true).get_shape()[0] == 2:
            axis = 0
        elif tf.shape(y_true).get_shape()[0] == 3:
            axis = [0,1]
        else:
            raise ValueError("y_true is neither 2- nor 3-dimensional")
        return tf.reduce_mean(tf.abs(tf.subtract(y_true[:,:], y_pred[:,:])), axis=axis)
    
    @staticmethod
    def loss360(y_true, y_pred):
        if tf.shape(y_true).get_shape()[0] == 2:
            axis = 0
        elif tf.shape(y_true).get_shape()[0] == 3:
            axis = [0,1]
        else:
            raise ValueError("y_true is neither 2- nor 3-dimensional")
        return tf.reduce_mean(tf.abs(tf.atan2(tf.sin(y_true - y_pred), tf.cos(y_true - y_pred))), axis=axis)
    
    @staticmethod
    def pearson_numpy(y_true, y_pred):
        n_angles = y_true.shape[-1]
        a_angles = np.split(np.cos(y_true.reshape(-1,)).reshape(-1,n_angles), 
                                        indices_or_sections=n_angles, axis=-1)
        b_angles = np.split(np.cos(y_pred.reshape(-1,)).reshape(-1,n_angles),
                                        indices_or_sections=n_angles, axis=-1)
        return [scipy.stats.pearsonr(a.reshape(-1,), b.reshape(-1,))[0] for a,b in zip(a_angles, b_angles)]