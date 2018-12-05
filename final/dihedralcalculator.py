import tensorflow as tf

class DihedralCalculator:
    @staticmethod
    def _get_dihedrals_from_euclidean(p):
        # takes a 4-dimensional tensor (N, K, 4, 3) and outputs (N, K, 3) angles
        p0 = tf.gather(p, 0, axis=2)
        p1 = tf.gather(p, 1, axis=2)
        p2 = tf.gather(p, 2, axis=2)
        p3 = tf.gather(p, 3, axis=2)

        b0 = -1.0 * (tf.subtract(p1, p0))
        b1 = tf.subtract(p2, p1)
        b2 = tf.subtract(p3, p2)

        b1 = tf.divide(b1, tf.norm(b1, axis=2, keepdims=True))
        b1 = tf.where(tf.is_nan(b1), tf.ones_like(b1), b1) # what to do when norm is 0?

        v = tf.subtract(b0, tf.einsum('bi,bij->bij', tf.einsum('bij,bij->bi', b0, b1), b1))
        w = tf.subtract(b2, tf.einsum('bi,bij->bij', tf.einsum('bij,bij->bi', b2, b1), b1))

        x = tf.reduce_sum( tf.multiply( v, w ), 2, keepdims=True )
        y = tf.reduce_sum( tf.multiply( tf.cross(b1, v), w ), 2, keepdims=True )

        return tf.atan2(y,x)
    
    @staticmethod
    def dihedral_pipeline(euclidean_coordinates, protein_length):
        # euclidean_coordinates are of shape (batch_size, protein_length, 3)
        
        # chooses all possible slices of length 4
        batch_size = tf.shape(euclidean_coordinates)[0]
        euclidean_coordinates = euclidean_coordinates[:,:,:,None]
        all_4_len_slices_euc_coord = tf.extract_image_patches(euclidean_coordinates,
          ksizes=[1, 4, 3, 1],
          strides=[1, 1, 1, 1],
          rates=[1, 1, 1, 1],
          padding='VALID')
        all_4_len_slices_euc_coord = tf.reshape(tf.squeeze(all_4_len_slices_euc_coord), [batch_size, -1, 4, 3])

        # calculates torsional angles on the entire batch
        dihedral_angles = DihedralCalculator._get_dihedrals_from_euclidean(all_4_len_slices_euc_coord)

        # adds 3 zeros at the end because I can't calculate the angle of
        # the last 3 atmos (need at least 4 atoms to calculate an angle)
        padding = tf.constant([[0, 0], [1,2], [0,0]])
        dihedral_angles = tf.pad(dihedral_angles, padding)

        # reshaping the angles (because input is 3 times the length of normal protein)
        dihedral_angles_shape = tf.gather(tf.shape(dihedral_angles), [0,1])
        dihedral_angles = tf.reshape(dihedral_angles, shape=dihedral_angles_shape)
        return tf.reshape(dihedral_angles, shape=(tf.gather(dihedral_angles_shape, 0), protein_length, 3))    