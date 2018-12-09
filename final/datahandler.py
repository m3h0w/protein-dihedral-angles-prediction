import glob
import tensorflow as tf
import numpy as np

class DataHandler:
    def __init__(self, data_path, casps, num_epochs, mode='training', percentages=None):
        if mode == 'training':
            assert percentages

        self.NUM_AAS = 20
        self.NUM_DIMENSIONS = 3
        
        # choose paths from which to get training data
        self.paths = []
        for casp in casps:
            if mode == 'training':
                self.paths += [data_path + casp + '/'+mode+'/' + str(perc) + '/*' for perc in percentages]
            else:
                self.paths += [data_path + casp + '/'+mode+'/*']
        print(self.paths)
        
        # load all the file names from these paths
        base_names = self._load_training_paths()
        
        # this queue is taking all the files and asynchronously
        # passes them forward (so that the rest of the computational
        # graph that actually does computation doesn't have to wait for new input)
        self.file_queue = tf.train.string_input_producer(
            tf.convert_to_tensor(base_names),
            num_epochs=num_epochs,
            shuffle=True # not sure if this shuffle works
        )
        
    def _load_training_paths(self):
        base_names = [glob.glob(a_path) for a_path in self.paths]
        base_names = list(np.concatenate(base_names))
        self.training_samples = np.sum([self._get_num_records(file) for file in base_names])
        print("Training samples available", self.training_samples)
        return base_names
    
    def generate_batches(self, batch_size, capacity, max_protein_len=None):
        # dynamic pad makes sure that the length of the proteins
        # is padded to the longest protein in the batch
        id_, one_hot_primary, evolutionary, secondary, tertiary, ter_mask, pri_length, keep = \
            tf.train.batch(
              self.read_protein(self.file_queue, max_length=max_protein_len), 
              batch_size=batch_size, 
              capacity=capacity, 
              dynamic_pad=True
            )
        return id_, one_hot_primary, evolutionary, secondary, tertiary, ter_mask, pri_length, keep
    
    # helper to count number of records in a TF record file
    def _get_num_records(self, tf_record_file):
        return len([x for x in tf.python_io.tf_record_iterator(tf_record_file)])

    # author: AlQuraishi
    def masking_matrix(self, mask, name=None):
        """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding. 

        Args:
            mask: 0/1 vector indicating whether a position should be masked (0) or not (1)

        Returns:
            A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
            [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
        """

        with tf.name_scope(name, 'masking_matrix', [mask]) as scope:
            mask = tf.convert_to_tensor(mask, name='mask')

            mask = tf.expand_dims(mask, 0)
            base = tf.ones([tf.size(mask), tf.size(mask)])
            matrix_mask = base * mask * tf.transpose(mask)

            return matrix_mask
    
    # author: AlQuraishi
    def read_protein(self, filename_queue, max_length, num_evo_entries=21, name=None):
        """ Reads and parses a ProteinNet TF Record. 

            Primary sequences are mapped onto 20-dimensional one-hot vectors.
            Evolutionary sequences are mapped onto num_evo_entries-dimensional real-valued vectors.
            Secondary structures are mapped onto ints indicating one of 8 class labels.
            Tertiary coordinates are flattened so that there are 3 times as many coordinates as 
            residues.

            Evolutionary, secondary, and tertiary entries are optional.

        Args:
            filename_queue: TF queue for reading files
            max_length:     Maximum length of sequence (number of residues) [MAX_LENGTH]. Not a 
                            TF tensor and is thus a fixed value.

        Returns:
            id: string identifier of record
            one_hot_primary: AA sequence as one-hot vectors
            evolutionary: PSSM sequence as vectors
            secondary: DSSP sequence as int class labels
            tertiary: 3D coordinates of structure
            matrix_mask: Masking matrix to zero out pairwise distances in the masked regions
            pri_length: Length of amino acid sequence
            keep: True if primary length is less than or equal to max_length
        """

        with tf.name_scope(name, 'read_protein', []) as scope:
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            context, features = tf.parse_single_sequence_example(serialized_example,
                                    context_features={'id': tf.FixedLenFeature((1,), tf.string)},
                                    sequence_features={
                                        'primary':      tf.FixedLenSequenceFeature((1,),               tf.int64),
                                        'evolutionary': tf.FixedLenSequenceFeature((num_evo_entries,), tf.float32, allow_missing=True),
                                        'secondary':    tf.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                                        'tertiary':     tf.FixedLenSequenceFeature((self.NUM_DIMENSIONS,),  tf.float32, allow_missing=True),
                                        'mask':         tf.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)})
            id_ = context['id'][0]
            primary =   tf.to_int32(features['primary'][:, 0])
            evolutionary =          features['evolutionary']
            secondary = tf.to_int32(features['secondary'][:, 0])
            tertiary =              features['tertiary']
            mask =                  features['mask'][:, 0]

            pri_length = tf.size(primary)
            if not max_length:
                max_length = 100000
            
            keep = pri_length <= max_length

            one_hot_primary = tf.one_hot(primary, self.NUM_AAS)

            # Generate tertiary masking matrix--if mask is missing then assume all residues are present
            mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones([pri_length]))
            ter_mask = self.masking_matrix(mask, name='ter_mask')        

            return id_, one_hot_primary, evolutionary, secondary, tertiary, mask, pri_length, keep