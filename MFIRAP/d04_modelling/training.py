import numpy as np
import glob
import os
import tensorflow as tf
import random
import MFIRAP.d00_utils.project as project

class Train_Validation_Generators:
    def __init__(self, dataset_path, view_IDs, train_size, batch_size=32, shuffle=True, RGB=False):
        self.devel_path = os.path.join(dataset_path, "development")    
        self.train_size = train_size
        self.view_IDs = view_IDs.copy()
        self.batch_size = batch_size
        self.RGB = RGB
        self.shuffle = shuffle
        if train_size > 1:
            train_size = 1
        if (train_size == -1):
            train_size = 1.0
        self.validation_size = 1-train_size
        def _read_samples_in_subfolder(zero_or_one):
            fp_list = glob.glob(os.path.join(self.devel_path, str(zero_or_one), "*.npz"))
            return [np.load(f) for f in fp_list]
        pos_samples = _read_samples_in_subfolder(1)
        neg_samples = _read_samples_in_subfolder(0)
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)
        m = len(pos_samples) if len(pos_samples) < len(neg_samples) else len(neg_samples)
        split_pt = int(self.train_size * m)
        self.train_pos_samples, self.train_neg_samples = pos_samples[:split_pt], neg_samples[:split_pt]
        self.valid_pos_samples, self.valid_neg_samples = pos_samples[split_pt:], neg_samples[split_pt:]
    
    def get_train(self):
        return Data_generator(self.train_pos_samples, self.train_neg_samples, view_IDs = self.view_IDs, batch_size = self.batch_size, shuffle = self.shuffle, RGB = self.RGB)
    
    def get_valid(self):
        return Data_generator(self.valid_pos_samples, self.valid_neg_samples, view_IDs = self.view_IDs, batch_size = self.batch_size, shuffle = self.shuffle, RGB = self.RGB)
        
class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, pos_samples, neg_samples, view_IDs, batch_size=32, shuffle=True, RGB=False):
        self.pos_samples = pos_samples.copy()
        self.neg_samples = neg_samples.copy()
        self.view_IDs = view_IDs.copy()
        self.RGB = RGB
        self.keys = ['array_ID{}'.format(id) for id in self.view_IDs]
        if self.RGB:
            self.keys += ['array_IDRGB']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pos_in_batch_n = self.batch_size // 2
        self.neg_in_batch_n = self.pos_in_batch_n
    
    def __len__(self):
        m = len(self.pos_samples) if len(self.pos_samples) < len(self.neg_samples) else len(self.neg_samples)
        return int(np.floor(m) / self.batch_size)
    
    def __getitem__(self, index):
        indices = list(range(index * self.pos_in_batch_n, (index + 1) * self.pos_in_batch_n))
        return self._load_data(indices)
    
    def _load_data(self, indices):
        data =  [[] for i in range(len(self.keys))]
        Y = []
        for idx in indices:
            samples = [self.pos_samples[idx], self.neg_samples[idx]]
            for sample in samples:
                [data[i].append(sample[id]) for i, id in enumerate(self.keys)]
                Y.append(sample['one_hot'])
        # Y [B,2] > [B, F, 2]
        Y = np.tile(np.expand_dims(np.array(Y), axis=1), [1, sample['frames'], 1])
        x = [np.array(view) for view in data]
        assert all([len(view) == len(Y) for view in x])
        return x, Y
        
    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.pos_samples)
            random.shuffle(self.neg_samples)



class Losses_Keras:
    def __init__(self, frames, frame_shift):
        self.frames = frames
        self.frame_shfit = frame_shift
    
    def get_by_name(self, name, **kwargs):
        if name == "exponential_loss":
            return self.get_exponential_loss(**kwargs)

    def get_exponential_loss(self, from_logits=False):
        def exponential_loss(y_true, y_pred, from_logits=from_logits):
            # [B, F, 2], [B, F, 2]
            # TODO ADD JAIN
            # L_p = sigma_t(-exp())
            # L_n = softmax_cross_entropy
            # EL = L_p + L_n
            # https://stackoverflow.com/questions/39192380/tensorflow-one-class-classification
            # https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/backend/cntk_backend.py#L106
            if not tf.is_tensor(y_pred):
                y_pred = tf.constant(y_pred, dtype=project.FLOATX)
            y_true = tf.cast(y_true, y_pred.dtype)
            if from_logits:
                y_pred = tf.keras.activations.sigmoid(y_pred)
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
            log_pos = (y_true)*tf.math.log(y_pred)
            log_neg = (1.0 - y_true)*tf.math.log(1.0 - y_pred)
            Y = tf.cast((tf.shape(y_true)[-1]), project.FLOATX)
            k = tf.cast(self.frames/5, project.FLOATX)
            exp = tf.cast(tf.math.exp(-(Y-tf.range(Y)-1)/k), project.FLOATX)
            positive_loss = -tf.reduce_sum(tf.broadcast_to(exp, tf.shape(log_pos)) * log_pos)
            negative_loss = -tf.reduce_sum(log_neg)
            total_loss = positive_loss + negative_loss
            return total_loss
        return exponential_loss