import numpy as np
import glob
import os
import tensorflow as tf
import random
import MFIRAP.d00_utils.project as project

K = tf.keras.backend

class Timestamper_counted_down(tf.keras.callbacks.Callback):
    def __init__(self, batch_timestamps, trainining_generator = None, validation_generator = None, testing_generator = None):
        self.batch_timestamps = batch_timestamps   
        self.trainining_generator = trainining_generator
        self.validation_generator = validation_generator
        self.testing_generator = testing_generator

    def _get_batch_timestamps_counted_down(self, batch, generator):
        ts = generator._get_batch_timestamps_counted_down(batch)
        K.set_value(self.batch_timestamps, ts)

    def on_train_batch_begin(self, batch, logs=None):
        if self.trainining_generator:
            self._get_batch_timestamps_counted_down(batch, self.trainining_generator)
        if self.validation_generator:
            self._get_batch_timestamps_counted_down(batch, self.validation_generator)

    def on_test_batch_begin(self, batch, logs=None):
        if self.testing_generator:
            self._get_batch_timestamps_counted_down(batch, self.testing_generator)
            


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
            # RGB will always be last!
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
    
    def _get_batch_timestamps(self, index):
        indices = list(range(index * self.pos_in_batch_n, (index + 1) * self.pos_in_batch_n))
        timestamps = []
        for idx in indices:
            samples = [self.pos_samples[idx], self.neg_samples[idx]]
            for sample in samples:
                timestamps.append(sample['tpa_avg_timestamps'])
        # Y [B,2] > [B, F, 2]
        timestamps = np.array(timestamps)
        return timestamps

    def _get_batch_timestamps_counted_down(self, index):
        x = self._get_batch_timestamps(index)
        ts_cd = x-np.expand_dims(np.amax(x, axis=-1), axis=-1)
        return -ts_cd


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
        if not all([len(view) == len(Y) for view in x]):
            msg = "Number of samples inconsistent between views and labels, "
            msg += str([len(view) for view in x])
            raise ValueError(msg)
        #if self.RGB:
        #    Y = [Y, Y]
        return x, Y
        
    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.pos_samples)
            random.shuffle(self.neg_samples)