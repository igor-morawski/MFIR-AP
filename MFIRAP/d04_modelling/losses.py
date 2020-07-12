import numpy as np
import glob
import os
import tensorflow as tf
import random
import MFIRAP.d00_utils.project as project
from tensorflow.keras import backend as K

POS_LOSS_WEIGHT = 1.3

class Losses_Keras:
    def __init__(self, frames, frame_shift):
        self.frames = frames
        self.frame_shfit = frame_shift
    
    def get_by_name(self, name, **kwargs):
        if name == "exponential_loss":
            return self.get_exponential_loss(**kwargs)
        if name == "early_exponential_loss":
            return self.get_early_exponential_loss(**kwargs)
        else:
            raise ValueError("No such loss function!")

    def get_exponential_loss(self, from_logits=False):
        E = np.exp(-(self.frames-np.arange(self.frames)-1)/(self.frames/20))
        L = np.log(np.zeros(self.frames)+0.01)
        weight = (L*E).sum()/L.sum()

        def exponential_loss(y_true, y_pred, from_logits=from_logits):
            #XXX add weight
            #XXX normalize by no of frames
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
            k = tf.cast(self.frames/20, project.FLOATX)
            exp = tf.cast(tf.math.exp(-(Y-tf.range(Y)-1)/k), project.FLOATX)
            positive_loss = -tf.reduce_sum(tf.broadcast_to(exp, tf.shape(log_pos)) * log_pos)
            # standard cross-entropy loss
            negative_loss = -tf.reduce_sum(log_neg)
            total_loss = POS_LOSS_WEIGHT*positive_loss + weight*negative_loss
            return total_loss
        return exponential_loss


    def get_early_exponential_loss(self, from_logits=False):
        E = 1 - np.exp(np.arange(self.frames)/(self.frames))
        L = np.log(np.zeros(self.frames)+0.01)
        weight = (L*E).sum()/L.sum()

        def early_exponential_loss(y_true, y_pred, from_logits=from_logits):
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
            exp = tf.cast(1.0 - tf.math.exp(-(tf.range(Y)-1)/Y), project.FLOATX)
            positive_loss = -tf.reduce_sum(tf.broadcast_to(exp, tf.shape(log_pos)) * log_pos)
            # standard cross-entropy loss
            negative_loss = -tf.reduce_sum(log_neg)
            total_loss = POS_LOSS_WEIGHT*positive_loss + negative_loss
            return total_loss
        return early_exponential_loss

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)