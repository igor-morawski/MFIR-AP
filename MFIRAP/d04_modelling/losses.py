import numpy as np
import glob
import os
import tensorflow as tf
import random
import MFIRAP.d00_utils.project as project

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