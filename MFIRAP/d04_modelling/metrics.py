import numpy as np
import glob
import os
import tensorflow as tf
import random
import MFIRAP.d00_utils.project as project
from tensorflow.python.ops import math_ops
K = tf.keras.backend

#XXX FIX <<< WHAT IF FRAME_SHIFT?
#https://stackoverflow.com/jobs/368577/engineer-platform-mongodb-rds-auth0
class Metrics_Keras:
    def __init__(self, frames, frame_shift, timestamps_countdowned):
        print("[WARNING] Ignore ATTA for validation in model.fit().")
        print("It is computed using timestamps of training batch and therefore inaccurate.")
        self.frames = frames
        self.frame_shfit = frame_shift
        self.timestamps_countdowned = timestamps_countdowned

    def get_average_time_to_accident(self):
        axis, index = 2, 1
        def average_time_to_accident(y_true, y_pred):
            timestamps_countdowned = self.timestamps_countdowned
            if not tf.is_tensor(y_pred):
                y_pred = tf.constant(y_pred, dtype=project.FLOATX)
            y_true = tf.cast(y_true, y_pred.dtype)
            shape = K.int_shape(y_pred)
            selection = [slice(shape[a]) if a != axis else index for a in range(len(shape))]
            thresh = math_ops.cast(y_pred[selection] > 0.5, "bool")
            time_to_accident = tf.math.reduce_max(tf.where(thresh, timestamps_countdowned, [0]), axis=-1)
            pos_idx = tf.where(tf.math.reduce_max(y_true[selection], axis=-1))
            return tf.math.reduce_mean(tf.gather_nd(time_to_accident, pos_idx))
        return average_time_to_accident
            
            