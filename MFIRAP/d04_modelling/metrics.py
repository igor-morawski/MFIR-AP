import numpy as np
import glob
import os
import functools
import tensorflow as tf
import random
import MFIRAP.d00_utils.project as project
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.keras.metrics import Mean

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

    def get_action_prediction_auc(self):
        return tf.keras.metrics.AUC()

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

def reduce_y(func):
    def wrapper(*args, **kwargs):
        y_true, y_pred = args
        mod_y_true, mod_y_pred = tf.math.reduce_max(y_true, axis=1), tf.math.reduce_max(y_pred, axis=1)
        return func(mod_y_true, mod_y_pred, **kwargs)
    return wrapper
    
class AUC_AP(tf.keras.metrics.AUC):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_state = reduce_y(self.update_state)

class Precision_AP(tf.keras.metrics.Precision):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_state = reduce_y(self.update_state)

class Recall_AP(tf.keras.metrics.Recall):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_state = reduce_y(self.update_state)

class PrecisionAtRecall_AP(tf.keras.metrics.Recall):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_state = reduce_y(self.update_state)

