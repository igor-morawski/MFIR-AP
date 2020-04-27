import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, TimeDistributed, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import GRU as RNN


EMBEDDING_UNITS = 1024
RNN_UNITS = EMBEDDING_UNITS
N_CLASSES = 1

KERAS_EPSILON = tf.keras.backend.epsilon()
EPSILON = KERAS_EPSILON

def build_embedding():
    # sequential
    # VGG-16 but dims are scaled by 1/7, only 3 blocks
    # FUTURE Think about filters -> skipping cncnts
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    # b=block c=conv m=maxpool
    # input>b1c1>b1c2>b1c3>b2m1>b2c1>b2c2>b2c3>b3m3>flatten>fc
    model = Sequential()
    # block1
    model.add(Conv2D(filters=64, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                  padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                  padding="same", activation="relu"))
    # block2
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                  padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                  padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                  padding="same", activation="relu"))
    # block3
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # FC
    model.add(Flatten())
    model.add(Dense(units=EMBEDDING_UNITS, activation="relu"))# flatten/fc = 6.125
    return model


def build_RNN():
    # sequential
    model = Sequential()
    model.add(RNN(RNN_UNITS, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    return model


def build_classifier_logits():
    # functional
    # embedding>rnn>softmax
    model = Sequential()
    embedding = build_embedding()
    model.add(TimeDistributed(embedding))
    rnn = build_RNN()
    model.add(rnn)
    model.add(TimeDistributed(Dense(N_CLASSES, activation=None)))
    return model

def build_classifier():
    # build_classifier_logits() followed by sigmoid activation
    model = Sequential()
    model.add(build_classifier_logits())
    model.add(Activation(activation='sigmoid'))
    return model

def exponential_loss(y_true, y_pred, from_logits=False):
    # [B, F, 2], [B, F, 2]
    # TODO ADD JAIN
    # L_p = sigma_t(-exp())
    # L_n = softmax_cross_entropy
    # EL = L_p + L_n
    # https://stackoverflow.com/questions/39192380/tensorflow-one-class-classification
    # https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/backend/cntk_backend.py#L1065
    if not tf.is_tensor(y_pred):
        y_pred = tf.constant(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    if from_logits:
        y_pred = tf.keras.activations.sigmoid(y_pred)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    log_pos = (y_true)*tf.math.log(y_pred)
    log_neg = (1.0 - y_true)*tf.math.log(1.0 - y_pred)
    # ONLY IF ACTION HAPPENS AT THE LAST FRAME
    Y = y_true.shape[-1]
    positive_loss = -tf.reduce_sum(tf.broadcast_to(tf.math.exp(-(Y-tf.range(Y)-1)/20), log_pos.shape) * log_pos)
    negative_loss = -tf.reduce_sum(log_neg)
    total_loss = positive_loss + negative_loss
    return total_loss
