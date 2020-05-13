import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, TimeDistributed, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GRU as RNN
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import glob
import os

import MFIRAP.d00_utils.verbosity as vb
import MFIRAP.d00_utils.paths as paths

import MFIRAP.d00_utils.project as project

FLOATX='float32'
tf.keras.backend.set_image_data_format('channels_last')

import numpy as np

def _build_TPA_embedding(view_id, dense_units):
    # VGG-16 but dims are scaled by 1/7, only 3 blocks
    # FUTURE Think about filters -> skipping cncnts
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    # b=block c=conv m=maxpool
    # input>b1c1>b1c2>b1c3>b2m1>b2c1>b2c2>b2c3>b3m1>flatten>fc
    embedding_input = Input(shape=(None, *project.TPA_FRAME_SHAPE), name='TPA{}_input'.format(view_id))
    # block1
    b1c1 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b1c1'.format(view_id))(embedding_input)
    b1c2 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b1c2'.format(view_id))(b1c1)
    b1c3 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b1c3'.format(view_id))(b1c2)
    # block2
    b2m1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)), name='TPA{}_b2m1'.format(view_id))(b1c3)
    b2c1 = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b2c1'.format(view_id))(b2m1)
    b2c2 = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b2c2'.format(view_id))(b2c1)
    b2c3 = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"), name='TPA{}_b2c3'.format(view_id))(b2c2)
    # block3
    b3m1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)), name='TPA{}_b3m1'.format(view_id))(b2c3)
    # FC
    flat = TimeDistributed(Flatten(), name='TPA{}_flat'.format(view_id))(b3m1)
    dense = TimeDistributed(Dense(units=dense_units, activation="relu"), name='TPA{}_dense'.format(view_id))(flat) # flatten/fc = 6.125
    embedding_output = dense
    return embedding_input, embedding_output

class Models_Training():
    def __init__(self, name, model, fit_kwargs, compile_kwargs, TPA_view_IDs):
        self.model = model
        self.fit_kwargs = fit_kwargs
        self.compile_kwargs = compile_kwargs
        self.TPA_view_IDs = TPA_view_IDs
        self.name=name
        self.data_models_model_path = os.path.join(project.DATA_MODELS_PATH, self.name)
        self.data_models_output_model_path = os.path.join(project.DATA_MODELS_OUTPUT_PATH, self.name)
        self.data_model_plot_path = os.path.join(project.DATA_MODELS_OUTPUT_PATH, self.name, "plots")
        paths.ensure_path_exists(self.data_models_model_path)
        paths.ensure_path_exists(self.data_models_output_model_path)
        paths.ensure_path_exists(self.data_model_plot_path)
        self.trained = False
        self.saved = False

    def compile(self):
        self.model.compile(**self.compile_kwargs)
    
    def fit(self):
        self.model.fit(**self.fit_kwargs)
        self.trained = True

    def save(self):
        if not self.trained:
            raise Exception("Train the model first!")
        model_json = self.model.to_json()
        with open(os.path.join(self.data_models_model_path, self.name+".json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(self.data_models_model_path, self.name+".h5"))
        tf.keras.utils.plot_model(self.model, os.path.join(self.data_models_model_path, self.name+".png"))
        self.saved = True

    @property
    def history(self):
        return self.model.history.history
    
    def plot_metrics(self, plot_val_metrics=False):
        if not self.trained:
            raise Exception("Train the model first!")
        for metric in self.history:
            if not plot_val_metrics:
                if "val_" in metric:
                    continue
            self._plot(self.history[metric], metric, "epoch")
            plt.savefig(os.path.join(self.data_model_plot_path, metric+".png"))
        return

    def _plot(self, x, x_label, y_label, title=None):
        if not title:
            title = 'Model {}'.format(x_label)
        plt.clf()
        plt.plot(x)
        plt.title(title)
        plt.ylabel(''.format(x_label))
        plt.xlabel(''.format(y_label))

    def delete_existing_model_data_and_output(self):
        '''
        Use carefully
        Clears model files including reporting files such as plots associated with model's name in self.name
        '''
        dirs = [self.data_models_model_path, self.data_models_output_model_path, self.data_model_plot_path]
        [[os.remove(f) for f in fl] for fl in [glob.glob(p) for p in [os.path.join(d, "*.*") for d in dirs]]]
    
    def delete_existing_plots(self):
        '''
        Use carefully
        Clears all plots associated with model's name in self.name
        '''
        dirs = [self.data_model_plot_path]
        [[os.remove(f) for f in fl] for fl in [glob.glob(p) for p in [os.path.join(d, "*.png") for d in dirs]]]
        
class Baseline1(Models_Training):
    '''
    TimeDistributed(classification(rnn(view_pooling(TPA x 3)))
    '''
    def __init__(self, fit_kwargs, compile_kwargs, name, TPA_view_IDs, TPA_dense_units = 1024//3):
        vb.print_general("Initializing Baseline1...")

        io_TPAs = [_build_TPA_embedding(id, TPA_dense_units) for id in TPA_view_IDs]
        i_TPAs = [x[0] for x in io_TPAs]
        o_TPAs = [x[1] for x in io_TPAs]

        if len(TPA_view_IDs) > 1:
            TPA_merged = Concatenate(name='view_concat', axis=-1)([*o_TPAs])
        if len(TPA_view_IDs) == 1:
            TPA_merged = [*o_TPAs]
        
        rnn = RNN(TPA_dense_units*len(io_TPAs), activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name = "TPA_GRU")(TPA_merged)
        TPA_dense = TimeDistributed(Dense(project.N_CLASSES, activation=None), name="TPA_dense")(rnn)
        TPA_classification = Activation(activation='sigmoid', name='TPA_classification')(TPA_dense)
        model = Model(i_TPAs, TPA_classification, name="Model_3xTPA")

        Models_Training.__init__(self, name = name, model = model, fit_kwargs=fit_kwargs, compile_kwargs=compile_kwargs, TPA_view_IDs=TPA_view_IDs)






SETUP_DIC = {"baseline1":Baseline1}
SETUP_RGB_FLAGS = {"baseline1":False}


'''

class Models_Training():
    def __init__(self, name, model, fit_kwargs, compile_kwargs, TPA_view_IDs):
        self.model = model
        self.fit_kwargs = fit_kwargs
        self.compile_kwargs = compile_kwargs
        self.TPA_view_IDs = TPA_view_IDs
        self.name=name
        self.data_models_model_path = os.path.join(project.DATA_MODELS_PATH, self.name)
        self.data_models_output_model_path = os.path.join(project.DATA_MODELS_OUTPUT_PATH, self.name)
        paths.ensure_path_exists(self.data_models_model_path)
        paths.ensure_path_exists(self.data_models_output_model_path)
        self.trained = False
        self._history = None
        try: 
            self.verbose = fit_kwargs["verbose"]
        except: 
            self.verbose = 1

    def compile(self):
        self.model.compile(**self.compile_kwargs)
    
    def fit(self):
        fit_kwargs = self.fit_kwargs.copy()
        valid_kwargs = self.fit_kwargs.copy()
        try:
            fit_kwargs["validation_data"] = None
            valid_kwargs["x"] = valid_kwargs["validation_data"]
            valid_kwargs.pop("validation_data")
        except KeyError:
            fit_kwargs["validation_data"] = None
            valid_kwargs["x"] = None
            vb.print_general("No validation data.")
        try: 
            epochs = fit_kwargs["epochs"]
            fit_kwargs.pop("epochs")
            valid_kwargs.pop("epochs")
        except KeyError:
            epochs = 1
        for epoch in range(epochs):
            if self.verbose:
                print("Epoch {}/{}: training...".format(epoch, epochs))
            history = self.model.fit(**fit_kwargs)
            self._update_history(history)
            if valid_kwargs["x"]:
                if self.verbose:
                    print("Epoch {}/{}: validation...".format(epoch, epochs))
                valid_history = self.model.evaluate(**valid_kwargs)
                self._update_valid_history(valid_history)
        self.trained = True
        return
        self.model.fit(**self.fit_kwargs)
        self.trained = True

    @property
    def history(self):
        return self._history
        #return self.model.history.history

    def _update_history(self, history_step):
        if not self._history:
            self._history = history_step.history
        for key in history_step.history:
            self._history[key] = self._history[key] + history_step.history[key]

    def _update_valid_history(self, valid_history_step):
        # no if not self._history here, self._history is assumed to be always handled
        # by _update_history during trainiing first, if this doesn't hold anymore modify here
        for key, value in zip(self.model.metrics_names, valid_history_step):
            valid_key = "valid_"+key
            try:
                self._history[valid_key] = self._history[valid_key] + [value]
            except KeyError:
                self._history[valid_key] = [value]

    def plot_loss(self):
        if not self.trained:
            raise Exception("Train the model first!")
        self._plot(self.history["loss"], "loss", "epoch")
        plt.savefig(os.path.join(self.data_models_model_path, "loss.png"))

    def _plot(self, x, x_label, y_label, title=None):
        if not title:
            title = 'Model {}'.format(x_label)
        plt.clf
        plt.plot(x)
        plt.title(title)
        plt.ylabel(''.format(x_label))
        plt.xlabel(''.format(y_label))
        plt.clf
'''