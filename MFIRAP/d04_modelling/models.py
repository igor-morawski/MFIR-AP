import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, TimeDistributed, Activation, Lambda
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.layers import Concatenate, Subtract, Multiply
from tensorflow.keras.layers import GRU as RNN
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import glob
import os
import json
import pickle
import shutil

import MFIRAP.d00_utils.verbosity as vb
import MFIRAP.d00_utils.paths as paths

import MFIRAP.d00_utils.project as project


RGB_FEATURES_LAYER_NAME = "RGB_features"

FLOATX = 'float32'
TPA_DENSE_DEFAULT = 1024 // 10
tf.keras.backend.set_image_data_format('channels_last')

def _build_TPA_embedding(view_id, dense_units, block1 = None):
    # VGG-16 but dims are scaled by 1/7, only 3 blocks
    # FUTURE Think about filters -> skipping cncnts
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    # b=block c=conv m=maxpool
    # input>b1c1>b1c2>b1c3>b2m1>b2c1>b2c2>b2c3>b3m1>flatten>fc
    embedding_input = Input(
        shape=(None, *project.TPA_FRAME_SHAPE), name='TPA{}_input'.format(view_id))
    # block1
    if not block1:
        b1c1 = TimeDistributed(Conv2D(filters=64, kernel_size=(
            3, 3), padding="same", activation="relu"), name='TPA{}_b1c1'.format(view_id))(embedding_input)
        b1c2 = TimeDistributed(Conv2D(filters=64, kernel_size=(
            3, 3), padding="same", activation="relu"), name='TPA{}_b1c2'.format(view_id))(b1c1)
        b1c3 = TimeDistributed(Conv2D(filters=64, kernel_size=(
            3, 3), padding="same", activation="relu"), name='TPA{}_b1c3'.format(view_id))(b1c2)
    else:
        b1c3 = block1(embedding_input)
    # block2
    b2m1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(
        2, 2)), name='TPA{}_b2m1'.format(view_id))(b1c3)
    b2c1 = TimeDistributed(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"), name='TPA{}_b2c1'.format(view_id))(b2m1)
    b2c2 = TimeDistributed(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"), name='TPA{}_b2c2'.format(view_id))(b2c1)
    b2c3 = TimeDistributed(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"), name='TPA{}_b2c3'.format(view_id))(b2c2)
    # block3
    b3m1 = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(
        2, 2)), name='TPA{}_b3m1'.format(view_id))(b2c3)
    # FC
    flat = TimeDistributed(Flatten(), name='TPA{}_flat'.format(view_id))(b3m1)
    dense = TimeDistributed(Dense(units=dense_units, activation="relu"),
                            name='TPA{}_dense'.format(view_id))(flat)  # flatten/fc = 6.125
    embedding_output = dense
    return embedding_input, embedding_output


def convert_to_stateful(original_model):
    original_model_json = original_model.to_json()
    inference_model_dict = json.loads(original_model_json)

    layers = inference_model_dict['config']['layers']
    for layer in layers:
        if 'stateful' in layer['config']:
            layer['config']['stateful'] = True

        if 'batch_input_shape' in layer['config']:
            layer['config']['batch_input_shape'][0] = 1
            layer['config']['batch_input_shape'][1] = 1

    inference_model = model_from_json(json.dumps(inference_model_dict))
    inference_model.set_weights(original_model.get_weights())
    return inference_model


class Model_Evaluation():
    def __init__(self, model_path, fold_name=None, stateful=False, weights_ext = "h5", load_scaling=True):
        if fold_name:
            self.name = fold_name
        else:
            self.name = os.path.split(model_path)[-1]
        json_fp = os.path.join(model_path, os.path.split(model_path)[-1]+".json")
        weights_fp = os.path.join(model_path, self.name+"."+weights_ext)
        with open(json_fp, 'r') as json_file:
            trained_architecture = tf.keras.models.model_from_json(
                json_file.read())
        if stateful:
            model = convert_to_stateful(trained_architecture)
        else:
            model = trained_architecture
        model.load_weights(weights_fp)
        self.model = model
        if load_scaling:
            with open(os.path.join(model_path, "scaling.pkl"), "rb") as f:
                self.scaling = pickle.load(f)
        else:
            self.scaling = None



class Models_Training():
    def __init__(self, name, model, fit_kwargs, compile_kwargs, TPA_view_IDs, pretraining=False, precompile_kwargs=None, prefit_kwargs=None):
        self.model = model
        self.fit_kwargs = fit_kwargs
        self.compile_kwargs = compile_kwargs
        self.TPA_view_IDs = TPA_view_IDs
        self.name = name
        self.data_models_model_path = os.path.join(
            project.DATA_MODELS_PATH, self.name)
        self.data_models_output_model_path = os.path.join(
            project.DATA_MODELS_OUTPUT_PATH, self.name)
        self.data_model_plot_path = os.path.join(
            project.DATA_MODELS_OUTPUT_PATH, self.name, "plots")
        paths.ensure_path_exists(self.data_models_model_path)
        paths.ensure_path_exists(self.data_models_output_model_path)
        paths.ensure_path_exists(self.data_model_plot_path)
        self.pretraining = pretraining
        self.prefit_kwargs = prefit_kwargs
        self.precompile_kwargs = precompile_kwargs
        self.pretrained = False
        self.trained = False
        self.saved = False
        if self.pretraining:
            if not all([self.prefit_kwargs, self.precompile_kwargs]):
                raise Exception(
                    "Pretraining requested but no args for compiling and/or fitting passed.")

    def train(self):
        if self.pretraining:
            vb.print_general("Pretraining...")
            self.model.compile(**self.precompile_kwargs)
            self.model.fit(**self.prefit_kwargs)
            self.pretrained = True
        if self.pretraining and not self.pretrained:
            raise Exception("Pretrain the model first!")
        if self.pretrained:
            print("\n\n\n\n\n\n")
        vb.print_general("Training...")
        self.model.compile(**self.compile_kwargs)
        self.model.fit(**self.fit_kwargs)
        self.trained = True

    def write_architecture(self):
        if not self.trained:
            raise Exception("Train the model first!")
        tf.keras.utils.plot_model(self.model, os.path.join(
            self.data_models_model_path, self.name+".png"))
        model_json = self.model.to_json()
        with open(os.path.join(self.data_models_model_path, self.name+".json"), "w") as json_file:
            json_file.write(model_json)
    

    def save(self):
        if not self.trained:
            raise Exception("Train the model first!")
        model_json = self.model.to_json()
        with open(os.path.join(self.data_models_model_path, self.name+".json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(
            self.data_models_model_path, self.name+".h5"))
        tf.keras.utils.plot_model(self.model, os.path.join(
            self.data_models_model_path, self.name+".png"))
        shutil.copy2(os.path.join("settings",self.name+".json"), os.path.join(self.data_models_model_path, "settings.json"))
        # Save report data
        '''
        report_data = dict()
        report_data["TPA_view_IDs"] = self.TPA_view_IDs
        report_data["Epochs"] = self.ep
        with open(os.path.join(self.data_models_model_path, self.name+"_reporting.json"), "w") as json_file:
            json.dump(report_data, json_file)
        '''
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
        dirs = [self.data_models_model_path,
                self.data_models_output_model_path, self.data_model_plot_path]
        [[os.remove(f) for f in fl] for fl in [glob.glob(p)
                                               for p in [os.path.join(d, "*.*") for d in dirs]]]

    def delete_existing_plots(self):
        '''
        Use carefully
        Clears all plots associated with model's name in self.name
        '''
        dirs = [self.data_model_plot_path]
        [[os.remove(f) for f in fl] for fl in [glob.glob(p)
                                               for p in [os.path.join(d, "*.png") for d in dirs]]]


class Baseline1(Models_Training):
    '''
    TimeDistributed(classification(rnn(view_pooling(TPA x 3)))
    '''

    def __init__(self, fit_kwargs, compile_kwargs, name, TPA_view_IDs, TPA_dense_units=TPA_DENSE_DEFAULT, **kwargs):
        vb.print_general("Initializing {} - Baseline1...".format(name))
        vb.print_general("Model will be trained on {} views".format(TPA_view_IDs))

        if "pretraining" in kwargs.keys():
            assert not kwargs["pretraining"]

        io_TPAs = [_build_TPA_embedding(id, TPA_dense_units)
                   for id in TPA_view_IDs]
        i_TPAs = [x[0] for x in io_TPAs]
        o_TPAs = [x[1] for x in io_TPAs]

        if len(TPA_view_IDs) > 1:
            TPA_merged = Concatenate(name='view_concat', axis=-1)([*o_TPAs])
        if len(TPA_view_IDs) == 1:
            TPA_merged = [*o_TPAs]

        rnn = RNN(TPA_dense_units*len(io_TPAs), activation='tanh',
                  recurrent_activation='sigmoid', return_sequences=True, name="TPA_GRU")(TPA_merged)
        TPA_dense = TimeDistributed(
            Dense(project.N_CLASSES, activation=None), name="TPA_dense")(rnn)
        TPA_classification = Activation(
            activation='softmax', name='TPA_classification')(TPA_dense)
        model = Model(i_TPAs, TPA_classification,
                      name="Model_{}xTPA".format(len(TPA_view_IDs)))

        Models_Training.__init__(self, name=name, model=model, fit_kwargs=fit_kwargs,
                                 compile_kwargs=compile_kwargs, TPA_view_IDs=TPA_view_IDs)



class Downsampled16(Models_Training):
    '''
    TimeDistributed(classification(rnn(view_pooling(TPA x 3)))
    '''

    def __init__(self, fit_kwargs, compile_kwargs, name, TPA_view_IDs, TPA_dense_units=TPA_DENSE_DEFAULT, **kwargs):
        vb.print_general("Initializing {} - Baseline1...".format(name))
        vb.print_general("Model will be trained on {} views".format(TPA_view_IDs))

        def downsampling(tpa_view_id):
            return TimeDistributed(AvgPool2D(pool_size=(2, 2), strides=None), name='TPA{}_'.format(tpa_view_id))
        io_TPAs = [_build_TPA_embedding(id, TPA_dense_units, block1=downsampling(id))
                   for id in TPA_view_IDs]
        i_TPAs = [x[0] for x in io_TPAs]
        o_TPAs = [x[1] for x in io_TPAs]

        if len(TPA_view_IDs) > 1:
            TPA_merged = Concatenate(name='view_concat', axis=-1)([*o_TPAs])
        if len(TPA_view_IDs) == 1:
            TPA_merged = [*o_TPAs]

        rnn = RNN(TPA_dense_units*len(io_TPAs), activation='tanh',
                  recurrent_activation='sigmoid', return_sequences=True, name="TPA_GRU")(TPA_merged)
        TPA_dense = TimeDistributed(
            Dense(project.N_CLASSES, activation=None), name="TPA_dense")(rnn)
        TPA_classification = Activation(
            activation='softmax', name='TPA_classification')(TPA_dense)
        model = Model(i_TPAs, TPA_classification,
                      name="Model_{}xTPA".format(len(TPA_view_IDs)))

        Models_Training.__init__(self, name=name, model=model, fit_kwargs=fit_kwargs,
                                 compile_kwargs=compile_kwargs, TPA_view_IDs=TPA_view_IDs)


class Downsampled8(Models_Training):
    '''
    TimeDistributed(classification(rnn(view_pooling(TPA x 3)))
    '''

    def __init__(self, fit_kwargs, compile_kwargs, name, TPA_view_IDs, TPA_dense_units=TPA_DENSE_DEFAULT, **kwargs):
        vb.print_general("Initializing {} - Baseline1...".format(name))
        vb.print_general("Model will be trained on {} views".format(TPA_view_IDs))

        def downsampling(tpa_view_id):
            return TimeDistributed(AvgPool2D(pool_size=(4, 4), strides=None), name='TPA{}_'.format(tpa_view_id))
        io_TPAs = [_build_TPA_embedding(id, TPA_dense_units, block1=downsampling(id))
                   for id in TPA_view_IDs]
        i_TPAs = [x[0] for x in io_TPAs]
        o_TPAs = [x[1] for x in io_TPAs]

        if len(TPA_view_IDs) > 1:
            TPA_merged = Concatenate(name='view_concat', axis=-1)([*o_TPAs])
        if len(TPA_view_IDs) == 1:
            TPA_merged = [*o_TPAs]

        rnn = RNN(TPA_dense_units*len(io_TPAs), activation='tanh',
                  recurrent_activation='sigmoid', return_sequences=True, name="TPA_GRU")(TPA_merged)
        TPA_dense = TimeDistributed(
            Dense(project.N_CLASSES, activation=None), name="TPA_dense")(rnn)
        TPA_classification = Activation(
            activation='softmax', name='TPA_classification')(TPA_dense)
        model = Model(i_TPAs, TPA_classification,
                      name="Model_{}xTPA".format(len(TPA_view_IDs)))

        Models_Training.__init__(self, name=name, model=model, fit_kwargs=fit_kwargs,
                                 compile_kwargs=compile_kwargs, TPA_view_IDs=TPA_view_IDs)

SETUP_DIC = {"baseline1": Baseline1, Baseline1:Baseline1, Downsampled16:Downsampled16, Downsampled8:Downsampled8}
SETUP_RGB_FLAGS = {"baseline1": False}
