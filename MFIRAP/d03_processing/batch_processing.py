import MFIRAP.d00_utils.paths
import MFIRAP.d00_utils.dataset
import MFIRAP.d00_utils.verbosity
import MFIRAP.d00_utils.io
from MFIRAP.d00_utils.verbosity import print_specific, print_general
from MFIRAP.d00_utils.paths import ensure_path_exists, ensure_parent_exists
from MFIRAP.d00_utils.dataset import get_pos_neg_fp_lists, filter_subjects_from_fp_list, samples_from_fp_list, get_TPA_mean_and_std
from MFIRAP.d00_utils.dataset import read_tpa123_from_npz, read_rgb_from_npz, standarize_TPA, standarize_RGB

from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model

import os
import numpy as np

def intermediate2processed(dataset_path, destination_parent_path, development_subjects, test_subjects, keras_model = None):
    '''
    keras_model : Keras model for feature extraction, if None ['array_IDRGB'] will be None], must be initialized
    '''
    ### BATCH MAKER START ###
    mu, sigma = _compute_devel_mu_sigma(dataset_path, destination_parent_path, development_subjects)
    _intermediate2processed_development(dataset_path, destination_parent_path, development_subjects, mu, sigma, keras_model)
    _intermediate2processed_test(dataset_path, destination_parent_path, test_subjects, mu, sigma, keras_model)
    ### BATCH MAKER END ###

def _intermediate2processed_development(dataset_path, destination_parent_path, subjects, mu, sigma, rgb_model):
    _batch_maker(dataset_path, destination_parent_path, subjects, "development", mu, sigma, rgb_model)

def _intermediate2processed_test(dataset_path, destination_parent_path, subjects, mu, sigma, rgb_model):
    _batch_maker(dataset_path, destination_parent_path, subjects, "test", mu, sigma, rgb_model)

def _compute_devel_mu_sigma(dataset_path, destination_parent_path, subjects):
    # 1. Inputs
    pos_fp_list, neg_fp_list = get_pos_neg_fp_lists(dataset_path)
    _, name = os.path.split(dataset_path)
    # 2. Leave only samples from the development subjects list!
    pos_fp_list = filter_subjects_from_fp_list(pos_fp_list, subjects)
    neg_fp_list = filter_subjects_from_fp_list(neg_fp_list, subjects)
    # 3. Load in npz
    pos_samples = samples_from_fp_list(pos_fp_list)
    neg_samples = samples_from_fp_list(neg_fp_list)
    # 4. Get mean and std for TPA normalization, RGB will be normalized per sequence.
    print_general('Getting mean and std of development TPA samples...')
    TPA_mean, TPA_std = get_TPA_mean_and_std(pos_samples + neg_samples)
    print_specific('TPA mean: {:.2f} \nTPA std: {:.2f}'.format(TPA_mean, TPA_std))
    return TPA_mean, TPA_std


def _batch_maker(dataset_path, destination_parent_path, subjects, devel_or_test, TPA_mean, TPA_std, rgb_model):
    '''
    devel_or_test : str
        'development' or 'test'
    '''
    assert (devel_or_test == 'test') or (devel_or_test == 'development')
    if devel_or_test == 'development':
        development = True
        test = False
    else:
        test = True
        development = False
    ### BATCH MAKER START ###
    # 1. Inputs
    pos_fp_list, neg_fp_list = get_pos_neg_fp_lists(dataset_path)
    _, name = os.path.split(dataset_path)
    batch_maker_dest = os.path.join(destination_parent_path, name, devel_or_test)
    ensure_path_exists(batch_maker_dest)
    # 2. Leave only samples from the development subjects list!
    pos_fp_list = filter_subjects_from_fp_list(pos_fp_list, subjects)
    neg_fp_list = filter_subjects_from_fp_list(neg_fp_list, subjects)
    # 3. Load in npz
    pos_samples = samples_from_fp_list(pos_fp_list)
    neg_samples = samples_from_fp_list(neg_fp_list)
    # 3A. Initialize feature extractor
    if rgb_model:
        rgb_shape = MFIRAP.d00_utils.io.read_json_key(os.path.join(dataset_path,'samples.nfo'), 'rgb_shape')
        input = Input(shape=(224,224,3),name = 'image_input')
        x = rgb_model(input)
        x = Flatten()(x)
        feature_extractor = Model(inputs=input, outputs=x)
    else: 
        feature_extractor = None
    # 4. Main loop
    print_general('Processing and saving {} samples...'.format(len(pos_fp_list + neg_fp_list)))
    for fp in pos_fp_list + neg_fp_list:
        print_specific("Reading {}".format(fp))
        sample = np.load(fp)
        head, tail = os.path.split(fp)
        tpa1, tpa2, tpa3 = read_tpa123_from_npz(sample)
        if rgb_model:
            #5A. Extract features per frame
            rgb_src = standarize_RGB(read_rgb_from_npz(sample))
            rgb = feature_extractor.predict(rgb_src)
        else:
            rgb = None
        #5B. TPA standarization according to pt. 4
        tpa1, tpa2, tpa3 = [standarize_TPA(t, TPA_mean, TPA_std) for t in [tpa1, tpa2, tpa3]]
        sample_keys = list(sample.keys())
        dict2save = {'array_ID121' : tpa1, 'array_ID122': tpa2, 'array_ID123' : tpa3, 'array_IDRGB' : rgb}
        for key in sample_keys: 
            if key not in dict2save.keys():
                dict2save[key] = sample[key]
        output = os.path.join(batch_maker_dest, str(int(np.argmax(sample['one_hot']))), tail)
        ensure_parent_exists(output)
        print_specific("Writing {}".format(output))
        np.savez_compressed(output, **dict2save)
    print_general('Processed and saved {} samples in {} set.'.format(len(pos_fp_list + neg_fp_list), devel_or_test))
    ### BATCH MAKER END ###