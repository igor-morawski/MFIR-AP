"""
Protocol cross-subject training and testing.
"""
import inspect
import copy
import os
import glob
from copy import deepcopy
import HTPA32x32d

from MFIRAP.d04_modelling.models import Models_Training
from MFIRAP.d04_modelling.models import Baseline1, Downsampled16, Downsampled8

PROTOCOL_FIXED = 0
PROTOCOL_CROSS_SUBJ = 1
PROTOCOLS = [PROTOCOL_FIXED, PROTOCOL_CROSS_SUBJ]
PROTOCOL_DICT = {PROTOCOL_FIXED:"FIXED_SPLIT", PROTOCOL_CROSS_SUBJ:"CROSS_SUBJECT"}
"""
# old split 
DEVELOPMENT_SUBJ_L = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7", "subject8", "subject9"]
TESTING_SUBJ_L = ["subject10", "subject11", "subject12"]
"""
DEVELOPMENT_SUBJ_L = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7", "subject8"]
TESTING_SUBJ_L = ["subject9", "subject10", "subject11", "subject12"]
def _return_if_obj_is_type(obj, type_expected):
    if not isinstance(obj, type_expected):
        raise TypeError("Expected {}, got {}.".format(
            type_expected, type(obj)))
    else:
        return obj


def _assert_obj_is_type(obj, type_expected):
    if not isinstance(obj, type_expected):
        raise TypeError("Expected {}, got {}.".format(
            type_expected, type(obj)))
    else:
        return True


class Model:
    def __init__(self, name, frames, frame_shift, loss_function, train_set_ratio, batch_size, view_IDs, epochs, architecture, affine_transform=None):
        self.name = _return_if_obj_is_type(name, str)
        self.frames = _return_if_obj_is_type(frames, int)
        self.frame_shift = _return_if_obj_is_type(frame_shift, int)
        self.loss_function = _return_if_obj_is_type(loss_function, str)
        self.train_set_ratio = _return_if_obj_is_type(
            float(train_set_ratio), float)
        if not (0 <= self.train_set_ratio <= 1.0):
            raise ValueError
        self.batch_size = _return_if_obj_is_type(batch_size, int)
        self.view_IDs = _return_if_obj_is_type(view_IDs, list)
        self.epochs = _return_if_obj_is_type(epochs, int)
        for e in [self.batch_size, self.epochs]:
            if not (e > 0):
                raise ValueError
        self.architecture = _return_if_obj_is_type(
            architecture, type(type))
        if affine_transform: 
            self.affine_transform = _return_if_obj_is_type(affine_transform, bool)
        else: 
            self.affine_transform = False
        

    def dict(self):
        result = {}
        result["name"] = self.name
        result["frames"] = self.frames
        result["frame_shift"] = self.frame_shift
        result["loss_function"] = self.loss_function
        result["train_set_ratio"] = self.train_set_ratio
        result["batch_size"] = self.batch_size
        result["view_IDs"] = self.view_IDs
        result["epochs"] = self.epochs
        result["architecture"] = self.architecture
        result["affine_transform"] = self.affine_transform
        return result


class Ablation:
    def __init__(self, name, protocol, description="", models=[]):
        self.name = _return_if_obj_is_type(name, str)
        if len(self.name.split(" ")) > 1:
            raise ValueError("Only names that can be directory names are allowed!")
        self.protocol = _return_if_obj_is_type(protocol, int)
        if not (protocol <= len(PROTOCOLS)):
            raise ValueError
        self.description = _return_if_obj_is_type(description, str)
        self.models = _return_if_obj_is_type(models, list)
        [_assert_obj_is_type(m, Model) for m in self.models]

    def dict(self):
        result = {}
        result["name"] = self.name
        result["protocol"] = self.protocol
        result["description"] = self.description
        result["models"] = self.models
        return result


class Dataset_Manager:
    def __init__(self, dataset_path):
        """
        dataset_dir
            subjectname1
                0
                1
            ...
            subjectname2
                0
                1
            ...
            subjectnamek
                0
                1
            ...
        """
        self.dataset_path = _return_if_obj_is_type(dataset_path, str)
        self._parse_dirs()

    def _parse_dirs(self):
        # get all the subjects
        _unfiltered_subj_l = [p if os.path.isdir(p) else None for p in glob.glob(os.path.join(self.dataset_path, "*"))]
        subj_l = [os.path.relpath(p, self.dataset_path) for p in list(filter(None, _unfiltered_subj_l))]
        subj_l = list((set(DEVELOPMENT_SUBJ_L) | set(TESTING_SUBJ_L)) & set(subj_l))
        self.subjects_relpaths = subj_l.copy()
        self.subjects_relpaths.sort()
        self.subjects = self.subjects_relpaths.copy()
        self.subjects.sort()
        self.dataset_development_subj = list(set(DEVELOPMENT_SUBJ_L) & set(self.subjects))
        self.dataset_development_subj.sort()
        self.dataset_testing_subj = list(set(TESTING_SUBJ_L) & set(self.subjects))
        self.dataset_testing_subj.sort()
        # test if balanced
        pos_per_subj_l = [glob.glob(os.path.join(self.dataset_path, s, "1", "*")) for s in self.subjects_relpaths]
        neg_per_subj_l = [glob.glob(os.path.join(self.dataset_path, s, "0", "*")) for s in self.subjects_relpaths]
        assert len(pos_per_subj_l) == len(neg_per_subj_l)
        pos_per_subj_l = [glob.glob(os.path.join(self.dataset_path, s, "1", "*.TXT")) for s in self.subjects_relpaths]
        neg_per_subj_l = [glob.glob(os.path.join(self.dataset_path, s, "0", "*.TXT")) for s in self.subjects_relpaths]
        assert len(pos_per_subj_l) == len(neg_per_subj_l)
        return True
    
        


class Experiment_Setup:
    def __init__(self, ablations, dataset_path, models=None):
        if not models:
            models = []
        [_assert_obj_is_type(m, Model) for m in models]
        [_assert_obj_is_type(a, Ablation) for a in ablations]
        self.ablations = []
        self.models = []
        self.model_protocol_pairs = []
        for m in models:
            self.add_model(m)
        for a in ablations:
            self.add_ablation(a)
        for a in ablations:
            self.add_model_protocol_pairs(a)
        self.dataset_path = _return_if_obj_is_type(dataset_path, str)
        self.dataset_manager = Dataset_Manager(self.dataset_path)

    def add_model(self, model):
        if model not in self.models:
            self.models.append(_return_if_obj_is_type(model, Model))

    def add_ablation(self, ablation):
        if ablation not in self.ablations:
            self.ablations.append(_return_if_obj_is_type(ablation, Ablation))
            for m in ablation.models:
                self.add_model(m)  
            self.add_model_protocol_pairs(ablation)

    def add_model_protocol_pairs(self, ablation):
        models, protocol = ablation.models, ablation.protocol
        for model in models:
            pair = (model, protocol)
            if pair not in self.model_protocol_pairs:
                self.model_protocol_pairs.append(pair)


    def get_summary(self):
        def _a_details(obj):
            result = ""
            result += "\n\tDescription: {}.".format(obj.description)
            result += "\n\tProtocol: {}.".format(PROTOCOL_DICT[obj.protocol])
            result += "\n\t{} Models: {}.".format(len(obj.models), ", ".join(m.dict()["name"] for m in obj.models))
            return result

        ablations_str = "\n".join(["Ablation {}: {} {}".format(idx, a.name, _a_details(a))
                                   for idx, a in enumerate(self.ablations)])

        result = """
Experiment setup summary:\n
Dataset path: {dataset_path}
Dataset subjects: {dataset_subj}
Protocol fixed split:
    DEVELOP: {dataset_development_subj}
    TESTING: {dataset_testing_subj}
Ablations:
{ablations_str}
""".format(ablations_str=ablations_str, dataset_path=self.dataset_manager.dataset_path, dataset_subj = self.dataset_manager.subjects, dataset_development_subj=self.dataset_manager.dataset_development_subj, dataset_testing_subj=self.dataset_manager.dataset_testing_subj)
        return result

    def print_summary(self):
        print(self.get_summary())

    def compile(self):
        for ablation in self.ablations:
            if not len(ablation.models):
                raise ValueError(
                    "No models in ablation {}.".format(ablation.name))

    def dict(self):
        result = {}
        result["ablations"] = self.ablations
        result["models"] = self.models
        return result


def configure_experiments(dataset_path : str):
    """
    Usage:

    Models >> Ablations >> Experiment Setup >> Setup Compilation
    """

    models_args = inspect.getargspec(Model).args
    models_args_dict = dict.fromkeys(models_args)
    models_args_dict.pop('self', None)
    ablation_args = inspect.getargspec(Ablation).args
    ablation_args_dict = dict.fromkeys(ablation_args)
    ablation_args_dict.pop('self', None)

    # 1A Model templates
    template_default_training_parameters = deepcopy(models_args_dict)
    template_default_training_parameters["batch_size"] = 16
    template_default_training_parameters["epochs"] = 15
    template_default_training_parameters["train_set_ratio"] = 0.8
    template_default_training_parameters["architecture"] = Baseline1
    template_default_training_parameters["loss_function"] = "exponential_loss"

    template_frame_abl = deepcopy(template_default_training_parameters)
    template_frame_abl["view_IDs"] = ["121", "122", "123"]

    # 1B Models
    # i
    modelA_dict = deepcopy(template_frame_abl)
    modelA_dict["name"] = "modelA"
    modelA_dict["frames"] = 50
    modelA_dict["frame_shift"] = 0
    modelB_dict = deepcopy(template_frame_abl)
    modelB_dict["name"] = "modelB"
    modelB_dict["frames"] = 70
    modelB_dict["frame_shift"] = 20
    modelC_dict = deepcopy(template_frame_abl)
    modelC_dict["name"] = "modelC"
    modelC_dict["frames"] = 100
    modelC_dict["frame_shift"] = 0
    modelD_dict = deepcopy(template_frame_abl)
    modelD_dict["name"] = "modelD"
    modelD_dict["frames"] = 120
    modelD_dict["frame_shift"] = 20
    modelA = Model(**modelA_dict)
    modelB = Model(**modelB_dict)
    modelC = Model(**modelC_dict)
    modelD = Model(**modelD_dict)
    modelsABCD = [modelA, modelB, modelC, modelD]
    # ii
    template_view_abl = deepcopy(modelC_dict)  # just like C
    model_rlc = modelC  # alias for D
    model_c_dict = deepcopy(template_view_abl)
    model_c_dict["name"] = "model_c"
    model_c_dict["view_IDs"] = ["121"]
    model_r_dict = deepcopy(template_view_abl)
    model_r_dict["name"] = "model_r"
    model_r_dict["view_IDs"] = ["122"]
    model_cr_dict = deepcopy(template_view_abl)
    model_cr_dict["name"] = "model_cr"
    model_cr_dict["view_IDs"] = ["121", "122"]
    model_rl_dict = deepcopy(template_view_abl)
    model_rl_dict["name"] = "model_rl"
    model_rl_dict["view_IDs"] = ["122", "123"]
    model_c = Model(**model_c_dict)
    model_r = Model(**model_r_dict)
    model_cr = Model(**model_cr_dict)
    model_rl = Model(**model_rl_dict)
    modelsVIEWS = [model_c, model_r, model_cr, model_rl, model_rlc]

    # iii Cross-subject study
    modelC_cross_subj_dict = deepcopy(modelC_dict)  # just like C
    modelC_cross_subj_dict["name"] = "modelC_CS"
    modelC_cross_subj = Model(**modelC_cross_subj_dict)
    modelsFINAL = [modelC_cross_subj]

    # iv Downsampling
    modelC_16_dict = deepcopy(modelC_dict)  # just like C
    modelC_8_dict = deepcopy(modelC_dict)  # just like C
    modelC_16_dict["architecture"] = Downsampled16
    modelC_8_dict["architecture"] = Downsampled8
    modelC_16_dict["name"] = "modelC_16"
    modelC_8_dict["name"] = "modelC_8"
    modelC_16 = Model(**modelC_16_dict)
    modelC_8 = Model(**modelC_8_dict)
    modelsDOWNSAMPLE = [modelC_16, modelC_8]



    # 2. Ablations
    ablation1 = Ablation(name="Frame", protocol=PROTOCOL_FIXED,
                         description="Frame and frame_shift ablation", models=modelsABCD)
    ablation2 = Ablation(name="View", protocol=PROTOCOL_FIXED,
                         description="View ablation", models=modelsVIEWS)
    ablation3 = Ablation(name="Final", protocol=PROTOCOL_CROSS_SUBJ,
                         description="Final results on cross-subject", models=modelsFINAL)
    ablationD = Ablation(name="Downsampling", protocol=PROTOCOL_FIXED,
                         description="Input resolution study", models=modelsDOWNSAMPLE)


    # 3. Experiment setup
    ablations = [ablation1, ablation2, ablation3, ablationD]
    modelA.epochs = 5
    modelA.train_set_ratio = 1.0
    modelA.loss_function = "early_exponential_loss"
    ablations = [Ablation(name="Frame", protocol=PROTOCOL_FIXED,
                         description="Frame and frame_shift ablation", models=[modelA])]
    experiment_setup = Experiment_Setup(ablations=ablations, dataset_path=dataset_path)

    # 4. Return
    return experiment_setup


if __name__ == "__main__":
    dataset_path = "/media/igor/DATA/D01_MFIR-AP-Dataset"
    result = configure_experiments(dataset_path)
    result.print_summary()
