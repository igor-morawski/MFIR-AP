import unittest
from MFIRAP import d00_utils
tested_module = d00_utils.dataset

import os

class Test_filter_subjects_from_fp_list(unittest.TestCase):
    def test_results(self):
        prefix = "~/prefix"
        subjects = ["subject1", "subject2", "a", "subject12"]
        target = ["subject1", "subject2"]
        files = ["f1", "f2", "f3", os.path.join("aa", "0")]
        suffixes = [os.path.join(subj, f) for f in files for subj in subjects]
        input =  [os.path.join(prefix, suffix) for suffix in suffixes]
        result = tested_module.filter_subjects_from_fp_list(input, target)
        expected_suffixes = [os.path.join(subj, f) for f in files for subj in target]
        expected_results = [os.path.join(prefix, suffix) for suffix in expected_suffixes]
        self.assertEqual(set(result), set(expected_results))
        self.assertEqual(len(result), len(expected_results))
