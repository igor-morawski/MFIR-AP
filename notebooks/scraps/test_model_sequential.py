import unittest
import model
import numpy as np
import tensorflow as tf

BATCH_SIZE = 64
H, W = 32, 32
FRAMES = 100
BATCH_SHAPE = (BATCH_SIZE, FRAMES, H, W, 1)
RNN_BATCH_SHAPE = (BATCH_SIZE, FRAMES, model.EMBEDDING_UNITS)
FRAME_BATCH_SHAPE = (BATCH_SIZE, H, W, 1)

class Test_build_embedding(unittest.TestCase):
    def test_input(self):
        embedding = model.build_embedding()
        self.assertTrue(embedding)
        result = tf.cast(embedding(np.empty(FRAME_BATCH_SHAPE)), dtype=np.float32)
        self.assertEqual(result.shape, (BATCH_SIZE, model.EMBEDDING_UNITS))


class Test_build_RNN(unittest.TestCase):
    def test_input(self):
        rnn = model.build_RNN()
        self.assertTrue(rnn)
        result = tf.cast(rnn(np.empty(RNN_BATCH_SHAPE)), dtype=np.float32)
        self.assertEqual(result.shape, (BATCH_SIZE, FRAMES, model.RNN_UNITS))
        

class Test_build_classifier_logits(unittest.TestCase):
    def test_input(self):
        classifier = model.build_classifier_logits()
        self.assertTrue(classifier)
        result = tf.cast(classifier(np.empty(BATCH_SHAPE)), dtype=np.float32)
        self.assertEqual(result.shape, (BATCH_SIZE, FRAMES, model.N_CLASSES))

class Test_build_classifier(unittest.TestCase):
    def test_input(self):
        classifier = model.build_classifier()
        self.assertTrue(classifier)
        result = tf.cast(classifier(np.empty(BATCH_SHAPE)), dtype=np.float32)
        self.assertEqual(result.shape, (BATCH_SIZE, FRAMES, model.N_CLASSES))

    def test_compile_classifier(self):
        data = np.random.rand(np.prod(BATCH_SHAPE)).astype(np.float32).reshape(BATCH_SHAPE)
        y_true = ((np.random.rand(BATCH_SIZE * FRAMES).reshape([BATCH_SIZE, FRAMES]) > 0.5) * 1)
        classifier = model.build_classifier()
        self.assertTrue(classifier)
        result = tf.cast(classifier(np.empty(BATCH_SHAPE)), dtype=np.float32)
        classifier.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        classifier.fit(data, y_true)
        tf.keras.backend.clear_session()
        
class Test_exponential_loss(unittest.TestCase):
    def test_loss(self):
        y_true = ((np.random.rand(BATCH_SIZE * FRAMES).reshape([BATCH_SIZE, FRAMES]) > 0.5) * 1)
        y_pred = np.random.rand(BATCH_SIZE * FRAMES).reshape([BATCH_SIZE, FRAMES])
        model.exponential_loss(y_true, y_pred)

    def test_training(self):
        data = np.random.rand(np.prod(BATCH_SHAPE)).astype(np.float32).reshape(BATCH_SHAPE)
        y_true = ((np.random.rand(BATCH_SIZE * FRAMES).reshape([BATCH_SIZE, FRAMES]) > 0.5) * 1)
        classifier = model.build_classifier()
        self.assertTrue(classifier)
        classifier.compile(loss=model.exponential_loss, optimizer="adam", metrics=['accuracy'])
        classifier.fit(data, y_true)
        tf.keras.backend.clear_session()

class Test_baselines(unittest.TestCase):
    def test_baseline1A(self):
        '''
        Multiview TPA + Feature Fusion + Loss
        RNN(TimeDistributed(cat(CNN_TPA1, CNN_TPA2, CNN_TPA3)))
        '''
        #build the model
        pass #XXX rebuild to functional api