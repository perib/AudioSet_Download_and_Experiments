import tensorflow as tf
import numpy as np
import h5py
import math
from math import sqrt
import argparse
import os
import matplotlib as plt

##########################################################################################

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calculate or keep track of the interpolated average precision.

It provides an interface for calculating interpolated average precision for an
entire list or the top-n ranked items. For the definition of the
(non-)interpolated average precision:
http://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf

Example usages:
1) Use it as a static function call to directly calculate average precision for
a short ranked list in the memory.

```
import random

p = np.array([random.random() for _ in xrange(10)])
a = np.array([random.choice([0, 1]) for _ in xrange(10)])

ap = average_precision_calculator.AveragePrecisionCalculator.ap(p, a)
```

2) Use it as an object for long ranked list that cannot be stored in memory or
the case where partial predictions can be observed at a time (Tensorflow
predictions). In this case, we first call the function accumulate many times
to process parts of the ranked list. After processing all the parts, we call
peek_interpolated_ap_at_n.
```
p1 = np.array([random.random() for _ in xrange(5)])
a1 = np.array([random.choice([0, 1]) for _ in xrange(5)])
p2 = np.array([random.random() for _ in xrange(5)])
a2 = np.array([random.choice([0, 1]) for _ in xrange(5)])

# interpolated average precision at 10 using 1000 break points
calculator = average_precision_calculator.AveragePrecisionCalculator(10)
calculator.accumulate(p1, a1)
calculator.accumulate(p2, a2)
ap3 = calculator.peek_ap_at_n()
```
"""

import heapq
import random
import numbers

import numpy


class AveragePrecisionCalculator(object):
    """Calculate the average precision and average precision at n."""

    def __init__(self, top_n=None):
        """Construct an AveragePrecisionCalculator to calculate average precision.

        This class is used to calculate the average precision for a single label.

        Args:
          top_n: A positive Integer specifying the average precision at n, or
            None to use all provided data points.

        Raises:
          ValueError: An error occurred when the top_n is not a positive integer.
        """
        if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
            raise ValueError("top_n must be a positive integer or None.")

        self._top_n = top_n  # average precision at n
        self._total_positives = 0  # total number of positives have seen
        self._heap = []  # max heap of (prediction, actual)

    @property
    def heap_size(self):
        """Gets the heap size maintained in the class."""
        return len(self._heap)

    @property
    def num_accumulated_positives(self):
        """Gets the number of positive samples that have been accumulated."""
        return self._total_positives

    def accumulate(self, predictions, actuals, num_positives=None):
        """Accumulate the predictions and their ground truth labels.

        After the function call, we may call peek_ap_at_n to actually calculate
        the average precision.
        Note predictions and actuals must have the same shape.

        Args:
          predictions: a list storing the prediction scores.
          actuals: a list storing the ground truth labels. Any value
          larger than 0 will be treated as positives, otherwise as negatives.
          num_positives = If the 'predictions' and 'actuals' inputs aren't complete,
          then it's possible some true positives were missed in them. In that case,
          you can provide 'num_positives' in order to accurately track recall.

        Raises:
          ValueError: An error occurred when the format of the input is not the
          numpy 1-D array or the shape of predictions and actuals does not match.
        """
        if len(predictions) != len(actuals):
            raise ValueError("the shape of predictions and actuals does not match.")

        if not num_positives is None:
            if not isinstance(num_positives, numbers.Number) or num_positives < 0:
                raise ValueError("'num_positives' was provided but it wan't a nonzero number.")

        if not num_positives is None:
            self._total_positives += num_positives
        else:
            self._total_positives += numpy.size(numpy.where(actuals > 0))
        topk = self._top_n
        heap = self._heap

        for i in range(numpy.size(predictions)):
            if topk is None or len(heap) < topk:
                heapq.heappush(heap, (predictions[i], actuals[i]))
            else:
                if predictions[i] > heap[0][0]:  # heap[0] is the smallest
                    heapq.heappop(heap)
                    heapq.heappush(heap, (predictions[i], actuals[i]))

    def clear(self):
        """Clear the accumulated predictions."""
        self._heap = []
        self._total_positives = 0

    def peek_ap_at_n(self):
        """Peek the non-interpolated average precision at n.

        Returns:
          The non-interpolated average precision at n (default 0).
          If n is larger than the length of the ranked list,
          the average precision will be returned.
        """
        if self.heap_size <= 0:
            return 0
        predlists = numpy.array(list(zip(*self._heap)))

        ap = self.ap_at_n(predlists[0],
                          predlists[1],
                          n=self._top_n,
                          total_num_positives=self._total_positives)
        return ap

    @staticmethod
    def ap(predictions, actuals):
        """Calculate the non-interpolated average precision.

        Args:
          predictions: a numpy 1-D array storing the sparse prediction scores.
          actuals: a numpy 1-D array storing the ground truth labels. Any value
          larger than 0 will be treated as positives, otherwise as negatives.

        Returns:
          The non-interpolated average precision at n.
          If n is larger than the length of the ranked list,
          the average precision will be returned.

        Raises:
          ValueError: An error occurred when the format of the input is not the
          numpy 1-D array or the shape of predictions and actuals does not match.
        """
        return AveragePrecisionCalculator.ap_at_n(predictions,
                                                  actuals,
                                                  n=None)

    @staticmethod
    def ap_at_n(predictions, actuals, n=15, total_num_positives=None):
        """Calculate the non-interpolated average precision.

        Args:
          predictions: a numpy 1-D array storing the sparse prediction scores.
          actuals: a numpy 1-D array storing the ground truth labels. Any value
          larger than 0 will be treated as positives, otherwise as negatives.
          n: the top n items to be considered in ap@n.
          total_num_positives : (optionally) you can specify the number of total
            positive
          in the list. If specified, it will be used in calculation.

        Returns:
          The non-interpolated average precision at n.
          If n is larger than the length of the ranked list,
          the average precision will be returned.

        Raises:
          ValueError: An error occurred when
          1) the format of the input is not the numpy 1-D array;
          2) the shape of predictions and actuals does not match;
          3) the input n is not a positive integer.
        """
        if len(predictions) != len(actuals):
            raise ValueError("the shape of predictions and actuals does not match.")

        if n is not None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be 'None' or a positive integer."
                                 " It was '%s'." % n)

        ap = 0.0

        predictions = numpy.array(predictions)
        actuals = numpy.array(actuals)

        # add a shuffler to avoid overestimating the ap
        predictions, actuals = AveragePrecisionCalculator._shuffle(predictions,
                                                                   actuals)
        sortidx = sorted(
            range(len(predictions)),
            key=lambda k: predictions[k],
            reverse=True)

        if total_num_positives is None:
            numpos = numpy.size(numpy.where(actuals > 0))
        else:
            numpos = total_num_positives

        if numpos == 0:
            return 0

        if n is not None:
            numpos = min(numpos, n)
        delta_recall = 1.0 / numpos
        poscount = 0.0

        # calculate the ap
        r = len(sortidx)
        if n is not None:
            r = min(r, n)
        for i in range(r):
            if actuals[sortidx[i]] > 0:
                poscount += 1
                ap += poscount / (i + 1.0) * delta_recall
        return ap

    @staticmethod
    def _shuffle(predictions, actuals):
        random.seed(0)
        suffidx = random.sample(range(len(predictions)), len(predictions))
        predictions = predictions[suffidx]
        actuals = actuals[suffidx]
        return predictions, actuals

    @staticmethod
    def _zero_one_normalize(predictions, epsilon=1e-7):
        """Normalize the predictions to the range between 0.0 and 1.0.

        For some predictions like SVM predictions, we need to normalize them before
        calculate the interpolated average precision. The normalization will not
        change the rank in the original list and thus won't change the average
        precision.

        Args:
          predictions: a numpy 1-D array storing the sparse prediction scores.
          epsilon: a small constant to avoid denominator being zero.

        Returns:
          The normalized prediction.
        """
        denominator = numpy.max(predictions) - numpy.min(predictions)
        ret = (predictions - numpy.min(predictions)) / numpy.max(denominator,
                                                                 epsilon)
        return ret


##########################################################################################

def get_saved_weights(MODEL_TO_LOAD, COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize,padding, poolmethod,conv1_times_hanning,Learning_Rate,multiple_labels):
    numlabels = 527
    nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH,numlabels,train_mean_coch,Conv1_filtersize,padding, poolmethod,conv1_times_hanning)
    #Get the loss functions
    Cross_Entropy_Train_on_Labels(nets,numlabels,Learning_Rate,multiple_labels)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,MODEL_TO_LOAD)
        ############################################
        #Save model weights and 

        saved_weights = {}


        saved_weights['conv1_Weights'] = nets['conv1_Weights'].eval(session = sess)
        saved_weights['bias1_Weights'] = nets['conv1_bias'].eval(session = sess)#ugh I had a type in the naming :/

        for layer in range(2,6):
            saved_weights['conv{0}_Weights'.format(layer)] = nets['conv{0}_Weights'.format(layer)].eval(session = sess)
            saved_weights['conv{0}_bias'.format(layer)] = nets['conv{0}_bias'.format(layer)].eval(session = sess)

    tf.reset_default_graph()
    
    return saved_weights
    

# Generates and returns the VGG graph (up to the predictions node)
def Gen_VGG(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch, padding="SAME"):
    final_filter = 512
    full_length = final_filter * math.ceil(COCHLEAGRAM_LENGTH / 171 / 2 / 2 / 2 / 2 / 2) * math.ceil(
        171 / 2 / 2 / 2 / 2 / 2)
    print(full_length)
    nets = {}
    with tf.name_scope('input'):
        nets['input_to_net'] = tf.placeholder(tf.float32, [None, COCHLEAGRAM_LENGTH], name='input_to_net')
        mean_tensor = tf.constant(train_mean_coch, dtype=tf.float32)
        nets['subtract_mean'] = tf.subtract(nets['input_to_net'], mean_tensor, name='Subtract_Mean')
        nets['reshapedCoch'] = tf.reshape(nets['subtract_mean'], [-1, 171, int(COCHLEAGRAM_LENGTH / 171), 1],
                                          name='reshape_input')

    with tf.name_scope("accuracy"):
        nets['accuracy'] = tf.placeholder(tf.float32, (), name='acc')
        nets['accsum'] = tf.summary.scalar("accuracy", nets['accuracy'])

    # sess.run(tf.global_variables_initializer())
    # thing = sess.run(reshapedCoch,feed_dict={input_to_net:train_coch[0:2],actual_labels:trainlabels[0:2]})

    print("building conv1")
    with tf.variable_scope('conv1') as scope:
        # Conv_1_1 and save the graphs
        tf.get_variable_scope().reuse_variables()
        nets['conv1_1_Weights'] = weight_variable([3, 3, 1, 64])

        nets['layer1'] = variable_summaries(nets['conv1_1_Weights'])

        nets['grid'] = put_kernels_on_grid(nets['conv1_1_Weights'], 8, 8)
        nets['conv1_weight_image'] = tf.summary.image('conv1/kernels', nets['grid'], max_outputs=3)
        nets['conv1_1'] = conv2d(nets['reshapedCoch'], Weights=nets['conv1_1_Weights'], bias=bias_variable([64]),
                                 strides=1, name='conv1_1')

    with tf.name_scope("maxpool1"):
        nets['maxpool1'] = maxpool2x2(nets['conv1_1'], k=2, name='maxpool1', padding=padding)

    print("building conv2")
    with tf.name_scope('conv2'):
        nets['conv2_1'] = conv2d(nets['maxpool1'], Weights=weight_variable([3, 3, 64, 128]), bias=bias_variable([128]),
                                 strides=1, name='conv2_1')

    with tf.name_scope("maxpool2"):
        nets['maxpool2'] = maxpool2x2(nets['conv2_1'], k=2, name='maxpool2', padding=padding)

    print("building conv3")
    with tf.name_scope('conv3'):
        nets['conv3_1'] = conv2d(nets['maxpool2'], Weights=weight_variable([3, 3, 128, 256]), bias=bias_variable([256]),
                                 strides=1, name='conv3_1')
        nets['conv3_2'] = conv2d(nets['conv3_1'], Weights=weight_variable([3, 3, 256, 256]), bias=bias_variable([256]),
                                 strides=1, name='conv3_2')
    with tf.name_scope("maxpool3"):
        nets['maxpool3'] = maxpool2x2(nets['conv3_2'], k=2, name='maxpool3', padding=padding)

    print("building conv4")
    with tf.name_scope('conv4'):
        nets['conv4_1'] = conv2d(nets['maxpool3'], Weights=weight_variable([3, 3, 256, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv4_1')
        nets['conv4_2'] = conv2d(nets['conv4_1'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv4_2')

    with tf.name_scope("maxpool4"):
        nets['maxpool4'] = maxpool2x2(nets['conv4_2'], k=2, name='maxpool4', padding=padding)

    print("building conv5")
    with tf.name_scope('conv5'):
        nets['conv5_1'] = conv2d(nets['maxpool4'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv5_1')
        nets['conv5_2'] = conv2d(nets['conv5_1'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv5_2')

    with tf.name_scope("maxpool5"):
        nets['maxpool5'] = maxpool2x2(nets['conv5_2'], k=2, name='maxpool5', padding=padding)

    with tf.name_scope("flatten"):
        nets['flattened'] = tf.reshape(nets['maxpool5'], [-1, full_length])

    with tf.name_scope("keep_prob"):
        nets['keep_prob'] = tf.placeholder(tf.float32)

    print("building fc_1")
    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([full_length, 4096])
        b_fc1 = bias_variable([4096])
        nets['fc_1'] = tf.nn.relu(tf.matmul(nets['flattened'], W_fc1) + b_fc1)
        nets['h_fc1_drop'] = tf.nn.dropout(nets['fc_1'], nets['keep_prob'])

    print("building fc_2")
    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([4096, 4096])
        b_fc2 = bias_variable([4096])
        nets['fc_2'] = tf.nn.relu(tf.matmul(nets['h_fc1_drop'], W_fc2) + b_fc2)
        nets['h_fc2_drop'] = tf.nn.dropout(nets['fc_2'], nets['keep_prob'])

    print("building fc_3")
    with tf.name_scope('fc_3'):
        W_fc3 = weight_variable([4096, numlabels], name='W_fc3')  # 4,959,232
        b_fc3 = bias_variable([numlabels], name='b_fc3')
        nets['predicted_labels'] = tf.nn.relu(tf.matmul(nets['h_fc2_drop'], W_fc3) + b_fc3)

    print("done building")

    return nets


# Generates and returns the VGG graph (up to the predictions node)
def Gen_VGG19(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch, padding="SAME"):
    final_filter = 512
    full_length = final_filter * math.ceil(COCHLEAGRAM_LENGTH / 171 / 2 / 2 / 2 / 2 / 2) * math.ceil(
        171 / 2 / 2 / 2 / 2 / 2)
    print(full_length)
    nets = {}
    with tf.name_scope('input'):
        nets['input_to_net'] = tf.placeholder(tf.float32, [None, COCHLEAGRAM_LENGTH], name='input_to_net')
        mean_tensor = tf.constant(train_mean_coch, dtype=tf.float32)
        nets['subtract_mean'] = tf.subtract(nets['input_to_net'], mean_tensor, name='Subtract_Mean')
        nets['reshapedCoch'] = tf.reshape(nets['subtract_mean'], [-1, 171, int(COCHLEAGRAM_LENGTH / 171), 1],
                                          name='reshape_input')

    with tf.name_scope("accuracy"):
        nets['accuracy'] = tf.placeholder(tf.float32, (), name='acc')
        nets['accsum'] = tf.summary.scalar("accuracy", nets['accuracy'])

    # sess.run(tf.global_variables_initializer())
    # thing = sess.run(reshapedCoch,feed_dict={input_to_net:train_coch[0:2],actual_labels:trainlabels[0:2]})

    print("building conv1")
    with tf.variable_scope('conv1') as scope:
        # Conv_1_1 and save the graphs
        tf.get_variable_scope().reuse_variables()
        nets['conv1_1_Weights'] = weight_variable([3, 3, 1, 64])

        nets['layer1'] = variable_summaries(nets['conv1_1_Weights'])

        nets['grid'] = put_kernels_on_grid(nets['conv1_1_Weights'], 8, 8)
        nets['conv1_weight_image'] = tf.summary.image('conv1/kernels', nets['grid'], max_outputs=3)
        nets['conv1_1'] = conv2d(nets['reshapedCoch'], Weights=nets['conv1_1_Weights'], bias=bias_variable([64]),
                                 strides=1, name='conv1_1')

        # conv_1_2
        nets['conv1_2'] = conv2d(nets['conv1_1'], Weights=weight_variable([3, 3, 64, 64]), bias=bias_variable([64]),
                                 strides=1, name='conv1_2')

    with tf.name_scope("maxpool1"):
        nets['maxpool1'] = maxpool2x2(nets['conv1_2'], k=2, name='maxpool1', padding=padding)

    print("building conv2")
    with tf.name_scope('conv2'):
        nets['conv2_1'] = conv2d(nets['maxpool1'], Weights=weight_variable([3, 3, 64, 128]), bias=bias_variable([128]),
                                 strides=1, name='conv2_1')
        nets['conv2_2'] = conv2d(nets['conv2_1'], Weights=weight_variable([3, 3, 128, 128]), bias=bias_variable([128]),
                                 strides=1, name='conv2_2')
    with tf.name_scope("maxpool2"):
        nets['maxpool2'] = maxpool2x2(nets['conv2_2'], k=2, name='maxpool2', padding=padding)

    print("building conv3")
    with tf.name_scope('conv3'):
        nets['conv3_1'] = conv2d(nets['maxpool2'], Weights=weight_variable([3, 3, 128, 256]), bias=bias_variable([256]),
                                 strides=1, name='conv3_1')
        nets['conv3_2'] = conv2d(nets['conv3_1'], Weights=weight_variable([3, 3, 256, 256]), bias=bias_variable([256]),
                                 strides=1, name='conv3_2')
        nets['conv3_3'] = conv2d(nets['conv3_2'], Weights=weight_variable([3, 3, 256, 256]), bias=bias_variable([256]),
                                 strides=1, name='conv3_3')
        nets['conv3_4'] = conv2d(nets['conv3_3'], Weights=weight_variable([3, 3, 256, 256]), bias=bias_variable([256]),
                                 strides=1, name='conv3_4')
    with tf.name_scope("maxpool3"):
        nets['maxpool3'] = maxpool2x2(nets['conv3_4'], k=2, name='maxpool3', padding=padding)

    print("building conv4")
    with tf.name_scope('conv4'):
        nets['conv4_1'] = conv2d(nets['maxpool3'], Weights=weight_variable([3, 3, 256, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv4_1')
        nets['conv4_2'] = conv2d(nets['conv4_1'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv4_2')
        nets['conv4_3'] = conv2d(nets['conv4_2'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv4_3')
        nets['conv4_4'] = conv2d(nets['conv4_3'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv4_4')

    with tf.name_scope("maxpool4"):
        nets['maxpool4'] = maxpool2x2(nets['conv4_4'], k=2, name='maxpool4', padding=padding)

    print("building conv5")
    with tf.name_scope('conv5'):
        nets['conv5_1'] = conv2d(nets['maxpool4'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv5_1')
        nets['conv5_2'] = conv2d(nets['conv5_1'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv5_2')
        nets['conv5_3'] = conv2d(nets['conv5_2'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv5_3')
        nets['conv5_4'] = conv2d(nets['conv5_3'], Weights=weight_variable([3, 3, 512, 512]), bias=bias_variable([512]),
                                 strides=1, name='conv5_4')

    with tf.name_scope("maxpool5"):
        nets['maxpool5'] = maxpool2x2(nets['conv5_4'], k=2, name='maxpool5', padding=padding)

    with tf.name_scope("flatten"):
        nets['flattened'] = tf.reshape(nets['maxpool5'], [-1, full_length])

    with tf.name_scope("keep_prob"):
        nets['keep_prob'] = tf.placeholder(tf.float32)

    print("building fc_1")
    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([full_length, 4096])
        b_fc1 = bias_variable([4096])
        nets['fc_1'] = tf.nn.relu(tf.matmul(nets['flattened'], W_fc1) + b_fc1)
        nets['h_fc1_drop'] = tf.nn.dropout(nets['fc_1'], nets['keep_prob'])

    print("building fc_2")
    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([4096, 4096])
        b_fc2 = bias_variable([4096])
        nets['fc_2'] = tf.nn.relu(tf.matmul(nets['h_fc1_drop'], W_fc2) + b_fc2)
        nets['h_fc2_drop'] = tf.nn.dropout(nets['fc_2'], nets['keep_prob'])

    print("building fc_3")
    with tf.name_scope('fc_3'):
        W_fc3 = weight_variable([4096, numlabels], name='W_fc3')  # 4,959,232
        b_fc3 = bias_variable([numlabels], name='b_fc3')
        nets['predicted_labels'] = tf.nn.relu(tf.matmul(nets['h_fc2_drop'], W_fc3) + b_fc3)

    print("done building")

    return nets


# Generates and returns our own graph (up to the predictions node)
def Gen_audiosetNet(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch, Conv1_filtersize, padding, poolmethod,
                    conv1_times_hanning,Saved_Weights = None):
    variable_list = []

    conv1_strides = 3
    conv2_strides = 2
    conv3_strides = 1
    conv4_strides = 1
    conv5_strides = 1
    final_filter = 512
    full_length = final_filter * math.ceil(
        COCHLEAGRAM_LENGTH / 171 / conv1_strides / conv2_strides / conv3_strides / conv4_strides / conv5_strides / 4 / 2) * math.ceil(
        171 / conv1_strides / conv2_strides / conv3_strides / conv4_strides / conv5_strides / 4 / 2)

    nets = {}
    with tf.name_scope('input'):
        nets['input_to_net'] = tf.placeholder(tf.float32, [None, COCHLEAGRAM_LENGTH], name='input_to_net')
        mean_tensor = tf.constant(train_mean_coch, dtype=tf.float32)
        nets['subtract_mean'] = tf.subtract(nets['input_to_net'], mean_tensor, name='Subtract_Mean')
        nets['reshapedCoch'] = tf.reshape(nets['subtract_mean'], [-1, 171, int(COCHLEAGRAM_LENGTH / 171), 1],
                                          name='reshape_input')

        # print("input ", nets['reshapedCoch'].shape)

    with tf.name_scope("images"):
        nets['coch_images'] = tf.summary.image('image', nets['reshapedCoch'][0:5, :, :, :])

    with tf.name_scope("accuracy"):
        nets['accuracy'] = tf.placeholder(tf.float32, (), name='acc')
        nets['accsum'] = tf.summary.scalar("accuracy", nets['accuracy'])

    with tf.variable_scope('conv1') as scope:
        tf.get_variable_scope().reuse_variables()
        nets['conv1_Weights'] = weight_variable([Conv1_filtersize, Conv1_filtersize, 1, 96], name='conv1_Weights',freeze_weight=Saved_Weights)
        nets['conv1_bias'] = bias_variable([96], name='bias1_Weights',freeze_bias=Saved_Weights)
        if conv1_times_hanning:
            square = make_hanning_kernel(downsample=2, length_of_window=Conv1_filtersize, make_plots=False)

            h_kernel = np.ones((Conv1_filtersize, Conv1_filtersize, 1, 96))
            for i in range(1):
                for j in range(96):
                    h_kernel[:, :, i, j] = square
            nets['h_kernel '] = tf.constant(h_kernel, dtype=tf.float32)
            nets['conv1_Weights_x_h_kernal'] = tf.multiply(nets['conv1_Weights'], nets['h_kernel '])
            nets['conv1'] = tf.nn.local_response_normalization(
                conv2d(nets['reshapedCoch'], Weights=nets['conv1_Weights_x_h_kernal'], bias=nets['conv1_bias'],
                       strides=conv1_strides, name='conv1', padding=padding), depth_radius=2)
            nets['layer1'] = variable_summaries(nets['conv1_Weights_x_h_kernal'])
            nets['grid'] = put_kernels_on_grid(nets['conv1_Weights_x_h_kernal'], 16, 6)
            nets['conv1_weight_image'] = tf.summary.image('conv1/kernels', nets['grid'], max_outputs=3)
        else:
            nets['conv1'] = tf.nn.local_response_normalization(
                conv2d(nets['reshapedCoch'], Weights=nets['conv1_Weights'], bias=nets['conv1_bias'],
                       strides=conv1_strides, name='conv1', padding=padding), depth_radius=2)
            nets['layer1'] = variable_summaries(nets['conv1_Weights'])
            nets['grid'] = put_kernels_on_grid(nets['conv1_Weights'], 16, 6)
            nets['conv1_weight_image'] = tf.summary.image('conv1/kernels', nets['grid'], max_outputs=3)

        '''tf.get_variable_scope().reuse_variables()
        square = make_hanning_kernel(downsample=2, length_of_window=Conv1_filtersize, make_plots=False)
        h_kernel = np.ones((Conv1_filtersize,Conv1_filtersize,1,96))
        for i in range(1):
            for j in range(96):
                h_kernel[:,:,i,j] = square             
        nets['h_kernel '] =  tf.constant(h_kernel, dtype=tf.float32) 
        nets['conv1_Weights'] = tf.multiply(weight_variable([Conv1_filtersize,Conv1_filtersize,1,96], name = 'conv1_Weights'),nets['h_kernel ']) 
        nets['conv1_bias'] = bias_variable([96],name = 'bias1_Weights')  
        nets['layer1'] = variable_summaries(nets['conv1_Weights'])
        nets['grid'] = put_kernels_on_grid(nets['conv1_Weights'],16,6)
        nets['conv1_weight_image'] = tf.summary.image('conv1/kernels', nets['grid'], max_outputs=3)

        nets['conv1'] = tf.nn.local_response_normalization(conv2d(nets['reshapedCoch'], Weights = nets['conv1_Weights'], bias = nets['conv1_bias'], strides=conv1_strides, name='conv1',padding=padding), depth_radius = 2)
        '''

    with tf.name_scope("conv1_summaries"):
        sum_list = []
        for im in range(2):
            for channel in range(2):
                sum_list.append(tf.summary.image('conv1/featuremaps_im_{0}_ch{0}'.format(im, channel),
                                                 nets['conv1'][im:im + 1, :, :, channel:channel + 1], max_outputs=3))
                sum_list.append(tf.summary.image('conv1/weight_im{0}_ch{0}'.format(im, channel),
                                                 tf.transpose(nets['conv1_Weights'], (2, 0, 1, 3))[0:1, :, :,
                                                 channel:channel + 1], max_outputs=3))
                sum_list.append(tf.summary.histogram('conv1/featuremaps_hist_im_{0}_ch{0}'.format(im, channel),
                                                     nets['conv1'][im:im + 1, :, :, channel:channel + 1]))
        nets['conv1_sums'] = tf.summary.merge(sum_list)

        # print("conv1", nets['conv1'].shape)

    if poolmethod == "MAXPOOL":
        with tf.name_scope("maxpool1"):
            nets['pool1'] = maxpool2x2(nets['conv1'], k=2, name='maxpool1', padding="SAME")
    elif poolmethod == "HPOOL":
        with tf.name_scope("h_pool1"):
            nets['pool1'] = add_hanning_pooling(nets, 'conv1', downsample=2, length_of_window=8, layer_name='pool1',
                                                make_plots=False)

            # print("pool1, ", nets['pool1'].shape)

    with tf.name_scope('conv2'):
        nets['conv2_Weights'] = weight_variable([5, 5, 96, 256], name='conv2_Weights',freeze_weight=Saved_Weights)
        nets['conv2_bias'] = bias_variable([256], name='conv2_bias',freeze_bias=Saved_Weights)

        nets['conv2'] = tf.nn.local_response_normalization(
            conv2d(nets['pool1'], Weights=nets['conv2_Weights'], bias=nets['conv2_bias'],
                   strides=conv2_strides, name='conv2', padding=padding), depth_radius=2)

        # print("conv2 ", nets['conv2'].shape)

    if poolmethod == "MAXPOOL":
        with tf.name_scope("maxpool2"):
            nets['pool2'] = maxpool2x2(nets['conv2'], k=2, name='maxpool2', padding="SAME")
    elif poolmethod == "HPOOL":
        with tf.name_scope("h_pool2"):
            nets['pool2'] = add_hanning_pooling(nets, 'conv2', downsample=2, length_of_window=8, layer_name='pool2',
                                                make_plots=False)

    # print("pool2, ", nets['pool2'].shape)

    with tf.name_scope('conv3'):
        nets['conv3_Weights'] = weight_variable([3, 3, 256, 512], name='conv3_Weights',freeze_weight=Saved_Weights)
        nets['conv3_bias'] =  bias_variable([512], name='conv3_bias',freeze_bias=Saved_Weights)
        nets['conv3'] = conv2d(nets['pool2'], Weights=nets['conv3_Weights'] ,
                               bias=nets['conv3_bias'], strides=conv3_strides, name='conv3',
                               padding=padding)
        # print("conv3 ", nets['conv3'].shape)

    with tf.name_scope('conv4'):
        nets['conv4_Weights'] = weight_variable([3, 3, 512, 1024], name='conv4_Weights',freeze_weight=Saved_Weights)
        nets['conv4_bias'] = bias_variable([1024], name='conv4_bias',freeze_bias=Saved_Weights)
        nets['conv4'] = conv2d(nets['conv3'], Weights=nets['conv4_Weights'],
                               bias=nets['conv4_bias'], strides=conv4_strides, name='conv4',
                               padding=padding)
        # print("conv4", nets['conv4'].shape)

    with tf.name_scope('conv5'):
        nets['conv5_Weights'] = weight_variable([3, 3, 1024, final_filter], name='conv5_Weights',freeze_weight=Saved_Weights)
        nets['conv5_bias'] = bias_variable([final_filter], name='conv5_bias',freeze_bias=Saved_Weights)
        nets['conv5'] = conv2d(nets['conv4'], Weights=nets['conv5_Weights'] ,
                               bias=nets['conv5_bias'], strides=conv5_strides,
                               name='conv5', padding=padding)

        # print("conv5",nets['conv5'].shape )

    if poolmethod == "MAXPOOL":
        with tf.name_scope("maxpool3"):
            nets['pool3'] = maxpool2x2(nets['conv5'], k=2, name='maxpool3', padding="SAME")
    elif poolmethod == "HPOOL":
        with tf.name_scope("h_pool3"):
            nets['pool3'] = add_hanning_pooling(nets, 'conv5', downsample=2, length_of_window=8, layer_name='pool3',
                                                make_plots=False)

    with tf.name_scope("flatten"):
        nets['flattened'] = tf.reshape(nets['pool3'], [-1, full_length])

    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([full_length, 1024])
        b_fc1 = bias_variable([1024])
        nets['fc_1'] = tf.nn.relu(tf.matmul(nets['flattened'], W_fc1) + b_fc1)

        nets['keep_prob'] = tf.placeholder(tf.float32)
        nets['h_fc1_drop'] = tf.nn.dropout(nets['fc_1'], nets['keep_prob'])

    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([1024, numlabels])
        b_fc2 = bias_variable([numlabels])
        nets['predicted_labels'] = tf.matmul(nets['h_fc1_drop'], W_fc2) + b_fc2

    return nets


def Cross_Entropy_Train_on_Labels(nets, numlabels, Learning_Rate, multiple_labels=True):
    # The evaluation/training of the graph
    with tf.variable_scope("Cross_Entropy"):
        nets['actual_labels'] = tf.placeholder(tf.float32, [None, numlabels], name='actual_labels')

        if multiple_labels:
            nets['cross_entroy'] = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=nets['actual_labels'], logits=nets['predicted_labels']))
        else:
            #nets['cross_entroy'] = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=nets['actual_labels'], logits=nets['predicted_labels']))
            nets['cross_entroy'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =nets['actual_labels'],logits = nets['predicted_labels'] ))
            # nets['cross_entroy'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels =nets['actual_labels'],logits = nets['predicted_labels'] ))

    with tf.variable_scope("TrainStep"):
        '''
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,'''
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08
        nets['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        nets['train_step'] = tf.train.AdamOptimizer(Learning_Rate).minimize(nets['cross_entroy'],
                                                                            global_step=nets['global_step'])
        # nets['train_step'] = tf.train.AdamOptimizer(learning_rate=Learning_Rate,beta1=beta1,beta2=beta2,epsilon=epsilon).minimize(nets['cross_entroy'])

        # num = np.sum(nets['actual_labelsi'])
        # nets['accuracy'] = tf.metrics.mean(tf.nn.in_top_k(nets['predicted_labels'],nets['actual_labelsi'],2))

        # calculating accuracy
    with tf.name_scope("Summaries"):
        nets['label_indeces'] = tf.placeholder(tf.int32, [None, 15], name='label_indeces')
        if numlabels >= 15:
            _, nets['indeces'] = tf.nn.top_k(nets['predicted_labels'], 15)
            prediction_sums = tf.summary.histogram('predicted_labels_histogram', nets['indeces'])
            nets['numCorrect'] = tf.shape(tf.sets.set_intersection(nets['indeces'], nets['label_indeces'], False).values)[0]

        _, nets['Top_1_index'] = tf.nn.top_k(nets['predicted_labels'], 1)
        nets['numCorrect_top_pred'] = \
        tf.shape(tf.sets.set_intersection(nets['Top_1_index'], nets['label_indeces'], False).values)[0]

    with tf.name_scope("Cross_Entropy_Loss"):
        nets['xe_ave'] = tf.placeholder(tf.float32, (), name='xe_ave')
        nets['xesum'] = tf.summary.scalar("Test_cross_entropy_loss", nets['xe_ave'])
        nets['cross_entropy_summary'] = tf.summary.scalar("Train_cross_entropy_loss", nets['cross_entroy'])

    with tf.name_scope("GAP"):
        nets['GAP'] = tf.placeholder(tf.float32, (), name='GAP')
        nets['GAPsum'] = tf.summary.scalar("GAPsum", nets['GAP'])
    
    with tf.name_scope("MAP"):
        nets['MAP'] = tf.placeholder(tf.float32, (), name='MAP')
        nets['MAPsum'] = tf.summary.scalar("MAPsum", nets['MAP'])
    
    with tf.name_scope("AUC"):
        nets['l'] = tf.placeholder(tf.int32, [None, numlabels], name='labels')
        nets['p'] = tf.placeholder(tf.float32, [None, numlabels], name='preds')

        nets["auc"], nets['update_auc'] = tf.metrics.auc(
            labels = nets['l'],
            predictions = nets['p'],
            weights=None,
            num_thresholds=10,
            metrics_collections=None,
            updates_collections=None,
            curve='ROC',
            name='auc_accumulator'
        )
        
        nets['AUCsum'] = tf.summary.scalar("AUCsum", nets["auc"])
        
        
    with tf.name_scope("Count_Tracker"):
        nets['current_epoch'] = tf.Variable(0, name='current_epoch', trainable=False, dtype=tf.int32)
        nets['increment_current_epoch'] = tf.assign(nets['current_epoch'], nets['current_epoch'] + 1)

        nets['current_step'] = tf.Variable(0, name='current_step', trainable=False, dtype=tf.int32)
        nets['increment_current_step'] = tf.assign(nets['current_step'], nets['current_step'] + 1)
        nets['reset_current_step'] = tf.assign(nets['current_step'], 0)


# thanks to https://gist.githubusercontent.com/kukuruza/03731dc494603ceab0c5/raw/3d708320145df0a962cfadb95b3f716b623994e0/gist_cifar10_train.py
def put_kernels_on_grid(kernel, grid_Y, grid_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))  # 3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # 3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


# a bunch of neat stats we can summarize
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        meansum = tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        stdv = tf.summary.scalar('stddev', stddev)
        maxsum = tf.summary.scalar('max', tf.reduce_max(var))
        minsum = tf.summary.scalar('min', tf.reduce_min(var))
        hist = tf.summary.histogram('histogram', var)

        return tf.summary.merge([meansum, stdv, maxsum, minsum, hist])


def weight_variable(shape, name=None, freeze_weight = None):
    if freeze_weight == None:
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(freeze_weight[name], name=name, trainable = False)

def bias_variable(shape, name=None,freeze_bias = None):
    if freeze_bias == None:
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(freeze_bias[name], name=name,trainable = False)

# builds the components
def conv2d(inputtensor, Weights, bias, strides=1, name=None, padding="SAME"):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(inputtensor, Weights, strides=[1, strides, strides, 1], padding=padding, name=name)
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)


def maxpool2x2(x, k=2, name=None, padding="SAME"):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding=padding, name=name)


def add_hanning_pooling(nets, top_node, downsample=2, length_of_window=8, layer_name=False, make_plots=False):
    """
    Add a layer using a hanning kernel for pooling

    Parameters
    ----------
    nets : dictionary
        a dictionary containing the graph
    downsample : int
        proportion downsampling
    top_node : string
        specify the node after which the spectemp filters will be added and used as input for the FFT.
    layer_name : False or string
        name for the layer. If false appends "_hpool" to the top_node name
    Returns
    -------
    nets : dictionary
        a dictionary containing the graph, containing a pooling layer
    """
    if not layer_name:
        layer_name = top_node + '_hpool'

    n_channels = nets[top_node].get_shape().as_list()[3]
    nets['hanning_tensor_' + top_node] = make_hanning_kernel_tensor(n_channels, downsample=downsample,
                                                                    length_of_window=length_of_window,
                                                                    make_plots=make_plots)
    return tf.nn.depthwise_conv2d(nets[top_node], nets['hanning_tensor_' + top_node], [1, downsample, downsample, 1],
                                  'SAME', name=layer_name)


def make_hanning_kernel_tensor(n_channels, downsample=2, length_of_window=8, make_plots=False):
    """
    Make a tensor containing the symmetric 2d hanning kernel to use for the pooling filters
    For downsample=2, using length_of_window=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For downsample=3, using length_of_window=12 gives a reduction of -28.607805482176282 at 1/6 cycles

    Parameters
    ----------
    n_channels : int
        number of channels to copy the kernel into
    downsample : int
        proportion downsampling
    length_of_window : int
        how large of a window to use
    make_plots: boolean
        make plots of the filters

    Returns
    -------
    hanning_tensor : tensorflow tensor
        tensorflow tensor containing the hanning kernel with size [1 length_of_window length_of_window n_channels]

    """
    hanning_kernel = make_hanning_kernel(downsample=downsample, length_of_window=length_of_window,
                                         make_plots=make_plots)
    hanning_kernel = np.expand_dims(np.dstack([hanning_kernel.astype(np.float32)] * n_channels), axis=3)
    hanning_tensor = tf.constant(hanning_kernel)
    return hanning_tensor


def make_hanning_kernel(downsample=2, length_of_window=8, make_plots=False):
    """
    Make the symmetric 2d hanning kernel to use for the pooling filters
    For downsample=2, using length_of_window=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For downsample=3, using length_of_window=12 gives a reduction of -28.607805482176282 at 1/6 cycles

    Parameters
    ----------
    downsample : int
        proportion downsampling
    length_of_window : int
        how large of a window to use
    make_plots: boolean
        make plots of the filters

    Returns
    -------
    two_dimensional_kernel : numpy array
        hanning kernel in 2d to use as a kernel for filtering

    """

    window = 0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(length_of_window)) / (length_of_window - 1)))
    A = np.fft.fft(window, 2048) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20.0 * np.log10(np.abs(np.fft.fftshift(A / abs(A).max())))

    nyquist = 1 / (2 * downsample)
    ny_idx = np.where(np.abs(freq - nyquist) == np.abs(freq - nyquist).min())[0][0]
    two_dimensional_kernel = np.sqrt(np.outer(window, window))

    if make_plots:
        print(['Frequency response at ' + 'nyquist' + ' is ' + 'response[ny_idx]'])
        plt.figure()
        plt.plot(window)
        plt.title(r"Hanning window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.figure()
        plt.plot(freq, response)
        plt.axis([-0.5, 0.5, -120, 0])
        plt.title(r"Frequency response of the Hanning window")
        plt.ylabel("Normalized magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.figure()
        plt.matshow(two_dimensional_kernel)
        plt.colorbar()
        plt.title(r"Two dimensional kernel")

    return two_dimensional_kernel
