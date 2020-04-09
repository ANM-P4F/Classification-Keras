# BSD 2-Clause License
# 
# Copyright (c) 2020, ANM-P4F
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import os
import tensorflow as tf
import keras
from keras import Model
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import argparse
# from tensorflow.contrib.lite.python import lite
from tensorflow.lite.python import lite
import sys
import glob
import cv2
import numpy as np

print('********************************Tensorflow version*****************')
print(tf.__version__)
print('********************************Keras version**********************')
print(keras.__version__)
print('*******************************************************************')

num_calibration_steps = 100

jpegs = glob.glob('dataset/raw-img/train/00.bat_logo/*.jpg')
# print(jpegs)

def representative_dataset_gen():
  for i in range(num_calibration_steps):
    # Get sample input data as a numpy array in a method of your choosing.
    img = cv2.imread(jpegs[i])
    img = cv2.resize(img, (32,32))
    img = img.astype(np.float32)
    img = img/255
    img = img[np.newaxis, ...]
    yield [img]

def _main():

    keras_model_path = "model_best_weights.h5"

    converter = lite.TFLiteConverter.from_keras_model_file(
        keras_model_path, input_shapes={'input_1': [1, 32, 32, 3]})
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    # open("./model_best.tflite", "wb").write(tflite_model)
    open("./model_best_quantized.tflite", "wb").write(tflite_model)

if __name__ == '__main__':
    _main()