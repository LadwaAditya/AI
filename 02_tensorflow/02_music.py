import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import midi_manipulator

lowest_note = midi_manipulator.lowerBound
highest_note = midi_manipulator.upperBound
note_range = highest_note - lowest_note

num_timesteps = 15
n_visible = 2 * note_range * num_timesteps
n_hidden = 50

num_epochs = 200
batch_size = 100

lr = tf.constant(0.005,tf.float32)

### TF variables

x = tf.placeholder(tf.float32,[None,n_visible],name="x")

W = tf.Variable(tf.random_normal([n_visible,n_hidden],0.01),name="W")

bh = tf.Variable(tf.zeros([1,n_hidden],tf.float32,name="bh"))

bv = tf.Variable(tf.zeros([1,n_visible],tf.float32,name="bv"))

x_sample = gibbs_sample(1)

h = sample(tf.sigmoid(tf.matmul(x,W)+bh))

h_sample = sample(tf.sigmoid(tf.matmul(x_sample,W) + bh))

size_bt = tf.case(tf.shape(x)[0],tf.float32)
W_addr = tf.mul