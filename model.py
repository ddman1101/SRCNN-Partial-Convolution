# -*- coding: UTF-8 -*-

from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  imread,
  merge2,
  psnr,
  psnr2
)

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob

import feature_map
import csv
import partial_conv as pc

try:
  xrange
except:
  xrange = range

class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=33,
               label_size=33, 
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    
    self.weights = {
      'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3, seed = 12345), name='w1'),
      'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3, seed = 54321), name='w2'),
      'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3, seed = 11111), name='w3')
    }
    self.biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
#    a = tf.zeros([17,17,self.c_dim],dtype=tf.dtypes.float32)
#    self.labels2 = self.labels
#    self.pred2 = self.pred
#    self.labels2[-1,8:25,8:25,self.c_dim] = a
#    self.pred2[-1,8:25,8:25,self.c_dim] = a
#
#    self.loss2 = tf.reduce_mean(tf.square(self.labels2 - self.pred2))
        
    self.saver = tf.train.Saver()
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config)
    else:
      nx, ny, h, w, h_, w_, temp_h2, temp_w2 = input_setup(self.sess, config)

    if config.is_train:     
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5_test")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5_test")

    train_data, train_label = read_data(data_dir)

    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
#    self.train_op2 = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss2)
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))
            loss_ = []
            loss_.append(err)
          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)
        
        with open('loss_residual.csv','a',newline='') as csvfile:
            writer = csv.writer(csvfile)
            np.array(loss_)
            writer.writerow([float(np.mean(loss_))])

    else:
      print("Testing...")
      visualize_layers = ['conv1','conv2','conv3']
      conv_out = self.sess.run(tf.get_collection('activations'), feed_dict={self.images: train_data, self.labels: train_label})
            
      result = self.pred.eval({self.images: train_data, self.labels: train_label})
      
#      print(merge(train_data,[nx,ny]).shape)
#      print(merge(train_label,[nx,ny]).shape)
#      print(len(result))
#      print(np.shape(result))
#      print(len(conv_out))
#      print(type(conv_out))
#      print(np.shape(conv_out[0]))
#      print(np.shape(conv_out[1]))
#      print(np.shape(conv_out[2]))
      result = merge(result, [nx, ny])
      result = result.squeeze()
      result_ = result[temp_h2:h + temp_h2 ,temp_w2: w + temp_w2]
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "set5-15000-0.png")
      imsave(result_, image_path)
      test = merge(train_label, [nx, ny])
      test_ = test[temp_h2:h + temp_h2 ,temp_w2: w + temp_w2]
      print('PSNR : {}'.format(psnr(test_, result_)))
      with open('psnr-set5.csv', 'a', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow([float(psnr(test_, result_))])
          
      print('result_:{}'.format(np.shape(result_)))
      print('test_:{}'.format(np.shape(test_)))
      print(result_)      
#      feature = []
#      feature2 = []
#      feature3 = []
#      feature_all = []
#      for number_1 in range(len(conv_out[0][0,0,0,:])):
#          feature.append((merge2(conv_out[0][:,:,:,number_1],[nx,ny])))
#      for number_2 in range(len(conv_out[1][0,0,0,:])):
#          feature2.append((merge2(conv_out[1][:,:,:,number_2],[nx,ny])))
#      for number_3 in range(len(conv_out[2][0,0,0,:])):
#          feature3.append((merge2(conv_out[2][:,:,:,number_3],[nx,ny])))
#      feature_all.append(feature)
#      feature_all.append(feature2)
#      feature_all.append(feature3)
#      for i, layer in enumerate(visualize_layers):
#            plot_dir=r'C:\Users\GL63\Desktop\paper\SRCNN-Tensorflow-master-trial3\SRCNN-Tensorflow-master\feature_map_15000-'#要保存的路径
#            if not os.path.exists(plot_dir+layer):#如果路径不存在，则创建文件夹
#                os.mkdir(plot_dir+layer)
#            for l in range(len(feature_all[i])):#保存为图片
#                feature_map.plot_conv_output(feature_all[i], plot_dir + layer, str(l), filters_all=False, filters=[l])

      
  def model(self):
    conv1 = tf.nn.relu(pc.partial_conv(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
    conv2 = tf.nn.relu(pc.partial_conv(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
    conv3 = pc.partial_conv(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']
    tf.add_to_collection('activations', conv1)
    tf.add_to_collection('activations', conv2)
    tf.add_to_collection('activations', conv3)
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
