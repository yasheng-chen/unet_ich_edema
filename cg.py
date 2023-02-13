
import os
import shutil
import numpy as np
import tensorflow as tf
import time
import threading
import util
import math
from random import randint
import nibabel as nib

class ComputingGraph(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """
    
    def __init__(self, channels=3, n_class=3, cost="cross_entropy", network="unet", in_size=256, out_size=216, cost_kwargs={}, **kwargs):
        if network == "unet":
            from networks import unet as create_conv_net
        elif network == "unet_bn":
            from networks import unet_bn as create_conv_net
        
        tf.reset_default_graph()
        
        self.n_class = n_class
        self.in_size = in_size
        self.out_size = out_size
        self.summaries = kwargs.get("summaries", True)

        self.x_raw = tf.placeholder("float", shape=[None, None, None, channels], name="x_raw")
        self.y_raw = tf.placeholder("float", shape=[None, None, None, n_class], name="y_raw")
        self.rot_ang = tf.placeholder("float", shape=[])
        
        self.x = tf.contrib.image.rotate(self.x_raw, self.rot_ang)
        self.y = tf.contrib.image.rotate(self.y_raw, self.rot_ang)
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        self.class_weights = tf.placeholder(tf.float32, shape=[n_class]) #dropout (keep probability)
        
        logits = create_conv_net(self.x, self.keep_prob, channels, n_class, is_training=True, in_size=self.in_size, **kwargs)
        logits_te = create_conv_net(self.x, self.keep_prob, channels, n_class, is_training=False, in_size=self.in_size, **kwargs)

        #self.cost = self._get_cost(logits, cost, self.class_weights, cost_kwargs)
        #self.cost_te = self._get_cost(logits_te, cost, self.class_weights, cost_kwargs)
        
        self.predicter = tf.contrib.layers.softmax(logits)
        self.predicter_te = tf.contrib.layers.softmax(logits_te)

        self.raw_logits = logits
        self.raw_logits_te = logits_te
        self.f_pred = tf.reshape(tf.argmax(self.predicter_te, 3),[-1])
        self.f_true = tf.reshape(tf.argmax(self.y, 3),[-1])
        self.correct_pred = tf.equal(self.f_pred, self.f_true)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.tpp = tf.reduce_sum(tf.cast(tf.equal(tf.add(self.f_pred, self.f_true),2), tf.float32))
        self.pp = tf.reduce_sum(tf.cast(tf.equal(self.f_pred, 1), tf.float32))
        self.tp = tf.reduce_sum(tf.cast(tf.equal(self.f_true, 1), tf.float32))

        self.dice = 2*self.tpp / (self.tp + self.pp + 1e-6)

        self.cost_name = cost

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            prediction = sess.run(self.predicter_te, feed_dict={self.x: x_test, self.keep_prob: 1.})
            
        return prediction

    def predict_big(self, model_path, x_test, batchSize=50):
        print('predict_big model %s'%(model_path))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            iStart = 0

            pred_list = list()

            while iStart < x_test.shape[0]:
                iEnd = iStart + batchSize
                if iEnd >= x_test.shape[0] or batchSize==0:iEnd = x_test.shape[0]
                crt_pred = sess.run(self.predicter_te, 
                                        feed_dict={self.x: x_test[iStart:iEnd,...], self.keep_prob: 1.})
                
                iStart = iEnd
                pred_list.append(crt_pred)
            prediction = np.concatenate(pred_list)
        return prediction
    
    def save(self, sess, model_path, global_step=None):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver(tf.global_variables())
        save_path = saver.save(sess, model_path, global_step)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

def record_summary(writer, name, value, epoch):
    summary=tf.Summary()
    summary.value.add(tag=name, simple_value = value)
    writer.add_summary(summary, epoch)
    writer.flush()

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

