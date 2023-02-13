# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import os
import numpy as np
from collections import OrderedDict
import tensorflow as tf

def unet(x, keep_prob, channels, n_class, layers=4, features_root=16, filter_size=3, 
                    pool_size=2, summaries=False, batch_norm=True, is_training=True, in_size=512):

    print("Creating normal unet...")
    with tf.variable_scope('unet', reuse=(not is_training)): 
        x_image = tf.reshape(x, tf.stack([-1,in_size,in_size,channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]
 
        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
    
        size = in_size
        # down layers
        for layer in range(0, layers):
            with tf.variable_scope('down_layer_' + str(layer)): 
                features = 2**layer*features_root
                stddev = np.sqrt(2 / (filter_size**2 * features))
                if layer == 0:
                    w1 = W_new(name='w1', shape=[filter_size, filter_size, channels, features], stddev=stddev)
                else:
                    w1 = W_new(name='w1', shape=[filter_size, filter_size, features//2, features], stddev=stddev)
            
                w2 = W_new(name='w2', shape=[filter_size, filter_size, features, features], stddev=stddev)
                b1 = b_new(name='b1', shape=[features])
                b2 = b_new(name='b2', shape=[features])
            
                ###conv1 = conv2d(in_node, w1, keep_prob)
                print("============================================================================================")
                print("layer %d, in_node, w1, w2 dw_h_conv pool in_node_end"%(layer))

                conv1 = tf.nn.conv2d(in_node, w1, strides=[1, 1, 1, 1], padding='SAME')
                tmp_h_conv = tf.nn.relu(conv1 + b1)
                ###conv2 = conv2d(tmp_h_conv, w2, keep_prob)
                conv2 = tf.nn.conv2d(tmp_h_conv, w2, strides=[1,1,1,1], padding='SAME')
                dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

                print("down layer %d\n"%(layer))
                print(in_node.get_shape())
                print(w1.get_shape())
                print(w2.get_shape())
                print(dw_h_convs[layer].get_shape())
            
                weights.append((w1, w2))
                biases.append((b1, b2))
                convs.append((conv1, conv2))
            
                ###size -= 4
                if layer < layers-1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    print(pools[layer].get_shape())
                    in_node = pools[layer]
                    print(in_node.get_shape())
                    size /= 2
            
        in_node = dw_h_convs[layers-1]
        print("before up in_node")        
        print(in_node.get_shape())
        # up layers
        for layer in range(layers-2, -1, -1):
            with tf.variable_scope('up_layer_' + str(layer)): 
                features = 2**(layer+1)*features_root
                stddev = np.sqrt(2 / (filter_size**2 * features))

                print("==================================================")
                print("uplayer %d in node deconv_z dw_h_conv h_dconv_concat up_h_conv"%(layer))
                print(in_node.get_shape()) 
		           
                deconv_z = tf.contrib.layers.convolution2d_transpose(in_node, 
                                                                    features//2, 
                                                                    kernel_size=2, #kernel_size=[pool_size, pool_size],
                                                                    stride=[pool_size, pool_size],
                                                                    padding='SAME',
                                                                    activation_fn=None,
                                                                    scope='deconv')
                print(deconv_z.get_shape())
                print(dw_h_convs[layer].get_shape())

                h_deconv = tf.nn.relu(deconv_z)
                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)

                print(h_deconv_concat.get_shape())
                deconv[layer] = h_deconv_concat

                print(deconv[layer].get_shape())

                w1 = W_new(name='w1', shape=[filter_size, filter_size, features, features//2], stddev=stddev)
                w2 = W_new(name='w2', shape=[filter_size, filter_size, features//2, features//2], stddev=stddev)
                b1 = b_new(name='b1', shape=[features//2])
                b2 = b_new(name='b2', shape=[features//2])
            
                ###conv1 = conv2d(h_deconv_concat, w1, keep_prob)
                conv1 = tf.nn.conv2d(h_deconv_concat, w1, strides=[1,1,1,1], padding='SAME')
                h_conv = tf.nn.relu(conv1 + b1)
                ###conv2 = conv2d(h_conv, w2, keep_prob)
                conv2 = tf.nn.conv2d(h_conv, w2, strides=[1,1,1,1], padding='SAME')
                in_node = tf.nn.relu(conv2 + b2)
                up_h_convs[layer] = in_node

                print(up_h_convs[layer].get_shape())
                weights.append((w1, w2))
                biases.append((b1, b2))
                convs.append((conv1, conv2))
            
                size *= 2
                ###size -= 4


        # Output Map
        with tf.variable_scope('output_layer_' + str(layer)): 
            weight = W_new(name='weight', shape=[1, 1, features_root, n_class], stddev=stddev)
            bias = b_new(name='bias', shape=[n_class])
            ###conv = conv2d(in_node, weight, tf.constant(1.0))
            conv = tf.nn.conv2d(in_node, weight, strides=[1,1,1,1], padding='SAME')

            output_map = tf.nn.relu(conv + bias)
            up_h_convs["out"] = output_map

    
    print("out....")            
    print(output_map.get_shape())
    return output_map


def unet_bn_old(x, keep_prob, channels, n_class, layers=3, 
                    features_root=16, filter_size=3, pool_size=2, 
                    summaries=False, batch_norm=True, is_training=True, in_size=256):

    print("Creating unet with batch normalization")
    print("Features Root: %d"%(features_root))
    # Placeholder for the input image
    #nx = tf.shape(x)[1]
    #ny = tf.shape(x)[2]
    with tf.variable_scope('unet_bn', reuse=(not is_training)): 
        x_image = tf.reshape(x, tf.stack([-1,in_size,in_size,channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]
    
        weights = []
        biases = []
        convs = []

        BN_trainable = []
        BN_manual = []

        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
        
        size = in_size
        # down layers
        for layer in range(0, layers):
            with tf.variable_scope('down_layer_' + str(layer)): 
                features = 2**layer*features_root
                stddev = np.sqrt(2 / (filter_size**2 * features))
            
                if layer == 0:
                    w1 = W_new(name='w1', shape=[filter_size, filter_size, channels, features], stddev=stddev)
                else:
                    w1 = W_new(name='w1', shape=[filter_size, filter_size, features//2, features], stddev=stddev)
                
                w2 = W_new(name='w2', shape=[filter_size, filter_size, features, features], stddev=stddev)
            
                conv1 = conv2d(in_node, w1, keep_prob)
                conv1_BN = tf.contrib.layers.batch_norm(inputs=conv1, scale=True, is_training=is_training, updates_collections=None, scope="BN1")
                tmp_h_conv = tf.nn.relu(conv1_BN)

                conv2 = conv2d(tmp_h_conv, w2, keep_prob)
                conv2_BN = tf.contrib.layers.batch_norm(inputs=conv2, scale=True, is_training=is_training, updates_collections=None, scope="BN2")
                dw_h_convs[layer] = tf.nn.relu(conv2_BN)

                weights.append(w1)
                weights.append(w2)
                convs.append((conv1, conv2))
            
                size -= 4
                if layer < layers-1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]
                    size /= 2

        in_node = dw_h_convs[layers-1]
            
        # up layers
        for layer in range(layers-2, -1, -1):
            with tf.variable_scope('up_layer_' + str(layer)): 
                features = 2**(layer+1)*features_root
                stddev = np.sqrt(2 / (filter_size**2 * features))

                deconv_z = tf.contrib.layers.convolution2d_transpose(in_node, 
                                                                features//2, 
                                                                kernel_size=[pool_size, pool_size],
                                                                stride=[pool_size, pool_size],
                                                                padding='VALID',
                                                                activation_fn=None,
                                                                scope='deconv')
                
                deconv_BN = tf.contrib.layers.batch_norm(inputs=deconv_z, scale=True, is_training=is_training, updates_collections=None, scope="BNd")
                h_deconv = tf.nn.relu(deconv_BN)

                # h_deconv = tf.contrib.layers.convolution2d_transpose(in_node, 
                #                                         features//2, 
                #                                         kernel_size=[pool_size, pool_size],
                #                                         stride=[pool_size, pool_size],
                #                                         padding='VALID',
                #                                         activation_fn=tf.nn.relu,
                #                                         normalizer_fn=tf.contrib.layers.batch_norm,
                #                                         normalizer_params={"scale":True, "is_training":is_training, "updates_collections":None, "scope":"BNd"},
                #                                         scope='deconv')

                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                deconv[layer] = h_deconv_concat
            
                w1 = W_new(name='w1', shape=[filter_size, filter_size, features, features//2], stddev=stddev)
                w2 = W_new(name='w2', shape=[filter_size, filter_size, features//2, features//2], stddev=stddev)
            
                conv1 = conv2d(h_deconv_concat, w1, keep_prob)
                conv1_BN = tf.contrib.layers.batch_norm(inputs=conv1, scale=True, is_training=is_training, updates_collections=None, scope="BN1d")
                h_conv = tf.nn.relu(conv1_BN)

                conv2 = conv2d(h_conv, w2, keep_prob)
                conv2_BN = tf.contrib.layers.batch_norm(inputs=conv2, scale=True, is_training=is_training, updates_collections=None, scope="BN2d")
                in_node = tf.nn.relu(conv2_BN)
                up_h_convs[layer] = in_node

                weights.append(w1)
                weights.append(w2)
                convs.append((conv1, conv2))
            
                size *= 2
                size -= 4

        # Output Map
        with tf.variable_scope('output_layer_' + str(layer)): 
            weight = W_new(name='weight', shape=[1, 1, features_root, n_class], stddev=stddev)
            bias = b_new(name='bias', shape=[n_class])
            conv = conv2d(in_node, weight, tf.constant(1.0))
            output_map = tf.nn.relu(conv + bias)
            up_h_convs["out"] = output_map
    
    return output_map

def unet_bn(x, keep_prob, channels, n_class, layers=3, 
                    features_root=16, filter_size=3, pool_size=2, 
                    summaries=False, batch_norm=True, is_training=True, in_size=256):

    print("Creating unet with batch normalization v2")
    print("Features Root: %d"%(features_root))
    # Placeholder for the input image
    #nx = tf.shape(x)[1]
    #ny = tf.shape(x)[2]
    with tf.variable_scope('unet_bn', reuse=(not is_training)): 
        x_image = tf.reshape(x, tf.stack([-1,in_size,in_size,channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]
    
        weights = []
        biases = []
        convs = []

        BN_trainable = []
        BN_manual = []

        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()
        
        size = in_size
        # down layers
        for layer in range(0, layers):
            with tf.variable_scope('down_layer_' + str(layer)): 
                features = 2**layer*features_root
                stddev = np.sqrt(2 / (filter_size**2 * features))
            
                if layer == 0:
                    w1 = W_new(name='w1', shape=[filter_size, filter_size, channels, features], stddev=stddev)
                else:
                    w1 = W_new(name='w1', shape=[filter_size, filter_size, features//2, features], stddev=stddev)
                
                w2 = W_new(name='w2', shape=[filter_size, filter_size, features, features], stddev=stddev)
            
                conv1 = conv2d(in_node, w1, keep_prob)
                conv1_BN = tf.layers.batch_normalization(inputs=conv1, training=is_training, name="BN1")
                tmp_h_conv = tf.nn.relu(conv1_BN)

                conv2 = conv2d(tmp_h_conv, w2, keep_prob)
                conv2_BN = tf.layers.batch_normalization(inputs=conv2, training=is_training, name="BN2")
                dw_h_convs[layer] = tf.nn.relu(conv2_BN)

                weights.append(w1)
                weights.append(w2)
                convs.append((conv1, conv2))
            
                size -= 4
                if layer < layers-1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]
                    size /= 2

        in_node = dw_h_convs[layers-1]
            
        # up layers
        for layer in range(layers-2, -1, -1):
            with tf.variable_scope('up_layer_' + str(layer)): 
                features = 2**(layer+1)*features_root
                stddev = np.sqrt(2 / (filter_size**2 * features))
                wd = W_new(name='wd', shape=[pool_size, pool_size, features//2, features], stddev=stddev)
                # bd = bias_variable([features//2])
                deconv_z = deconv2d(in_node, wd, pool_size)
                # deconv_z = tf.contrib.layers.convolution2d_transpose(in_node, 
                #                                                 features//2, 
                #                                                 kernel_size=[pool_size, pool_size],
                #                                                 stride=[pool_size, pool_size],
                #                                                 padding='VALID',
                #                                                 activation_fn=None,
                #                                                 scope='deconv')
                
                deconv_BN = tf.layers.batch_normalization(inputs=deconv_z, training=is_training, name="BNd")
                h_deconv = tf.nn.relu(deconv_BN)

                # h_deconv = tf.contrib.layers.convolution2d_transpose(in_node, 
                #                                         features//2, 
                #                                         kernel_size=[pool_size, pool_size],
                #                                         stride=[pool_size, pool_size],
                #                                         padding='VALID',
                #                                         activation_fn=tf.nn.relu,
                #                                         normalizer_fn=tf.contrib.layers.batch_norm,
                #                                         normalizer_params={"scale":True, "is_training":is_training, "updates_collections":None, "scope":"BNd"},
                #                                         scope='deconv')

                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                deconv[layer] = h_deconv_concat
            
                w1 = W_new(name='w1', shape=[filter_size, filter_size, features, features//2], stddev=stddev)
                w2 = W_new(name='w2', shape=[filter_size, filter_size, features//2, features//2], stddev=stddev)
            
                conv1 = conv2d(h_deconv_concat, w1, keep_prob)
                conv1_BN = tf.layers.batch_normalization(inputs=conv1, training=is_training, name="BN1d")
                h_conv = tf.nn.relu(conv1_BN)

                conv2 = conv2d(h_conv, w2, keep_prob)
                conv2_BN = tf.layers.batch_normalization(inputs=conv2, training=is_training, name="BN2d")
                in_node = tf.nn.relu(conv2_BN)
                up_h_convs[layer] = in_node

                weights.append(w1)
                weights.append(w2)
                convs.append((conv1, conv2))
            
                size *= 2
                size -= 4

        # Output Map
        with tf.variable_scope('output_layer_' + str(layer)): 
            weight = W_new(name='weight', shape=[1, 1, features_root, n_class], stddev=stddev)
            # bias = b_new(name='bias', shape=[n_class])
            conv = conv2d(in_node, weight, tf.constant(1.0))
            conv_BN = tf.layers.batch_normalization(inputs=conv, training=is_training, name="BNout")
            output_map = tf.nn.relu(conv_BN)
            up_h_convs["out"] = output_map
    
    return output_map

def weight_variable(shape, stddev=0.1, trainable=True):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, trainable=trainable)

def weight_variable_devonc(shape, stddev=0.1, trainable=True):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), trainable=trainable)

# calculation functions
def W_new(name, shape, stddev=0.1, trainable=True):
    ret = tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=stddev), trainable=trainable)
    return ret

def b_new(name, shape, trainable=True):
    ret =  tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.1), trainable=trainable)
    return ret

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True)

def conv2d(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W, stride):
    x_shape = x.get_shape().as_list()
    # output_shape = tf.stack([None, x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    out = tf.nn.conv2d_transpose(x, W, output_shape=[tf.shape(x)[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2], strides=[1, stride, stride, 1], padding='VALID')
    return out

def get_deconv2d_shape(x, W, stride):
    x_shape = x.get_shape().as_list()
    x_shape[0] = 20
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID').get_shape().as_list()

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    
    return tf.concat([x1_crop, x2], 3)   

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)



def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
