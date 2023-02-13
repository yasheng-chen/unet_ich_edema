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
Created on Aug 10, 2016

author: jakeret
'''
import numpy as np
import scipy

# directories

settingFile = '../settings.txt'

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= (np.amax(img)+1e-10)
    img *= 255
    return img

def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(data.shape[1]-offset0), offset1:(data.shape[2]-offset1)]

def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image
    
    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)), 
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img

def combine_img_prediction_v2(data, gt, pred, mode='soft', weights=[1,1]):
    """
    Combines the data, grouth thruth and the prediction into one rgb image
    
    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    pred[...,0] = pred[...,0]*weights[0]
    pred[...,1] = pred[...,1]*weights[1]
    
    if mode=='hard':
        pred[...,1] = np.argmax(pred, 3)

    if mode=='diff':
        pred[...,1] = pred[...,1] - pred[...,0]
    
    ny = pred.shape[2]
    ch = data.shape[3]
    img = None

    if gt is None:
        img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)), 
                            to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    else:
        img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)), 
                            to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
                            to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)

    return img

def save_image(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    # Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)
    scipy.misc.imsave(path, img.round().astype(np.uint8))

def unet_dim_calculator(min_size, n_layers):
    input_size = min_size
    output_size = min_size 

    for i in range(n_layers-1):
        input_size = 2*(input_size + 4)
        output_size = 2*output_size - 4

    input_size += 4

    return input_size, output_size

def unet_dim_calculator_v2(pred_size, n_layers):

    # get min dimension
    min_size = pred_size
    for i in range(n_layers-1):
        min_size = (min_size + 4)//2
        
    # get fixed output dimension
    true_pred_size = min_size
    for i in range(n_layers-1):
        true_pred_size = 2*true_pred_size - 4

    input_size = min_size + 4
    for i in range(n_layers-1):
        input_size = 2*input_size+4

    return input_size, true_pred_size, min_size

def unet_dim_calculator_from_input_layer(in_size, n_layers):

    # get min dimension
    out_size = in_size
    for i in range(0, n_layers):
        out_size = out_size-4
	
        if i<n_layers-1:
           out_size /= 2

    min_size = out_size
        
    # get fixed output dimension
    for i in range(n_layers-2, -1, -1):
        out_size = 2*out_size - 4

    return min_size, out_size
