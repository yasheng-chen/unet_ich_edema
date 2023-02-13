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
author: jakeret
'''
import numpy as np
import nibabel as nib
from random import randint
import scipy.ndimage

class MriDataProvider:
    channels = 1
    n_class = 3
    
    def __init__(self, in_size, **kwargs):
        super(MriDataProvider, self).__init__()
        self.in_size = in_size
        self.kwargs = kwargs
        self.clear_data()

    def load_data(self, dataDir, splits, normalization='none', input_mode='valid'):
        for i in range(len(splits[0])):
            X_temp, Y_temp = self.load_flair_subjects(dataDir, splits[0][i], normalization=normalization, input_mode=input_mode)
            print(splits[0][i])
            print(X_temp.shape, Y_temp.shape)
            print(" ")
            self.X = np.concatenate([self.X, X_temp],axis=0)
            self.Y = np.concatenate([self.Y, Y_temp],axis=0)

        for i in range(len(splits[1])):
            X_temp, Y_temp = self.load_flair_subjects(dataDir, splits[1][i], normalization=normalization, input_mode='full')
            print(splits[1][i])
            print(X_temp.shape, Y_temp.shape)
            print(" ")
            self.X_test = np.concatenate([self.X_test, X_temp],axis=0)
            self.Y_test = np.concatenate([self.Y_test, Y_temp],axis=0)
    
        for i in range(len(splits[2])):
            X_temp, Y_temp = self.load_flair_subjects(dataDir, splits[2][i])
            print(splits[2][i])
            print(X_temp.shape, Y_temp.shape)
            print(" ")
            self.X_monitor = np.concatenate([self.X_monitor, X_temp],axis=0)
            self.Y_monitor = np.concatenate([self.Y_monitor, Y_temp],axis=0)

    def load_test_data(self, dataDir, splits, normalization='none'):
        for i in range(len(splits[1])):
            X_temp, Y_temp = self.load_flair_subjects(dataDir, splits[1][i], input_mode="full", normalization=normalization)
            print(splits[1][i])
            print(X_temp.shape, Y_temp.shape)
            print(" ")
            self.X_test = np.concatenate([self.X_test, X_temp],axis=0)
            self.Y_test = np.concatenate([self.Y_test, Y_temp],axis=0)

    def load_test_subject(self, dataDir, subj_name, normalization='none', input_mode="full"):
        X_temp, Y_temp = self.load_flair_subjects(dataDir, subj_name)
        print(subj_name)
        print(X_temp.shape, Y_temp.shape)
        print(" ")
        self.X_test = np.concatenate([self.X_test, X_temp],axis=0)
        self.Y_test = np.concatenate([self.Y_test, Y_temp],axis=0)

    def load_predict_subject(self, dataDir, subj_name, tag="", normalization='none', input_mode="full"):
        X_temp = self.load_flair_subjects_for_predict(dataDir, subj_name, tag=tag)
        #print(subj_name)
        #print(X_temp.shape)
        #print(" ")
        self.X_test = np.concatenate([self.X_test, X_temp],axis=0)

    def get_monitor_data(self):
        return self.X_monitor, self.convert_y(self.Y_monitor)

    def get_test_data(self):
        return self.X_test, self.convert_y(self.Y_test)

    def get_train_data(self):
        return self.X, self.convert_y(self.Y)

    def clear_data(self):
        self.X = np.empty((0,self.in_size,self.in_size,1),dtype=np.float32)
        self.Y = np.empty((0,self.in_size,self.in_size,1),dtype=np.bool)

        self.X_test = np.empty((0,self.in_size,self.in_size,1),dtype=np.float32)
        self.Y_test = np.empty((0,self.in_size,self.in_size,1),dtype=np.bool)

        self.X_monitor = np.empty((0,self.in_size,self.in_size,1),dtype=np.float32)
        self.Y_monitor = np.empty((0,self.in_size,self.in_size,1),dtype=np.bool)
        
	
    def load_flair_subjects(self, dataDir, s_xxx, input_mode="valid", normalization="none"):
        # load nii
        nii_image = nib.load(dataDir + '/' + s_xxx + "_normalized.nii.gz")
        nii_label = nib.load(dataDir + '/' + s_xxx + "_seg.nii.gz")

        n = nii_image.shape[2]
        if n != nii_label.shape[2]:
            print("Waring: Input and label have different number of slices")

        nii_size = nii_image.shape[0]

        X = np.empty((0, self.in_size, self.in_size, 1))
        Y = np.empty((0, self.in_size, self.in_size, 1))

        X_list = [X]
        Y_list = [Y]

        margin = (self.in_size-nii_size)//2
        #print('insize--------------------------------')	
        #print(self.in_size)
        #print('nii_size-------------------------------')
        #print(nii_size)
        #print(margin)
        #print(np.amax(nii_image.dataobj))
        #print(np.amax(nii_label.dataobj))
       	
        for i in range(n):
            image_slice = np.zeros((self.in_size, self.in_size))
            label_slice = np.zeros((self.in_size, self.in_size))

            image_slice[margin:self.in_size-margin, margin:self.in_size-margin] = nii_image.dataobj[:,:,i]
            label_slice[margin:self.in_size-margin, margin:self.in_size-margin] = nii_label.dataobj[:,:,i]

            can_normalize = True
            if input_mode != 'full' and np.min(image_slice) == np.max(image_slice):
                continue

            #label_slice = (label_slice==1.).astype(np.float32)
            wh_volume = np.sum(label_slice)

            #print('input_mode is %s'%(input_mode)), input_mode is valid

            if input_mode == 'positive' and wh_volume <=0:
                continue

            X_slice = np.zeros(shape=(1, self.in_size, self.in_size, 1))
            # reshape input data
            X_slice[0,:,:,0] =  image_slice
            X_list.append(X_slice)

            Y_slice = np.zeros(shape=(1, self.in_size, self.in_size, 1))
            Y_slice[0,:,:,0] = label_slice
            # Y_slice[0,:,:,1] = label_slice
            Y_list.append(Y_slice)

        X_ret = np.concatenate(X_list)
        Y_ret = np.concatenate(Y_list)

        if normalization != "none":
            hi = np.amax(X_ret)
            lo = np.amin(X_ret)
            X_ret = (X_ret-lo)*(1/(hi - lo))

        #print(X_ret.size)
        #print(Y_ret.size)

        return [X_ret, Y_ret]

    def load_flair_subjects_for_predict(self, dataDir, s_xxx, tag="", input_mode="valid", normalization="none"):
        # load nii
        nii_image = nib.load(dataDir + '/' + s_xxx + ".nii.gz")

        #nii_image = nib.load(dataDir + '/' + s_xxx + tag)
        n = nii_image.shape[2]
        nii_size = nii_image.shape[0]
        X = np.empty((0, self.in_size, self.in_size, 1))
        X_list = [X]
        margin = (self.in_size-nii_size)//2
        for i in range(n):
            image_slice = np.zeros((self.in_size, self.in_size))
            image_slice[margin:self.in_size-margin, margin:self.in_size-margin] = nii_image.dataobj[:,:,i]
        
            X_slice = np.zeros(shape=(1, self.in_size, self.in_size, 1))
            # reshape input data
            X_slice[0,:,:,0] =  image_slice
            X_list.append(X_slice)
        X_ret = np.concatenate(X_list)

        # if normalization != "none":
        #     hi = np.amax(X_ret)
        #     lo = np.amin(X_ret)
        #     X_ret = (X_ret-lo)*(1/(hi - lo))

        return X_ret

    def convert_y(self, y_single):
        #y_double = np.concatenate([1-y_single, y_single],axis=-1).astype(np.float32)
        y_double=np.concatenate([y_single==0, y_single==1, y_single==2], axis=-1).astype(np.float32)
        return y_double

    def get_number_slices(self):
        return np.shape(self.X)


def get_volume(Y):
    rs = list()
    for i in range(Y.shape[0]):
        label_slice = Y[i,:,:,1]
        rs.append(np.sum(label_slice))
    return rs

def read_full_subject_list(dir):
    rs = list()
    f = open(dir, 'r')
    line = f.readline()
    while line != None and line != '':
        rs.append(line.strip(' \t\n\r'))
        line = f.readline()
    return rs

def read_manual_splits(settingFile):
    f = open(settingFile, 'r')
    f.readline() # [Inputs]
    trainSubjects = []
    line = f.readline().strip(' \t\n\r')
    while line != '[Test]':
        if line != '':
            trainSubjects.append(line)
        line = f.readline().strip(' \t\n\r')

    line = f.readline().strip(' \t\n\r')
    testSubjects = []
    while line != '[Monitoring]':
        if line != '':
            testSubjects.append(line)
        line = f.readline().strip(' \t\n\r')

    line = f.readline().strip(' \t\n\r')
    monitorSubjects = []
    while line != '[End]':
        if line != '':
            monitorSubjects.append(line)
        line = f.readline().strip(' \t\n\r')
    print([trainSubjects, testSubjects, monitorSubjects])
    return [trainSubjects, testSubjects, monitorSubjects]


def read_datahome(settingFile):
    f = open(settingFile, 'r')
    f.readline() # [Data Home]
    dataHome = f.readline().strip(' \t\n\r')
    f.close()
    return dataHome

def save_NWH_to_nifty(NWH_tensor, output_dir, base, out_l=512):
    l = NWH_tensor.shape[1]
    border = (out_l - l)//2
    
    NWH_tensor_expanded = np.zeros((NWH_tensor.shape[0], l + 2*border, l + 2*border))
    if base != 0:
        NWH_tensor_expanded = base*np.ones((NWH_tensor.shape[0], l + 2*border, l + 2*border))
    NWH_tensor_expanded[:,border:out_l-border,border:out_l-border] = NWH_tensor[:,:,:]
    rolled = np.rollaxis(NWH_tensor_expanded, 0, 3)
    img = nib.Nifti1Image(rolled, np.eye(4))
    nib.save(img, output_dir)

def save_NWH_to_nifty_v2(NWH_tensor, output_dir, base, out_l=256):
    l = NWH_tensor.shape[1]
    border = (out_l - l)//2
    NWH_tensor_expanded = np.zeros((NWH_tensor.shape[0], l + 2*border, l + 2*border))
    if base != 0:
        NWH_tensor_expanded = base*np.ones((NWH_tensor.shape[0], l + 2*border, l + 2*border))
    NWH_tensor_expanded[:,border:out_l-border,border:out_l-border] = NWH_tensor[:,:,:]
    rolled = np.moveaxis(NWH_tensor_expanded, 0, 2)
    img = nib.Nifti1Image(rolled, np.eye(4))
    nib.save(img, output_dir)

def patch_extractor_2d(image_slice, label_slice, patch_shape, margin):
    patch_span = np.array(image_slice.shape)//patch_shape
    print(patch_span)
    
    expanded_shape = np.array(image_slice.shape) + 2*margin
    expanded_image = np.zeros(expanded_shape)
    expanded_image[margin:margin+image_slice.shape[0], margin:margin+image_slice.shape[1]] = image_slice

    expanded_label = np.zeros(expanded_shape)
    expanded_label[margin:margin+label_slice.shape[0], margin:margin+label_slice.shape[1]] = label_slice

    print(expanded_shape)
    image_patches=[]
    label_patches=[]
    for i in range(patch_span[0]):
        for j in range(patch_span[1]):
            start_loc = [margin+i*patch_shape[0], margin+j*patch_shape[1]]

            image_patches.append(expanded_image[start_loc[0]-margin:start_loc[0]+margin+patch_shape[0], start_loc[1]-margin:start_loc[1]+margin+patch_shape[1]])
            label_patches.append(expanded_label[start_loc[0]:start_loc[0]+patch_shape[0], start_loc[1]:start_loc[1]+patch_shape[1]])
        
    return image_patches, label_patches, patch_span

def image_reshape(input, tgt_shape):
    #crop
    expanded = np.zeros(np.amax([input.shape, tgt_shape], axis=0))
    margin = np.amax([(np.array(input.shape) - np.array(tgt_shape))//2, np.zeros(shape=len(tgt_shape),dtype=int)], axis=0)
    cropped = input[margin[0]:margin[0]+tgt_shape[0], margin[1]:margin[1]+tgt_shape[1]]

    #expand
    output = np.zeros(tgt_shape)
    margin = (np.array(tgt_shape) - np.array(cropped.shape))//2
    output[margin[0]:margin[0]+cropped.shape[0], margin[1]:margin[1]+cropped.shape[1]] = cropped

    return output


def combine_patch(patches_list, patch_span, tgt_shape):
    patch_shape = patches_list[0].shape
    eff_shape = patch_shape*patch_span
    margin = eff_shape - tgt_shape

    out = np.zeros(shape=tgt_shape+margin)

    for i in range(len(patches_list)):
        loc = np.array([i//patch_span[0], i%patch_span[0]])
        offset = margin + patch_shape*loc
        out[offset[0]:offset[0]+patch_shape[0], offset[1]:offset[1]+patch_shape[1]] = patches_list[i]

    return out
