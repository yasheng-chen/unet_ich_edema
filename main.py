
from __future__ import print_function, division, absolute_import, unicode_literals

import IO as io
import os, shutil, sys, getopt

# import unet_new as unet
# import unet_bn_nobias_built_in as unet
from cg import ComputingGraph
import util
import tensorflow as tf

from IO import MriDataProvider
import time

import nibabel as nib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# paths
#dataHome = io.read_datahome('settings/paths.txt')
dataHome = '/'
dataDir = dataHome + '/images_input'
grpDefFile = dataDir + '/names_for_testing_in_different_groups.txt'
model_in_path = None
manual_splits = None

# gather info
mode = None
group = 0
outputdir = dataHome + '/prediction'
device = '1'

# parameters
training_iters = 20
epochs = 2000
dropout = 0.75 # Dropout, probability to keep units
display_step = 10

batch_size = 10
loss_func = "cross_entropy"
#loss_func = "dice"
optimizer = "adam"
learning_rate = 0.001

network = "unet"

pos_weight = 1

clean_up = False
normalization = "none"
input_mode = "valid"
pred_mode = "soft"

min_size = 57
n_unet_layers = 3
features_root = 16
in_size = 256
#in_size, out_size = util.unet_dim_calculator(min_size, n_unet_layers)
min_size, out_size = util.unet_dim_calculator_from_input_layer(in_size, n_unet_layers)
print("out_size %d min_size %d"%(out_size, min_size))

# in_size = 284
# out_size = 196
# n_unet_layers = 4

in_size = 512
#out_size = 216
n_unet_layers = 4

subject_tag = ""

def start_predict(model_in_path, label_out_path, model_specified=False):
    generator = MriDataProvider(in_size=in_size)
    splits = list()
    #model_in_ckpt = model_in_path + '/model.cpkt'
    model_in_ckpt = model_in_path

    print(label_out_path)
    
    print ('Predicting model using manual split')
    splits = io.read_manual_splits(manual_splits)
    #splits = io.read_manual_splits('settings/manual_splits.txt')

    net = ComputingGraph(channels = generator.channels, 
                n_class = generator.n_class, 
                layers=n_unet_layers, 
                features_root=features_root,
                network=network,
                in_size=in_size,
                out_size=out_size)
        
    for subj in splits[1]:
        generator.clear_data()
        print(dataDir)
        print(subject_tag)
        print(subj)
        generator.load_predict_subject(dataDir = dataDir, tag=subject_tag, subj_name = subj, normalization=normalization)
        # generator.load_predict_subject_experiment(dataDir = dataDir, tag=subject_tag, subj_name = subj, normalization=normalization)

        X_test, _ = generator.get_test_data()
        prediction = net.predict_big(model_path = model_in_ckpt, x_test = X_test)

        #io.save_NWH_to_nifty(prediction[...,0], "%s/%s_neg.nii.gz"%(label_out_path, subj),1.0)
        io.save_NWH_to_nifty(prediction[...,1], "%s/%s_class1.nii.gz"%(label_out_path, subj),0.0)
        io.save_NWH_to_nifty(prediction[...,2], "%s/%s_class2.nii.gz"%(label_out_path, subj),0.0)

        img = util.combine_img_prediction_v2(X_test, None, prediction, pred_mode, weights=[1,pos_weight])

        #if not os.path.exists(label_out_path):
        #    os.makedirs(label_out_path)
        #util.save_image(img, "%s/%s.jpg"%(label_out_path, subj))

if __name__ == "__main__":
    # read command line options
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"i:o:d:",
                                   ["output_dir=", "gpu_id=", "model_in_dir=",
                                    "data_in=", "model_in_path=", "network=",
                                    "manual_splits="])

    except getopt.GetoptError:
        print ('Syntax error')
        sys.exit(2)

    model_in_path_tmp = None
    model_specified = False

    for opt, arg in opts:

        if opt in ("-i", "--data_in"):
            dataDir = arg

        elif opt in ("-o", "--output_dir"):
            outputdir = arg

        elif opt in ("-d", "--gpu_id"):
            device = arg
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(arg)-1)

        elif opt in ("--model_in_path"):
            ev_model_path = arg
            model_specified = True

        elif opt in ("--network"):
            network = arg

        elif opt in ("--manual_splits"):
            manual_splits = arg

    start_predict(ev_model_path, outputdir, model_specified)



    


