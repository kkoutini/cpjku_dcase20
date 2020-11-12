from __future__ import print_function

import argparse
import datetime
import os
import shutil
import time
import random
import json

from trainer import Trainer
import utils_funcs
import traceback
import importlib
available_models = [m[:-3] for m  in os.listdir("models") if m.startswith("cp_")]


parser = argparse.ArgumentParser(description='CP_ResNet Training')
# Optimization options

parser.add_argument('--dataset', default="dcase2020b.json", type=str, 
                    help='dataset JSON configuration to load from `configs/datasets/` default is dcase2020b.json \n\
                    other options include dcase2019.json, dcase2018.json')

parser.add_argument('--arch', default="cp_resnet", type=str, 
                    choices=available_models,
                    help='The CNN architecture, one from the files located in `models/`')

# rho value control the MAX RF of the Network values from 5-9 corresponds max rf similar to the popular VGG-like CNNs.
parser.add_argument('--rho', default=5, type=int,
                    help='rho value as explained in DCASE2019 workshop paper '
                         '"Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification"'
                         '# rho value control the MAX RF of the Network values from 5-9 corresponds max rf similar to the popular VGG-like nets.')
# Optimization options
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--mixup', default=1, type=int,
                    help='use mixup if 1. ')


# model args
# Optimization options
parser.add_argument('--decomp_factor', default=4, type=int, 
                    help='decomposition factor, used for all decomposed convolution layers. Needs the --arch to be cp_resnet_decomp or cp_resnet_decomp_freq_damp.\
                    if the architecture is not decomposed, this argument will be silently ignored.')

# pre-trained models config
parser.add_argument('--load', default=None, type=str,
                    help='the pre-trained model path to load, in this case the model is only evaluated')


args = parser.parse_args()
if args.load is  None:
    with open("configs/cp_resnet.json", "r") as text_file:
        default_conf = json.load(text_file)
else:
    with open("configs/cp_resnet_eval.json", "r") as text_file:
        default_conf = json.load(text_file)


model_kwargs = {"decomp_factor": args.decomp_factor,}



# overriding the database config 
print(f"\nSelected training dataset is configs/datasets{args.dataset} ...\n")
with open("configs/datasets/"+args.dataset, "r") as text_file:
    dataset_conf = json.load(text_file)
default_conf=utils_funcs.update_dict(default_conf,dataset_conf)


default_conf['out_dir'] = default_conf['out_dir'] + str(datetime.datetime.now().strftime('%b%d_%H.%M.%S'))

print("The experiment outputs will be found at: ", default_conf['out_dir'])
tensorboard_write_path = default_conf['out_dir'].replace("out", "runs", 1)
print("The experiment tesnorboard can be accessed: tensorboard --logdir  ", tensorboard_write_path)

print("Rho value : ", args.rho)
print("Use Mix-up : ", args.mixup)

arch = importlib.import_module('models.{}'.format(args.arch))

default_conf['model_config'] = arch.get_model_based_on_rho(args.rho, args.arch, config_only=True)


# find the RF at the 24th layer of the model defined by this config
# this equations are explained in:
# The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification,
# Koutini et al.
# EUSIPCO 2019

try:
    # set utils_funcs.model_config to the current model (not safe with lru)
    utils_funcs.model_config = default_conf['model_config']
    _, max_rf = utils_funcs.get_maxrf(24)
    print("For this Rho, the maximium RF is: ", max_rf)
except:
    print("couldn't determine the max RF, maybe non-standard model_config")
    traceback.print_exc()

if args.mixup:
    default_conf['use_mixup'] = True
    default_conf['loss_criterion'] = 'mixup_default'
else:
    default_conf['use_mixup'] = False

epochs = args.epochs
trainer = Trainer(default_conf)
if args.load is not None:
    model_path = args.load
    print("will load pre-trained model from ", model_path)
    import torch
    from datetime import datetime
    checkpoint = torch.load(model_path)
    try:
        trainer.bare_model.load_state_dict(checkpoint['state_dict'])
    except:
        print("\n\nFailed: to load weights check that you have the correct rho value\n\n")
        raise
    print("model loaded, predicting...")
    sids, propbs = trainer.do_predict("eval",{})
    print("sids:",len(sids),propbs.shape)
    torch.save((sids, propbs),str(datetime.now())+"eval_predictions.pth")
else:
    trainer.fit(epochs)
    trainer.predict("last")
    trainer.load_best_model()
    trainer.predict()
