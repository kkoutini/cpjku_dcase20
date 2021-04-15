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

parser.add_argument('--mixed_precision_training', default=0, type=int,
                    help='use mixed_precision_training (torch>1.5). 0 disabled, 1 using optimization o1, 2 using optimization o2')


# rho value control the MAX RF of the Network values from 5-9 corresponds max rf similar to the popular VGG-like CNNs.
parser.add_argument('--rho', default=5, type=int,
                    help='rho value as explained in DCASE2019 workshop paper '
                         '"Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification"'
                         '# rho value control the MAX RF of the Network values from 5-9 corresponds max rf similar to the popular VGG-like nets.')

# Parameter Reduction options:
#  CP-ResNet inital width.
parser.add_argument('--width', default=128, type=int,
                    help='Width determines the initial number of channels.'
                         'increasing the width may improve the performance but at the cost of efficiency.')


# Removes tailing 1x1 layers from CP-ResNet to save parameters, make sure that no 3x3 layers are removed otherwise the Receptive Field of the network will will change.
parser.add_argument('--depth_restriction', default="0,0,0", type=str,
                    help='The number of tailing layers to be removed from each of the three stages. A string contains 3 integers separated by commmas.')

# only for decomp networks
parser.add_argument('--decomp_factor', default=4, type=int, 
                    help='decomposition factor (Z in the paper), used for all decomposed convolution layers. Needs the --arch to be cp_resnet_decomp or cp_resnet_decomp_freq_damp.\
                    if the architecture is not decomposed, this argument will be silently ignored.')

# only for pruned networks
# http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Koutini_91.pdf
parser.set_defaults(prunning_mode=False)
parser.add_argument('--prune', dest='prunning_mode', action='store_true', help="run the training in pruning mode. requires an architecture that supports pruning.\
        for example: cp_resnet_prune or cp_resnet_df_prune.")
parser.add_argument('--prune_rampup', default="linear", type=str, 
                    choices=["exponential","linear"],
                    help='The function describing number of pruned parameters each epochs, linear or exponential. See DCASE20 Workshop paper for details `\
                    Low-Complexity Models for Acoustic Scene Classification Based on Receptive Field Regularization and Frequency Damping`.')
parser.add_argument('--prune_rampup_len', default=50, type=int, 
                    help='Number of epochs until the number of pruned parameters reaches the final required number.')
parser.add_argument('--prune_ratio', default=-1, type=float, 
                    help='The ratio of parameters to be pruned. for example 0.9 means 90\% of parameters will be pruned.\
                    if set to `-1` then the percentage will be calculate based on the arg `prune_target_params`.')
parser.add_argument('--prune_target_params', default=-1, type=int, 
                    help='The number of parameters to remain after pruning. `prune_ratio` have to be `-1` otherwise this argument will be ignored.')
parser.add_argument('--prune_method', default="all", type=str, 
                    choices=["all","layer"],
                    help='Indicates wether to select the parameters to be pruned per layer or globally from all the network parameters. \
                    Pruning per layer is more robust agaisnt layers collapsing.')


# Optimization options
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--mixup', default=1, type=int,
                    help='use mixup if 1. ')


# model args


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




# overriding the database config 
print(f"\nSelected training dataset is configs/datasets{args.dataset} ...\n")
with open("configs/datasets/"+args.dataset, "r") as text_file:
    dataset_conf = json.load(text_file)
default_conf=utils_funcs.update_dict(default_conf,dataset_conf)


default_conf['out_dir'] = default_conf['out_dir'].replace("cp_resnet",args.arch) + str(datetime.datetime.now().strftime('%b%d_%H.%M.%S'))

print("The experiment outputs will be found at: ", default_conf['out_dir'])
tensorboard_write_path = default_conf['out_dir'].replace("out", "runs", 1)
print("The experiment tesnorboard can be accessed: tensorboard --logdir  ", tensorboard_write_path)

print("Rho value : ", args.rho)
print("Use Mix-up : ", args.mixup)

arch = importlib.import_module('models.{}'.format(args.arch))

## parsing model config updates
model_config_overrides = {"base_channels": args.width, "n_blocks_per_stage": [4-int(b) for b in args.depth_restriction.split(",")],
                          "decomp_factor": args.decomp_factor,
                          "n_classes":  default_conf['audiodataset']['num_of_classes'], # corrent the number of classes from the dataset config
                          }

# pruning mode
if args.prunning_mode:
    default_conf['prune_mode']= args.prune_method
    default_conf['adaptive_prune_rampup_mode']=args.prune_rampup
    default_conf['adaptive_prune_rampup_len']=args.prune_rampup_len
    default_conf['prune_percentage']=args.prune_ratio
    default_conf['prune_percentage_target_params']=args.prune_target_params
    if args.prune_ratio!=-1 and  args.prune_ratio>1.:
        raise RuntimeError("prune_ratio should be -1 (then it will be calculated from prune_target_params) or between 0 and 1.")
    if args.prune_ratio==-1 and args.prune_target_params==-1:
        raise RuntimeError("prune_ratio or prune_target_params need to be set.")

# get the final architecture config
default_conf['model_config'] = arch.get_model_based_on_rho(args.rho, args.arch, config_only=True, model_config_overrides=model_config_overrides)







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
trainer = Trainer(default_conf, mixed_precision_training=args.mixed_precision_training)


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



print("The experiment outputs will be found at: ", default_conf['out_dir'])

print("The experiment tesnorboard can be accessed: tensorboard --logdir  ", tensorboard_write_path)

