

import dcase_util


import argparse
import os
import shutil
import time
import random
from trainer import  Trainer


parser = argparse.ArgumentParser(description='use the arg --version [2016:2019],2019_eval,2020a,2020b to work with older DCASE datasets. \
	list of datasets https://dcase-repo.github.io/dcase_util/datasets.html')
# Optimization options
parser.add_argument('--version', default="2020b")
args = parser.parse_args()


print(" ")
datasets_map={
    "2016": "TUTAcousticScenes_2016_DevelopmentSet",
    "2017": "TUTAcousticScenes_2017_DevelopmentSet",
    "2018": "TUTUrbanAcousticScenes_2018_DevelopmentSet",
    "2019": "TAUUrbanAcousticScenes_2019_DevelopmentSet",
    "2019_eval": "TAUUrbanAcousticScenes_2019_EvaluationSet",
    "2020a": "TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet",
    "2020b": "TAUUrbanAcousticScenes_2020_3Class_DevelopmentSet",
    "2020b_eval": "TAUUrbanAcousticScenes_2020_3Class_EvaluationSet",


}
chosen_dataset=args.version
if chosen_dataset in datasets_map:
	print("chosen dataset: ",datasets_map[args.version])
	chosen_dataset=datasets_map[chosen_dataset]
	
# from task1 baseline at http://dcase.community/challenge2019/

ds_path="./datasets/"
dcase_util.utils.Path().create(
    paths=ds_path
)
dcase_util.datasets.dataset_factory(
    dataset_class_name=chosen_dataset,
    data_path=ds_path,
).initialize().log()
