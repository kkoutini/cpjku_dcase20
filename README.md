# CP JKU Submission for DCASE 2020

Technical report [DCASE](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Koutini_142.pdf). 
Workshop paper with more analysis of the parameter reduction methods [DCASE](http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Koutini_91.pdf).
This Repo code is based on [CP JKU Submission for DCASE 2019](https://github.com/kkoutini/cpjku_dcase19). 
The architectures are based on Receptive-Field-Regularized CNNs  [The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification]( https://arxiv.org/abs/1907.01803), [Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification](https://arxiv.org/abs/1909.02859)

Feel free to raise an issue or contact the authors in case of an issue.


## Requirements

[Conda]( https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda ) should be installed on the system.

```install_dependencies.sh``` installs the following:
* Python 3
* PyTorch  
* torchvision
* [tensorboard-pytorch]( https://github.com/lanpa/tensorboard-pytorch )
* etc..

## Installation
* Install [Anaconda](https://www.anaconda.com/) or conda

* Run the install dependencies script:
```bash
./install_dependencies.sh
```
This creates conda environment ```cpjku_dcase20``` with all the dependencies.

Running
``` source activate cpjku_dcase20``` is needed before running ```exp*.py```


## Usage
After installing dependencies:

- Activate Conda environment created by ```./install_dependencies.sh```
    ```bash
    $ source activate cpjku_dcase20
    ```

- Download the dataset:
    ```bash
    $ python download_dataset.py --version 2020b
    ```
    You can also download previous versions of DCASE ```--version year```, year is one of 2018,2017,2016,2019,2020a or any dataset from [dcase_util](https://dcase-repo.github.io/dcase_util/datasets.html).
    
    Alternatively, if you already have the dataset downloaded:
    - You can make link to the dataset: 
    ```bash
    ln -s ~/some_shared_folder/TAU-urban-acoustic-scenes-2019-development ./datasets/TAU-urban-acoustic-scenes-2019-development
    ```
    
    - Change the paths in ```config/[expermient_name].json```.
    
- Run the experiment script:
    ```
    $ CUDA_VISIBLE_DEVICES=0 python exp_[expeirment_name].py 
    ```
- The output of each run is stored in ``outdir``, you can also monitor the experiments with TensorBoard, using the logs stored in the tensorboard runs dir ```runsdir```. 
 Example: 
     ```bash
     tensorboard --logdir   ./runsdir/cp_resnet/exp_Aug20_14.11.28
     ```
 The exact commmand is printed when you run the experiment script.

## Example runs
### DCASE 2020 DCASE 1b
#### CP_ResNet
default adapted receptive field RN1,RN1 (in Koutini2019Receptive below):
```
$ CUDA_VISIBLE_DEVICES=0 python exp_cp_resnet.py 
```
Large receptive Field
```
$ CUDA_VISIBLE_DEVICES=0 python exp_cp_resnet.py  --rho 15
```
very small max receptive Field:

```
$ CUDA_VISIBLE_DEVICES=0 python exp_cp_resnet.py  --rho 2
```
# Loading pretrained models
(DCASE20 models will be added soon)
Download the evaluation set:
```bash
$ python download_dataset.py --version 2019eval
```
Download the trained models (from [zando](  https://zenodo.org/record/3674034))

Run the experiment with the `load` the correct `rho` value, because the `rho` value changes the network weights shape) 
```bash
$ CUDA_VISIBLE_DEVICES=0 python exp_cp_resnet.py  --rho 5 --load=path_to_model.pth
```
In case that you want to predict on a different dataset, you should add the dataset to the config file.
For example look at the `eval` dataset in  `configs/cp_resnet_eval.json`.
# Missing Features
This repo is used to publish for our submission to DCASE 2019 and MediaEval 2019. If some feauture/architecture/dataset missing feel free to contact the authors or to open an issue.

# Citation


[The Receptive Field as a Regularizer ]( https://arxiv.org/abs/1907.01803 ), 
[Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification](https://arxiv.org/abs/1909.02859), 
[Low-Complexity Models for Acoustic Scene Classification Based on Receptive Field Regularization and Frequency Damping](http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Koutini_91.pdf)

```
@INPROCEEDINGS{Koutini2019Receptive,
AUTHOR={ Koutini, Khaled and Eghbal-zadeh, Hamid and Dorfer, Matthias and Widmer, Gerhard},
TITLE={{The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification}},
booktitle = {Proceedings of the European Signal Processing Conference (EUSIPCO)},
ADDRESS={A Coru\~{n}a, Spain},
YEAR=2019
}


@inproceedings{KoutiniDCASE2019CNNVars,
  title = {Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification},
  booktitle = {Preprint},
  date = {2019-10},
  author = {Koutini, Khaled and Eghbal-zadeh, Hamid and Widmer, Gerhard},
}


@inproceedings{Koutini2020,
    author = "Koutini, Khaled and Henkel, Florian and Eghbal-Zadeh, Hamid and Widmer, Gerhard",
    title = "Low-Complexity Models for Acoustic Scene Classification Based on Receptive Field Regularization and Frequency Damping",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020)",
    address = "Tokyo, Japan",
    month = "November",
    year = "2020",
    pages = "86--90",
}


 ```