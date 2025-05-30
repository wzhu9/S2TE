# S2TE: Staged Scale-Free Topology Evolution for Sparse Spiking Neural Networks

## Prerequisites

The Following Setup is tested and it is working:

python = 3.9

Pytorch = 2.0.0

Cuda = 11.8

tensorboard = 2.12.0

tqdm = 4.61.2

scipy = 1.10.1

matplotlib = 3.7.1



## Preprocess of DVS-CIFAR10

- Download CIFAR10-DVS dataset
- transform .aedat to .mat by test_dvs.m with matlab.
- prepare the train and test data set by dvscifar_dataloader.py
- you can obtain processed data in this [link](https://pan.baidu.com/s/1KhQk7z2irBQnr8BSWq7F3w?pwd=SEDS)

## Quick Start

 First set up the requirements needed:

```
pip install -r requirements.txt
```

Then, the running modes are as follows:

```
#you should cd to the experiment directory such as ./S2TE/experiment#

#running codes on different datasets
#CIFAR10
python development/main.py --seed 60 --arch resnet19  --auto_aug --cutout --wd 5e-4 --dataset CIFAR10 --act mns_rec  --T 2 --decay 0.5 --thresh 1.0 --data_path data/CIFAR10   --bn_type tdbn  --gamma 1.0 --sparsity 0.7  --phi_c 0.2 --phi_a  0.4 --zeta 0.4

#CIFAR10
python development/main.py --seed 60 --arch resnet19  --auto_aug --cutout --wd 5e-4 --dataset CIFAR100 --act mns_rec  --T 2 --decay 0.5 --thresh 1.0 --data_path data/CIFAR100   --bn_type tdbn  --gamma 1.0 --sparsity 0.7  --phi_c 0.2 --phi_a  0.4 --zeta 0.4

#DVSCIFAR10
python dvs/main.py --seed 200 --arch VGGSNN2  --bn_type tdbn --wd 1e-3 --num_workers 4  --act mns_rec --decay 0.5  --alpha 3.0 --dataset CIFAR10DVS --data_path ./data/CIFAR10DVS --sparsity 0.7 --zeta 0.3
```

Config files are in the directory  `experiment\development\config` and `experiment\dvs\config`



