## SLNAS implement with pytorch 
Paper: Stage and Lite Search Technology in Convolutional Neural Network Architecture Search  
Authors: Tsung-Yi Chen

Includes code for search and scratch.


## Experiment Dataset
- CIFAR10 : https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- AID :　https://captain-whu.github.io/AID/
- NWPU-RESISC45　: https://arxiv.org/ftp/arxiv/papers/1703/1703.00121.pdf
- Flowers102 : https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/nilsback08.pdf
- Sport8 : http://vision.stanford.edu/documents/LiFei-Fei_ICCV07.pdf
- MIT67 : http://people.csail.mit.edu/torralba/publications/indoor.pdf
- PlantVillage : https://arxiv.org/ftp/arxiv/papers/1604/1604.03169.pdf

## Data path 
```
../data
  └── ../data/YOUR_IMAGE_DATASET
      ├── ../data/YOUR_IMAGE_DATASET/train
      │   ├── xxx.jpg
      │   └── ...
      └── ../data/YOUR_IMAGE_DATASET/val
          ├── xxx.jpg
          └── ...
```

## Prerequisites

- PyTorch 1.4.0
- apex 
- numpy 

## search 

Run  SLNAS search experiments , please use:
```
cmd>>python retrainer_cifar.py    --save cifar10_retrain 
                                  --seed 2
                                  --batch_size 32
                                  --child_out_filters 40
                                  --data ../data/exp_data/AID_50
                                  --num_class 30
                                  --epochs 600
                                  --blocks "[2,2]"
                                  --arch "[[0, 1, 0, 4, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0], [0, 3, 1, 0, 0, 4, 1, 1, 1, 1, 0, 3, 0, 2, 1, 2, 1, 1, 0, 3]]"
```
## scratch 
```
cmd>>python train_search_cifar.py    --save cifa10_searcher
                                     --seed 0
                                     --batch_size 96
                                     --child_out_filters 16
                                     --data ../data/cifar10
                                     --num_class 30
                                     --epochs 150
                                     --blocks "[2,2]"
                                     --weight_decay 3e-4
```
## Result on cifar10
| Tables  | parameters / M  | error rate / %| search cost / days|
| :------:|:--------:|:-------------:|:-------------:|
| ENAS    | 4.6      | 2.89          |0.45|
| DARTS   | 3.3      |  2.76         |4|
| NAO     | 2.5      |    2.93       |0.3|
| SLNAS   | 1.31     |    2.86       |0.3|

## TODO

- [ ] Search without label 
- [ ] Search for detection and Segmentation 
- [ ] AutoAugment
- [ ] decrease inference time and require samples

## Acknowledgements

This implementation is based on

- [melodyguan/enas](https://github.com/melodyguan/enas/)
- [carpedm20/ENAS-pytorch](https://github.com/carpedm20/ENAS-pytorch/)
- [quark0/darts](https://github.com/quark0/darts/)
- [MINGUKKANG](https://github.com/MINGUKKANG/ENAS-Tensorflow)
- [adobe](https://github.com/adobe/antialiased-cnns)
- [moskomule](https://github.com/moskomule/senet.pytorch)


