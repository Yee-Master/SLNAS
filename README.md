## SLNAS implement with pytorch 
Paper: Stage and Lite Search Technology in Convolutional Neural Network Architecture Search 
Authors: Tsung-Yi Chen

Includes code for search task and scratch .


## experiment dataset
- CIFAR10 : https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- AID :　https://captain-whu.github.io/AID/
- NWPU-RESISC45　: https://arxiv.org/ftp/arxiv/papers/1703/1703.00121.pdf
- Flowers102 : https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/nilsback08.pdf
- Sport8 : http://vision.stanford.edu/documents/LiFei-Fei_ICCV07.pdf
- MIT67 : http://people.csail.mit.edu/torralba/publications/indoor.pdf
- PlantVillage : https://arxiv.org/ftp/arxiv/papers/1604/1604.03169.pdf


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


## Acknowledgements

This implementation is based on

- [melodyguan/enas](https://github.com/melodyguan/enas/)
- [carpedm20/ENAS-pytorch](https://github.com/carpedm20/ENAS-pytorch/)
- [quark0/darts](https://github.com/quark0/darts/)
- [MINGUKKANG](https://github.com/MINGUKKANG/ENAS-Tensorflow)
- [adobe](https://github.com/adobe/antialiased-cnns)
- [moskomule](https://github.com/moskomule/senet.pytorch)
