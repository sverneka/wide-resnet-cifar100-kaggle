# Wide Residual Networks Using Ensemble for CIFAR100 Kaggle submission (.csv is generated)


This is taken from https://github.com/meliketoy/wide-residual-network/ and modified to create solutionto STAT946 kaggle challenge - Deep Learning and gives 0.83971 on the private leaderboard: https://www.kaggle.com/c/fall2017stat946/leaderboard

NOTE: Before submitting the solution, make sure to remove "true_labels" column from the csv file generated.

All my experiments were run on GeForce GTX 1080.

Wide-residual network implementations for cifar100

Torch Implementation of Sergey Zagoruyko's [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v2.pdf).

In order to figure out what 'width' & 'height' does on wide-residual networks, 
several experiments were conducted on different settings of weights and heights.
It turns out that **increasing the number of filters(increasing width)** gave more positive influence 
to the model than making the model deeper.

Last but not least, simply averaging a few models with different parameter settings showed a significant increase in both top1 and top5 accuracy. The CIFAR dataset test results approached to **84.19%** for CIFAR-100 with only **meanstd** normalization.

## Requirements
See the [installation instruction](INSTALL.md) for a step-by-step installation guide.
See the [server instruction](SERVER.md) for server setup.
- Install [Torch](http://torch.ch/docs/getting-started.html)
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Install luarocks packages
```bash
$ luarocks install cutorch
$ luarocks install xlua
$ luarocks install optnet
```
## Directions and datasets
- modelState    : The best model will be saved in this directory
- datasets      : Data preparation & preprocessing directory
- networks      : Wide-residual network model structure file directory
- gen           : Generated t7 file for each dataset will be saved in this directory
- scripts       : Directory where the run file scripts are contained

## Best Results


Adapting weight adjustments for each model will promise a more improved accuracy.

You can see that the ensemble network improves the results of single WRNs.

Test error (%, random flip, **meanstd** normaliztion, median of 5 runs) on CIFAR:

|   Dataset   | network      |  Top1 Err(%) |
|:-----------:|:------------:|:------------:|
| CIFAR-100   | WRN-28x10    |    18.85     |
| CIFAR-100   | Ensemble-WRN |  **15.81**   |

## How to run
You can train each dataset of cifar100 by editing and running the script below.
```bash
$ ./scripts/train.sh

```

You can test your own trained model of cifar100 by editing and running the script below. This will print test results and save the .csv file for submission to kaggle
```bash
$ ./scripts/test.sh
```

To train an ensemble follow the steps below.
```bash
$ vi ensemble.lua
# Press :32 in vi, which will move your cursor to line 32
ens_depth         = torch.Tensor({28, 28, 28, 28, 40, 40})
ens_widen_factor  = torch.Tensor({10, 20, 20, 20, 10, 14})
ens_nExperiment   = torch.Tensor({ 1,  1,  2,  3,  1,  1})
```
then edit and run
```bash
$ ./scripts/ensemble_train.sh
```
This will create .csv file for kaggle submission


If you want to just create the ensemble of already trained models, edit ensemble.lua as mentioned above and then edit and run scripts/ensemble_test.sh
```bash
$ ./scripts/ensemble_test.sh
```

## Implementation Details

* CIFAR-100

|   epoch   | learning rate |  weight decay | Optimizer | Momentum | Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:--------:|
|   0 ~ 60  |      0.1      |     0.0005    | Momentum  |    0.9   |   true   |
|  61 ~ 120 |      0.02     |     0.0005    | Momentum  |    0.9   |   true   |
| 121 ~ 160 |     0.004     |     0.0005    | Momentum  |    0.9   |   true   |
| 161 ~ 200 |     0.0008    |     0.0005    | Momentum  |    0.9   |   true   |

## CIFAR-100 Results

![alt tag](IMAGES/cifar100_image.png)

Below is the result of the test set accuracy for **CIFAR-100 dataset** training.

NOTE: 28x20 configuration consumes the highest GPU memory ~11 GB while running on a single GPU, if you get error, it is possibly due to GPU memory insufficiency. Also note, while running script/ensemble_test.sh, you might hit into memory issues, in which case, either reduce number of models in the ensemble by modifying, ensemble.lua or edit test.lua to load models differently.
 
**Accuracy is the average of 5 runs**

| network           | dropout |  preprocess | GPU:0 | GPU:1 | per epoch    | Top1 acc(%)| Top5 acc(%) |
|:-----------------:|:-------:|:-----------:|:-----:|:-----:|:------------:|:----------:|:-----------:|
| pre-ResNet-1001   |    0    |   meanstd   |   -   |   -   | 3 min 25 sec |    77.29   |    93.44    |
| wide-resnet 28x10 |    0    |     ZCA     | 5.90G |   -   | 2 min 03 sec |    80.03   |    95.01    |
| wide-resnet 28x10 |    0    |   meanstd   | 5.90G |   -   | 2 min 03 sec |    81.01   |    95.44    |
| wide-resnet 28x10 |   0.3   |   meanstd   | 5.90G |   -   | 2 min 03 sec |    81.47   |    95.53    |
| wide-resnet 28x20 |   0.3   |   meanstd   | 8.13G | 6.93G | 4 min 05 sec |  **82.43** |  **96.02**  |
| wide-resnet 40x10 |   0.3   |   meanstd   | 8.93G |   -   | 3 min 06 sec |    81.47   |    95.65    |
| wide-resnet 40x14 |   0.3   |   meanstd   | 7.39G | 6.46G | 3 min 23 sec |    81.83   |    95.50    |
