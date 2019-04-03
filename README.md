# Investigating the effect of novel classes in semi-supervised learning

This repository contains code and other files used to investigate the effect of novel classes in unlabelled data when using semi-supervised learning. We define _novel_ classes as the classes that exist in the unlabelled data but not in the labelled data. We used Pseudo-Label and Mean Teacher semi-supervised algorithms in our experiments. We also proposed a 1-nearest-neighbour based weighting scheme to reduce the negative effect of novel classes. The experiments were run on three datasets: MNIST, Fashion-MNIST and CIFAR-10. In our experiments, the novel classes are the last 5 classes - {5, 6, 7, 8, 9}. The labelled data (including validation and test sets) only include classes {0, 1, 2, 3, 4}.

Example of use:
```
python meanteacher_cifar10_main.py \
    --n-labeled 3000 \
    --alpha 100 \
    --ema-decay 0.99 \
    --pure False \
    --rampup-period 10 \
    --lr 0.0001 \
    --epochs 180 \
    --runs 10
```
The above command will run our implementation of Mean Teacher on the CIFAR-10 dataset. The number of labelled training examples is set to 3000. The coefficient that controls the importance of consistency cost in Mean Teacher is set as 100. The rampup period for alpha is 10 epochs. The ema decay rate is set to 0.99. Learning rate is 0.0001. We set --pure to False, this means the unlabelled data include examples from novel classes. The training will proceed for 100 epochs. The experiment will be run 10 times, the data seed will be different for each run.
