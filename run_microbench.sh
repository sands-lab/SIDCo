#!/usr/bin/env bash

#'''
# *
# * SIDCo - Efficient Statistical-based Compression Technique for Distributed ML.
# *
# *  Author: Ahmed Mohamed Abdelmoniem Sayed, <ahmedcs982@gmail.com, github:ahmedcs>
# *
# * This program is free software; you can redistribute it and/or
# * modify it under the terms of CRAPL LICENCE avaliable at
# *    http://matt.might.net/articles/crapl/
# *    http://matt.might.net/articles/crapl/CRAPL-LICENSE.txt
# *
# * This program is distributed in the hope that it will be useful, but
# * WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# * See the CRAPL LICENSE for more details.
# *
# * Please READ carefully the attached README and LICENCE file with this software
# *
# '''

###CPU
python microbenchmark.py --grad-file=grads/cifar10_resnet20_r0_gradients_iter_100.npy --ratio=0.1 --no-cuda --file-prefix=resnet20
python microbenchmark.py --grad-file=grads/cifar10_resnet20_r0_gradients_iter_100.npy --ratio=0.01 --no-cuda --file-prefix=resnet20
python microbenchmark.py --grad-file=grads/cifar10_resnet20_r0_gradients_iter_100.npy --ratio=0.001 --no-cuda --file-prefix=resnet20
#CUDA
python microbenchmark.py --grad-file=grads/cifar10_resnet20_r0_gradients_iter_100.npy --ratio=0.1  --file-prefix=resnet20
python microbenchmark.py --grad-file=grads/cifar10_resnet20_r0_gradients_iter_100.npy --ratio=0.01  --file-prefix=resnet20
python microbenchmark.py --grad-file=grads/cifar10_resnet20_r0_gradients_iter_100.npy --ratio=0.001  --file-prefix=resnet20
#
#
##CPU
python microbenchmark.py --grad-file=grads/cifar10_vgg16_r0_gradients_iter_100.npy --ratio=0.1 --no-cuda --file-prefix=vgg16
python microbenchmark.py --grad-file=grads/cifar10_vgg16_r0_gradients_iter_100.npy --ratio=0.01 --no-cuda --file-prefix=vgg16
python microbenchmark.py --grad-file=grads/cifar10_vgg16_r0_gradients_iter_100.npy --ratio=0.001 --no-cuda --file-prefix=vgg16
#CUDA
python microbenchmark.py --grad-file=grads/cifar10_vgg16_r0_gradients_iter_100.npy --ratio=0.1  --file-prefix=vgg16
python microbenchmark.py --grad-file=grads/cifar10_vgg16_r0_gradients_iter_100.npy --ratio=0.01  --file-prefix=vgg16
python microbenchmark.py --grad-file=grads/cifar10_vgg16_r0_gradients_iter_100.npy --ratio=0.001  --file-prefix=vgg16

#CPU
python microbenchmark.py --grad-file=grads/imagenet_resnet50_r0_gradients_iter_100.npy --ratio=0.1 --no-cuda --file-prefix=resnet50
python microbenchmark.py --grad-file=grads/imagenet_resnet50_r0_gradients_iter_100.npy --ratio=0.01 --no-cuda --file-prefix=resnet50
python microbenchmark.py --grad-file=grads/imagenet_resnet50_r0_gradients_iter_100.npy --ratio=0.001 --no-cuda --file-prefix=resnet50
#CUDA
python microbenchmark.py --grad-file=grads/imagenet_resnet50_r0_gradients_iter_100.npy --ratio=0.1  --file-prefix=resnet50
python microbenchmark.py --grad-file=grads/imagenet_resnet50_r0_gradients_iter_100.npy --ratio=0.01  --file-prefix=resnet50
python microbenchmark.py --grad-file=grads/imagenet_resnet50_r0_gradients_iter_100.npy --ratio=0.001  --file-prefix=resnet50

##CPU
python microbenchmark.py --grad-file=grads/ptb_lstm_r0_gradients_iter_100.npy --ratio=0.1 --no-cuda --file-prefix=lstm
python microbenchmark.py --grad-file=grads/ptb_lstm_r0_gradients_iter_100.npy --ratio=0.01 --no-cuda --file-prefix=lstm
python microbenchmark.py --grad-file=grads/ptb_lstm_r0_gradients_iter_100.npy --ratio=0.001 --no-cuda --file-prefix=lstm
#CUDA
python microbenchmark.py --grad-file=grads/ptb_lstm_r0_gradients_iter_100.npy --ratio=0.1  --file-prefix=lstm
python microbenchmark.py --grad-file=grads/ptb_lstm_r0_gradients_iter_100.npy --ratio=0.01  --file-prefix=lstm
python microbenchmark.py --grad-file=grads/ptb_lstm_r0_gradients_iter_100.npy --ratio=0.001  --file-prefix=lstm

##CPU
#python microbenchmark.py --grad-file=grads/imagenet_googlenet_r0_gradients_iter_100.npy --ratio=0.1 --no-cuda --file-prefix=googlenet
#python microbenchmark.py --grad-file=grads/imagenet_googlenet_r0_gradients_iter_100.npy --ratio=0.01 --no-cuda --file-prefix=googlenet
#python microbenchmark.py --grad-file=grads/imagenet_googlenet_r0_gradients_iter_100.npy --ratio=0.001 --no-cuda --file-prefix=googlenet
##CUDA
#python microbenchmark.py --grad-file=grads/imagenet_googlenet_r0_gradients_iter_100.npy --ratio=0.1  --file-prefix=googlenet
#python microbenchmark.py --grad-file=grads/imagenet_googlenet_r0_gradients_iter_100.npy --ratio=0.01  --file-prefix=googlenet
#python microbenchmark.py --grad-file=grads/imagenet_googlenet_r0_gradients_iter_100.npy --ratio=0.001  --file-prefix=googlenet

