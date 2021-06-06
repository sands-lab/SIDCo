#!/usr/bin/env bash

#'''
## *
## * SIDCo - Efficient Statistical-based Compression Technique for Distributed ML.
## *
## *  Author: Ahmed Mohamed Abdelmoniem Sayed, <ahmedcs982@gmail.com, github:ahmedcs>
## *
## * This program is free software; you can redistribute it and/or
## * modify it under the terms of CRAPL LICENCE avaliable at
## *    http://matt.might.net/articles/crapl/
## *    http://matt.might.net/articles/crapl/CRAPL-LICENSE.txt
## *
## * This program is distributed in the hope that it will be useful, but
## * WITHOUT ANY WARRANTY; without even the implied warranty of
## * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## * See the CRAPL LICENSE for more details.
## *
## * Please READ carefully the attached README and LICENCE file with this software
## *
## '''

size=1

#CPU
python microbenchmark.py  --ratio=0.1 --no-cuda  --size=$size
python microbenchmark.py  --ratio=0.01 --no-cuda  --size=$size
python microbenchmark.py  --ratio=0.001 --no-cuda  --size=$size
#CUDA
python microbenchmark.py  --ratio=0.1   --size=$size
python microbenchmark.py  --ratio=0.01   --size=$size
python microbenchmark.py  --ratio=0.001   --size=$size

size=10

#CPU
python microbenchmark.py  --ratio=0.1 --no-cuda  --size=$size
python microbenchmark.py  --ratio=0.01 --no-cuda  --size=$size
python microbenchmark.py  --ratio=0.001 --no-cuda  --size=$size
#CUDA
python microbenchmark.py  --ratio=0.1   --size=$size
python microbenchmark.py  --ratio=0.01   --size=$size
python microbenchmark.py  --ratio=0.001   --size=$size

size=100

#CPU
python microbenchmark.py  --ratio=0.1 --no-cuda  --size=$size
python microbenchmark.py  --ratio=0.01 --no-cuda  --size=$size
python microbenchmark.py  --ratio=0.001 --no-cuda  --size=$size
#CUDA
python microbenchmark.py  --ratio=0.1   --size=$size
python microbenchmark.py  --ratio=0.01   --size=$size
python microbenchmark.py  --ratio=0.001   --size=$size


#size=1000
#
##CPU
#python microbenchmark.py  --ratio=0.1 --no-cuda  --size=$size
#python microbenchmark.py  --ratio=0.01 --no-cuda  --size=$size
#python microbenchmark.py  --ratio=0.001 --no-cuda  --size=$size
##CUDA
#python microbenchmark.py  --ratio=0.1   --size=$size
#python microbenchmark.py  --ratio=0.01   --size=$size
#python microbenchmark.py  --ratio=0.001   --size=$size

