'''
 *
 * SIDCo - Efficient Statistical-based Compression Technique for Distributed ML.
 *
 *  Author: Ahmed Mohamed Abdelmoniem Sayed, <ahmedcs982@gmail.com, github:ahmedcs>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of CRAPL LICENCE avaliable at
 *    http://matt.might.net/articles/crapl/
 *    http://matt.might.net/articles/crapl/CRAPL-LICENSE.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the CRAPL LICENSE for more details.
 *
 * Please READ carefully the attached README and LICENCE file with this software
 *
 '''

import os
import logging
import socket

SEED=12345
REPRODUCIBLE=True
DISPLAY=5

UPDATE_ITER=5 #15
RES_DIFF_UP=0.2
RES_DIFF_DOWN=0.2
FIRST_RATIO=0.25
FR_UPDATE=0.05
ADJUST_FR=False
SCALE_LR=False
CPU_COMPRESS=False

USE_CPU=False

DEBUG = 0
SERVER_PORT = 5911
PORT = 5922

WARMUP=True

PREFIX=''
if WARMUP:
    PREFIX=PREFIX+'gwarmup'

LOGGING_ASSUMPTION=False
LOGGING_GRADIENTS=False

EXP='-convergence'
PREFIX=PREFIX+EXP
ADAPTIVE_MERGE=False
ADAPTIVE_SPARSE=False
if ADAPTIVE_MERGE:
    PREFIX=PREFIX+'-ada'

TENSORBOARD=False
USE_FP16=False

MAX_EPOCHS=30

hostname=socket.gethostname() 
logger=logging.getLogger(hostname)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

