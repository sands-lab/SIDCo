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

# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import argparse, os
import settings
import utils
import logging
import subprocess

import math

import distributed_optimizer as hvd
from dl_trainer import DLTrainer, _support_datasets, _support_dnns
from compression import compressors
from profiling import benchmark
from mpi4py import MPI
comm = MPI.COMM_WORLD
writer = None

from logger import TensorboardLogger #, FileLogger
import wandb

from settings import * #logger, formatter

if settings.REPRODUCIBLE:
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(settings.SEED)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(settings.SEED)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(settings.SEED)

    # 4. Set `pytorch` pseudo-random generator at a fixed value
    import torch
    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)
    #torch.cuda.manual_seed_all(settings.SEED)

def ssgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, threshold, gradient_path=None, tb=None, iratio=0.1, stages=1, partitions=0, ec_gradw=1.0, ec_memw=0.0, optimizer='nesterov', totaltime=0):
    global SPEED
    if not settings.USE_CPU:
        if nworkers > 1:
            rank = hvd.rank()
            torch.cuda.set_device(hvd.local_rank()) #%rank%nwpernode)
        else:
            rank=0
            torch.cuda.set_device(rank)
    if rank != 0:
        pretrain = None

    #### CHECK whether to use GPU or CPU
    if settings.USE_CPU:
        trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=0, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', pretrain=pretrain, num_steps=num_steps, tb_writer=writer, tb=tb,  optimizer_str=optimizer)
    else:
        trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', pretrain=pretrain, num_steps=num_steps, tb_writer=writer, tb=tb, optimizer_str=optimizer)

    init_epoch = torch.ones(1) * trainer.get_train_epoch()
    init_iter = torch.ones(1) * trainer.get_train_iter()
    trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))
    is_sparse = True #density < 1
    if not is_sparse:
        compressor = None

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None


    norm_clip = None
    if dnn == 'lstm':
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor], is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, tb=tb, iratio=iratio, stages=stages, partitions=partitions, ec_gradw=ec_gradw, ec_memw=ec_memw)
    hvd.SPEED
    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)

    start = time.time()
    times = []
    noupdate_times = []
    logger.info('max_epochs: %d', max_epochs)
    display = settings.DISPLAY if iters_per_epoch > settings.DISPLAY else iters_per_epoch-1

    for epoch in range(max_epochs):
        hidden = None
        if dnn == 'lstm':
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch):
            s = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if j < nsteps_update - 1 and nsteps_update > 1:
                    optimizer.local = True
                else:
                    optimizer.local = False
                if dnn == 'lstm':
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if dnn == 'lstm':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
            noupdate_times.append(time.time() - s)

            trainer.update_model()
            torch.cuda.synchronize()
            times.append(time.time() - s)

            if i % display == 0 and i > 0:
                time_per_iter = np.mean(times)
                update_per_iter = time_per_iter - np.mean(noupdate_times)
                throughput = batch_size * nsteps_update / time_per_iter
                trainer.log_info(time_per_iter, throughput, update_per_iter)
                logger.warning('Time per iteration: %f, communication: %f, Speed: %f images/s', time_per_iter, update_per_iter, throughput)
                times = []
                noupdate_times = []

        optimizer.increase_one_epoch()

        if totaltime > 0 and time.time() - start > totaltime:
            trainer.test(trainer.get_train_epoch())
            break
    if not(dataset == 'cifar10'):
        trainer.test(trainer.get_train_epoch())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets, help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--saved-dir', type=str, default='.', help='Specify the saved weights or gradients root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument('--compressor', type=str, default='sigmathresallgather', choices=compressors.keys(), help='Specify the compressors if \'compression\' is open')
    parser.add_argument('--density', type=float, default=1, help='Density for sparsification')
    parser.add_argument('--threshold', type=int, default=524288000, help='Specify the threshold for gradient merging')

    parser.add_argument('--netdevice', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='nesterov')

    parser.add_argument('--wandbkey', type=str, default='none')
    parser.add_argument('--projname', type=str, default='test')
    parser.add_argument('--name', type=str, default='testing',
                        help="name of the current run, used for machine naming and tensorboard visualization")
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--cluster', type=str, default='mcnodes')

    parser.add_argument('--totaltime', type=int, default=0, help='total time for the experiment in seconds')
    parser.add_argument('--iratio', type=float, default=0.1, help='ratio used in the loops of estimation')
    parser.add_argument('--stages', type=int, default=1, help='Number of stages used in the loops of estimation')
    parser.add_argument('--partitions', type=int, default=1, help='Number of paritions to divide the model gradient into')
    parser.add_argument('--ec-gradw', type=float, default=1.0, help='the weight of gradient component in error compensation')
    parser.add_argument('--ec-memw', type=float, default=0.0, help='the weight of old memory component in error compensation')

    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX
    if args.density < 1:
        prefix = 'comp-' + args.compressor + '-' + prefix
    logdir = 'allreduce-%s-thres-%dkbytes/%s-n%d-bs%d-lr%.4f-ns%d-ds%s' % (prefix, args.threshold/1024, args.dnn, args.nworkers, batch_size, args.lr, args.nsteps_update, str(args.density)) 
    relative_path = './logs/%s'%logdir
    gradient_relative_path = None 
    utils.create_path(relative_path)
    if settings.LOGGING_GRADIENTS:
        gradient_relative_path = '%s/gradients/%s'%(args.saved_dir, logdir)
        utils.create_path(gradient_relative_path)
    rank = 0
    if args.nworkers > 1:
        hvd.init()
        rank = hvd.rank()
    else:
        rank = 0
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    # Wandb and tensorboard logging
    # initialize WANDB
    if rank != 0:
        os.environ['WANDB_MODE'] = 'dryrun'  # all wandb.log are no-op
        logger.info("local-only wandb logging for run " + args.name)

    tb = TensorboardLogger(relative_path, is_master=(rank == 0))
    # log = FileLogger(args.logdir, is_master=is_master, is_rank0=is_master)

    # Ahmed - Scale learning rate with respect to compression ratio
    lr = args.lr

    if settings.SCALE_LR and args.density < 1:
        if args.optimizer == 'sgd':
            lr = args.lr * abs(math.log(args.density, 10))
        else:
            lr = args.lr / abs(math.log(args.density))

    # Ahmed - Update it to add configs
    if args.wandbkey != 'none':
        os.environ["WANDB_API_KEY"] = args.wandbkey
    if args.tags is None or args.tags == 'notags':
        wandb.init(project=args.projname, name=args.name,
                   config={"epochs": args.max_epochs, "lr": lr, "dataset": args.dataset, "model": args.dnn,
                           "batch_size": args.batch_size, "optimizer": args.optimizer,
                           "nodes": args.nworkers, "workerspernode": args.nwpernode, "partitions": args.partitions,
                           "threshold": args.threshold, "compressor": args.compressor,
                           "ratio": args.density, 'datadir': args.data_dir, 'stages': args.stages,
                           'ec_gradw': args.ec_gradw, 'ec_memw': args.ec_memw, 'nstepsupdate': args.nsteps_update
                       , 'iratio': args.iratio, 'netdevice': args.netdevice, 'runtime': args.totaltime,
                           'cluster': args.cluster})
    else:
        wandb.init(project=args.projname, name=args.name, tags=args.tags,
                   config={"epochs": args.max_epochs, "lr": lr, "dataset": args.dataset, "model": args.dnn,
                           "batch_size": args.batch_size, "optimizer": args.optimizer,
                           "nodes": args.nworkers, "workerspernode": args.nwpernode, "partitions": args.partitions,
                           "threshold": args.threshold, "compressor": args.compressor,
                           "ratio": args.density, 'datadir': args.data_dir, 'stages': args.stages,
                           'ec_gradw': args.ec_gradw, 'ec_memw': args.ec_memw, 'nstepsupdate': args.nsteps_update
                           ,'iratio': args.iratio, 'netdevice': args.netdevice, 'runtime': args.totaltime,
                           'cluster': args.cluster})

    logger.info("initializing wandb logging to run " + args.name + " name ")

    ssgd(args.dnn, args.dataset, args.data_dir, args.nworkers, lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.threshold, gradient_relative_path, tb, args.iratio, args.stages, args.partitions, args.ec_gradw, args.ec_memw, args.optimizer, args.totaltime)