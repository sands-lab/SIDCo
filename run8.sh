#!/bin/bash

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

curdir=`pwd`
dnn="${dnn:-resnet20}"
source $curdir/exp_configs/$dnn.conf
nworkers="${nworkers:-8}"
density="${density:-0.001}"
compressor="${compressor:-topk}"
threshold="${threshold:--1}"
partition="${partition:-1}"

if [[ "$dnn" == "resnet50" ]]; then
  totaltime="${totaltime:-18000}"  # for Resnet50
elif [[ "$dnn" == "vgg19" ]]; then
   totaltime="${totaltime:-17400}" #for vgg19
else
  totaltime="${totaltime:-0}"
fi

if [[ "$compressor" == "gpareto"* ]]; then
  iratio="${iratio:-0.075}";
  echo "setting iratio for Gen Pareto to $iratio";
elif [[ "$compressor" == "exp"* ]]; then
  iratio="${iratio:-0.2}";
  echo "setting iratio for exponetial to $iratio";
else
  iratio="${iratio:-0.1}"
fi
stages="${stages:-1}"
optimizer="${optimizer:-sgd}"
ecgradw="${ecgradw:-1.0}"
ecmemw="${ecmemw:-0.0}"
nwpernode=1
nstepsupdate=1
cluster="${cluster:-mcnodes}"
#MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=python
GRADSPATH=$curdir/logs

#Ahmed  -- added extra params
projname="${projname:-test}"
runname="${runname:-cifar10_none}"
tags="${tags:-notags}"
netdevice="${netdevice:-enp1s0f1}"


echo "calling MPIRUN: $PY $curdir/dist_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --saved-dir $GRADSPATH  --projname $projname --name $runname --netdevice $netdevice --threshold $threshold --partition $partition --iratio $iratio --stages $stages --ec-gradw $ecgradw --ec-memw $ecmemw --optimizer $optimizer --wandbkey $wandbkey --totaltime $totaltime --cluster $cluster"

mpirun --allow-run-as-root -wdir $curdir --mca orte_base_help_aggregate 0 \
     --oversubscribe -np $nworkers -hostfile $curdir/clusters/cluster8 -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -mca plm_rsh_args "-p 12346" -mca btl_tcp_if_include $netdevice \
    -x NCCL_DEBUG=INFO -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=$netdevice -x CUDA_VISIBLE_DEVICES=0,1 \
    -x HOROVOD_FUSION_THRESHOLD=0 \
    $PY $curdir/dist_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --saved-dir $GRADSPATH  --projname $projname --name $runname --tags $tags --netdevice $netdevice --threshold $threshold --partition $partition --iratio $iratio --stages $stages --ec-gradw $ecgradw --ec-memw $ecmemw --optimizer $optimizer --totaltime $totaltime --cluster $cluster
