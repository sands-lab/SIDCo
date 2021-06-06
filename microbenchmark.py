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

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import torch
import time
import os
import sys
import numpy as np
import argparse
from compression import *
from pathlib import Path

#from tensorflow.python.client import timeline

# def write_metadata(run_metadata, run_type, name):
#     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#     chrome_trace = fetched_timeline.generate_chrome_trace_format()
#     name += f'{run_type}.json'
#     print("writing trace file to {}".format(name))
#     with open(name, 'w') as f:
#         f.write(chrome_trace)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', help='use CPU if this flag is set')
    parser.add_argument('--ratio', default=0.1, type=float, help='compression ratio')
    parser.add_argument('--num_rounds', default=10, type=int, help='number of runs')
    parser.add_argument('--size', default=0, type=int, help='tensor size in MB')
    parser.add_argument('--warmup', default=3, type=int, help='warmup steps')
    parser.add_argument('--grad-file', default="", help='grad file to read grad elements from')
    parser.add_argument('--file-prefix', default="", help='prefix of the file name to write the trace')
    parser.add_argument('--trace-file', default="", help='write trace file using this file name')
    parser.add_argument('--method', default="topkec,randomkec,dgcsampling,gaussianksgdec,redsync,redsynctrim,gammagparetoec,gparetoec,expec,gaussianec", type=str, help='comma seperated lists of methods to run')
    args = parser.parse_args()

    grad = None
    if args.grad_file is not '':
        grad = np.load(args.grad_file)

    methods = args.method.split(",")

    if grad is not None:
        NUM_ELEMENTS = len(grad)
    else:
        NUM_ELEMENTS = int(args.size * 1024 * 1024 / 4)

    #torch.cuda.set_device(0)
    DEVICE = 'cuda' if  (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    device = torch.device(DEVICE)
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ##Open file for writing
    name = args.trace_file
    if name == '':
        if args.size > 0:
            Path('microbench/randnormal/').mkdir(parents=True, exist_ok=True)
            name = 'microbench/randnormal/randnormal_' + DEVICE + '_' + str(NUM_ELEMENTS) + '_' + str(args.ratio)  + '_' + str(args.num_rounds) +  '.csv'
        else:
            Path('microbench/models/').mkdir(parents=True, exist_ok=True)
            name = 'microbench/models/' + args.file_prefix + '_' + DEVICE + '_' + str(NUM_ELEMENTS) + '_' + str(args.ratio)  + '_' + str(args.num_rounds) +  '.csv'

    f = open(name, 'w')
    f.write("Method,\t Ratio,\t Compress T_AVG,\t Compress T_STD,\t Compress TPUT_AVG ME/sec,\t Decompress T_AVG,\t Decompress T_STD,\t Decompress TPUT_AVG ME/sec\n")
    ##########################

    if grad is not None:
        tensor_to_compress = torch.tensor(grad, device=DEVICE)

    for method in methods:
        compobj = compressors[method]
        print("running {} {} times for {} elements on {}".format(method, args.num_rounds, NUM_ELEMENTS, DEVICE))
        total_time_compress = []
        total_time_decompress = []
        compress_ratio = []
        for i in range(1, args.num_rounds+args.warmup+2):
            if grad is not None:
                tensor = tensor_to_compress.clone()
            else:
                tensor = torch.randn(NUM_ELEMENTS, device=DEVICE)

            compress_start = time.time()
            compressed_tensor, indexes, vals = compobj.compress(tensor, ratio=args.ratio, stages=2 + math.ceil(1/math.log(args.ratio,10)))
            if device == 'cuda':
                torch.cuda.synchronize()
            compress_end = time.time()

            cdelay = compress_end-compress_start
            if i > args.num_rounds+args.warmup:
                pass
            elif i > args.warmup:
                print("compress ROUND {} {} seconds".format(i-args.warmup, cdelay))
                total_time_compress.append(cdelay)
                if method == 'none':
                    compress_ratio.append(1.0)
                else:
                    compress_ratio.append(1.0 * vals.numel() / NUM_ELEMENTS)
            else:
                print("warmup compress ROUND {} {} seconds".format(i, cdelay))
            # if run_metadata_compress:
            #     write_metadata(run_metadata_compress, "compress", method)

            dectx = vals, indexes, 1


            decompress_start = time.time()
            decompressed_tensor = compobj.decompress(compressed_tensor, dectx)
            if device == 'cuda':
                torch.cuda.synchronize()
            decompress_end = time.time()

            ddelay = decompress_end-decompress_start
            if i > args.num_rounds+args.warmup:
                pass
            elif i > args.warmup:
                print("decompress ROUND {} {} seconds".format(i-args.warmup, ddelay))
                total_time_decompress.append(ddelay)
            else:
                print("warmup decompress ROUND {} {} seconds".format(i, ddelay))

            # if run_metadata_decompress:
            #     write_metadata(run_metadata_decompress, "decompress", method)

        print("{} compress {} MB/sec, decompress {} MB/sec".format(method,NUM_ELEMENTS/ (1024*1024*np.average(total_time_compress)), NUM_ELEMENTS /(1024*1024*np.average(total_time_decompress))))
        f.write("{},\t {},\t {},\t {},\t {},\t {},\t {},\t {}\n".format(method, np.average(compress_ratio), np.average(total_time_compress), np.std(total_time_compress), NUM_ELEMENTS /(1024*1024*np.average(total_time_compress)),\
                                                      np.average(total_time_decompress), np.std(total_time_decompress), NUM_ELEMENTS /(1024*1024*np.average(total_time_decompress))))