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
import sys

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import settings

# util is one level up, so import that
module_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, os.path.abspath(f'{module_path}/..'))


class TensorboardLogger:
  def __init__(self, output_dir, is_master=False):
    self.output_dir = output_dir
    self.current_step = 0
    self.is_master = is_master
    print('this is the master: ' + str(is_master))
    #if is_master:
    #  self.writer = SummaryWriter(self.output_dir)
    #else:
    self.writer = NoOp()
    #    self.log('first', time.time())

  def log(self, tag, val):
    """Log value to tensorboard (relies on global_example_count being set properly)"""
    if not self.writer:
      print('writer is not initialzied')
      return
    self.writer.add_scalar(tag, val, self.current_step)
    #Ahmed - WANDB logging
    try:
      wandb.log({tag: val}, step=int(self.current_step))
      #print('logging to wandb: ' + str(tag) + ':' + str(val) + '-' + str(self.current_step))
    except:
      pass

  def log_display(self, tag, val):
    """Log value to tensorboard (relies on global_example_count being set properly)"""
    if not self.writer:
      print('writer is not initialzied')
      return
    if self.current_step > 0 and self.current_step % settings.DISPLAY != 0:
      return
    self.writer.add_scalar(tag, val, self.current_step)
    try:
      wandb.log({tag: val}, step=int(self.current_step))
      #print('logging to wandb: ' + str(tag) + ':' + str(val) + '-' + str(self.current_step))
    except:
      pass

  def update_step_count(self, batch_total):
    self.current_step += batch_total

  def close(self):
    self.writer.export_scalars_to_json(self.output_dir+'/scalars.json')
    self.writer.close()

  # Convenience logging methods
  def log_size(self, bs=None, sz=None):
    if bs: self.log('sizes/batch', bs)
    if sz: self.log('sizes/image', sz)
    
  def log_eval(self, top1, top5, time):
    self.log('losses/test_1', top1)
    self.log('losses/test_5', top5)
    self.log('times/eval_ms', 1000*time)

  def log_trn_loss(self, loss, top1, top5):
    self.log("losses/loss", loss)      # cross_entropy
    self.log("losses/train_1", top1)   # precision@1
    self.log("losses/train_5", top5)   # precision@5

  def log_memory(self):
    if not self.writer: return
    self.log("memory/allocated_gb", torch.cuda.memory_allocated()/1e9)
    self.log("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9)
    self.log("memory/cached_gb", torch.cuda.memory_cached()/1e9)
    self.log("memory/max_cached_gb", torch.cuda.max_memory_cached()/1e9)

  def log_trn_times(self, batch_time, data_time, batch_size):
    if not self.writer: return
    self.log("times/step", 1000*batch_time)
    self.log("times/data", 1000*data_time)
    images_per_sec = batch_size/batch_time
    self.log("times/1gpu_images_per_sec", images_per_sec)
    self.log("times/8gpu_images_per_sec", 8*images_per_sec)

  def log_iter_times(self, forward_time, backward_time, io_time, total_time):
    if not self.writer: return
    self.log("iter/forward_ms", 1000 * forward_time)
    self.log("iter/backward_ms", 1000 * backward_time)
    self.log("iter/io_ms", 1000 * io_time)
    self.log("iter/tot_nocomm_ms", 1000 * total_time)

import logging


class FileLogger:
  def __init__(self, output_dir, is_master=False, is_rank0=False):
    self.output_dir = output_dir

    # Log to console if rank 0, Log to console and file if master
    if not is_rank0: self.logger = NoOp()
    else: self.logger = self.get_logger(output_dir, log_to_file=is_master)

  def get_logger(self, output_dir, log_to_file=True):
    logger = logging.getLogger('imagenet_training')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')

    if log_to_file:
      vlog = logging.FileHandler(output_dir+'/verbose.log')
      vlog.setLevel(logging.INFO)
      vlog.setFormatter(formatter)
      logger.addHandler(vlog)

      eventlog = logging.FileHandler(output_dir+'/event.log')
      eventlog.setLevel(logging.WARN)
      eventlog.setFormatter(formatter)
      logger.addHandler(eventlog)

      debuglog = logging.FileHandler(output_dir+'/debug.log')
      debuglog.setLevel(logging.DEBUG)
      debuglog.setFormatter(time_formatter)
      logger.addHandler(debuglog)
      
    console = logging.StreamHandler()
    console.setFormatter(time_formatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    return logger

  def console(self, *args):
    if args and args[0]:
      args0 = 'rank-'+os.environ.get('RANK', '0')+' '+str(args[0])
    new_args = (args0,)+args[1:]
    self.logger.debug(*new_args)

  def event(self, *args):
    self.logger.warning(*args)

  def verbose(self, *args):
    self.logger.info(*args)

# no_op method/object that accept every signature
class NoOp:
  def __getattr__(self, *args):
    def no_op(*args, **kwargs): pass
    return no_op
