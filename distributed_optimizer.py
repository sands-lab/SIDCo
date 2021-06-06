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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import init, broadcast

import time
import torch
import numpy as np
import utils
import math

import collections
import settings
from settings import * #logger, ADAPTIVE_MERGE, ADAPTIVE_SPARSE, DEBUG, DENISTY_INC, DENSITY_MAX

from compression import compressors

SPEED = 0
class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression, is_sparse=False, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None, tb=None, iratio=0.1, stages=1, partitions=0, ec_gradw=1.0, ec_memw=0.0):
        super(self.__class__, self).__init__(params)
        #Ahmed's added parameters
        self._tb = tb
        self.iratio = iratio
        self.stages = stages
        self.ec_gradw = ec_gradw
        self.ec_memw = ec_memw
        self.partitions = partitions

        self._sum_elems = 0
        self._num_sample = 0
        self._avg_ratio = 0
        self._num_avg_sample = 0
        self._sum_volume = 0
        self._num_vol_sample = 0
        self._comp_group_num = 0

        self.model_elemnum = 0
        #################################
        self._compression = compression
        self._sparse = is_sparse
        self._density = density
        self._profiling = True
        self._seq_layernames = seq_layernames
        self._layerwise_times = layerwise_times 
        self._original_layerwise_times_kv = None
        self._norm_clip = norm_clip
        self._threshold = threshold
        self._writer = writer
        self._gradient_path = gradient_path
        if self._layerwise_times is not None and self._seq_layernames is not None:
            self._original_layerwise_times_kv = dict(zip(self._seq_layernames, self._layerwise_times))
        self._layerwise_compressors= {}
        self._compression_timers = {} # compression
        self._allreduce_timers = {} # allreduce times
        self._update_times = {} # allreduce times
        self.train_epoch = 0
        self.train_iter = 0

        self._dynamic_densities = None
        logger.info('_dynamic_densities: %s', self._dynamic_densities)
        self._selected_num_gradients = []

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._named_parameters = {k: v for k, v
                                in named_parameters}

        #Ahmed - total number of elements in the model
        for k, v in named_parameters:
            self.model_elemnum += v.numel()

        self._total_num_param = 0
        self._da_allreduce_timers = {}
        self._da_allreduce_mintime = {}  # min observed time

        for name, param in named_parameters:
            self._total_num_param += param.data.numel()

        if self._seq_layernames is not None:
            self._sequential_keys = self._seq_layernames
        else:
            self._sequential_keys = [k for k, v in named_parameters]

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        self._generate_merged_parameters()

        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self.local = False
        if size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)


    def _generate_groups_with_threshold(self, threshold):
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        sub_size = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for k in self._sequential_keys[::-1]:
            # Ahmed - Fix the logic of grouping as it was exceeding the threshold per group
            numel = self._named_parameters[k].data.numel()
            sub_size += numel
            key_groupidx_maps[k] = idx
            if sub_size < threshold:
                group.append(k)
            elif sub_size < threshold * 1.5:
                group.append(k)
                groups.append(group)
                idx += 1
                group = []
                sub_size = 0
            else:
                group.append(k)
                groups.append(group)
                group = []
                idx += 1
                key_groupidx_maps[k] = idx
                group.append(k)
                sub_size = numel
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def increase_one_epoch(self):
        self.train_epoch += 1
        if rank() == 0:
            density = self.get_current_density()
            size = np.sum(self._sizes)
            k = max(int(size * density), 1)
            logger.info('Average number of selected gradients: %f, exact k: %d', np.mean(self._selected_num_gradients) * len(self._groups), k)
            logger.info('The number of selected gradients: %s', self._selected_num_gradients)

        self._selected_num_gradients = []

    def get_current_density(self, name=None):
        density = self._density
        if self._dynamic_densities is not None:
            if self.train_epoch >= len(self._dynamic_densities):
                density = self._dynamic_densities[-1]
            else:
                density = self._dynamic_densities[self.train_epoch]
        if name is not None and self._layerwise_compressors is not None:
            if name not in self._layerwise_compressors:
                errstr = 'compressor density not found at layer: %s' % name
                logger.error(errstr)
                raise Exception(errstr)
            ld = self._layerwise_compressors[name]
            density = max(ld, density)
        return density

    def _generate_groups_mgwfbp(self):
        num_of_workers = size()
        p_alpha_beta = {
                16: (0.00010632079996292579, 1.5*3.2713239529771973e-10),
                8: (9.75367204301171e-05, 3.0568230536676206e-10),
                4: (4.204298980348825e-05, 2.0589360830118177e-10),
                2: (2.554691138304671e-06, 9.837548167872609e-11)
                }
        alpha, beta = p_alpha_beta[num_of_workers]
        def __calculate_comm_start(tc, tb, taob, L):
            taoc = [0] * L 
            taoc[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taob[l] + tb[l])
            return taoc
        def __merge(taob, tc, p, l):
            tc[l] = 0
            p[l-1] = p[l-1]+p[l]
            p[l] = 0
            tc[l-1] = utils.predict_allreduce_time_with_size(alpha, beta, p[l-1]*4, num_of_workers)
        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]
        seq_layernames = self._seq_layernames
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        tc = [utils.predict_allreduce_time_with_size(alpha, beta, s*4, num_of_workers) for s in sizes]
        tb = list(self._layerwise_times)
        taob = [0]*L
        for l in range(0,L-1)[::-1]:
            taob[l] = taob[l+1] + tb[l+1]
        taoc = __calculate_comm_start(tc, tb, taob, L)
        if rank() == 0:
            logger.warning('tc sum: %f', np.sum(tc))
            logger.warning('tc: %s', tc)
            logger.warning('taoc: %s', taoc)
        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l] 
        key_groupidx_maps[key] = idx
        group.append(key)
        for l in range(1, L-1)[::-1]:
            key = seq_layernames[l]
            group.append(key)
            key_groupidx_maps[key] = idx
            current_taob = taob[l-1] + tb[l-1]
            if current_taob < taoc[l+1]+tc[l+1]:
                __merge(taob, tc, p, l)
                taoc = __calculate_comm_start(tc, tb, taob, L)
            elif current_taob > taoc[l+1]+tc[l+1] and current_taob < taoc[l]+tc[l] and taoc[l]+alpha > current_taob:
                __merge(taob, tc, p, l)
                taoc = __calculate_comm_start(tc, tb, taob, L)
            else:
                idx += 1
                groups.append(group)
                group = []
        l = 0
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        logger.info('Predicted non-overlapped time: %f', taoc[0]+tc[0]-(taob[0]+tb[0]))
        logger.info('Predicted tb+tc= %f', taoc[0]+tc[0])
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def _generate_groups_mgs(self):
        P = size() # number of wokers

        def __calculate_sparse_and_backward_start(tb, sizes, L, start=0):
            taos = [start] * L 
            ts = [utils.topk_perf_model(s) for s in sizes]
            taob = [start] * L 
            taob[L-1] = start 
            taos[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taob[l] = taos[l+1] + ts[l+1]
                taos[l] = taob[l] + tb[l]
            return taob, taos, ts

        def __calculate_comm_start(ts, taos, sizes, L):
            taoc = [0] * L 
            tc = [utils.allgather_perf_model(s, P, self._density) for s in sizes]
            taoc[L-1] = taos[L-1] + ts[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taos[l] + ts[l])
            return taoc, tc

        def __merge(tb, ts, tc, p, l):
            tb[l-1] += tb[l]
            tb[l] = 0

            p[l-1] = p[l-1]+p[l]
            p[l] = 0

            tc[l-1] = utils.allgather_perf_model(p[l-1], P, self._density) 
            tc[l] = 0

            ts[l-1] = utils.topk_perf_model(p[l-1])
            ts[l] = 0

        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]
        seq_layernames = self._seq_layernames
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        tb = list(self._layerwise_times)
        taob, taos, ts = __calculate_sparse_and_backward_start(tb, p, L)
        taoc, tc = __calculate_comm_start(ts, taos, p, L)

        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l] 
        key_groupidx_maps[key] = idx
        group.append(key)
        for l in range(1, L-1)[::-1]:
            key = seq_layernames[l]
            group.append(key)
            key_groupidx_maps[key] = idx

            tw = tb[l-1]+utils.topk_perf_model(p[l]+p[l-1])\
                - utils.topk_perf_model(p[l]) - utils.topk_perf_model(p[l-1])\
                - (taoc[l] - (taos[l]+ts[l]))
            tsave = utils.allgather_perf_model(p[l], P, self._density)+utils.allgather_perf_model(p[l-1], P, self._density)-\
                    utils.allgather_perf_model((p[l]+p[l-1]), P, self._density)
            if tw < tsave:
                __merge(tb, ts, tc, p, l)
                taob2, taos2, ts2 = __calculate_sparse_and_backward_start(tb[:l], p[:l], l, start=taob[l]+tb[l])
                taob[:l] = taob2
                taos[:l] = taos2
                taoc, tc = __calculate_comm_start(ts, taos, p, L)
            else:
                idx += 1
                groups.append(group)
                group = []
        logger.info('Predicted non-overlapped time: %f', taoc[0]+tc[0]-(taos[0]+ts[0]))
        logger.info('Predicted compression time: %f', np.sum(ts))
        l = 0
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def _generate_merged_parameters(self):
        self._merged_parameters = {}
        self._merged_parameter_names = {}

        if ADAPTIVE_MERGE and self._layerwise_times is not None:
            if self._density < 1: # MGS 
                groups, key_groupidx_maps = self._generate_groups_mgs()
            else:
                groups, key_groupidx_maps = self._generate_groups_mgwfbp()
        elif self._threshold >= 0:
            groups, key_groupidx_maps = self._generate_groups_with_threshold(self._threshold)
        elif self.partitions > 0:
            groups, key_groupidx_maps = self._generate_groups_with_threshold(math.ceil(1.0 * self.model_elemnum / self.partitions))
        logger.info('# of parameters: %d', np.sum(self._sizes))
        logger.info('Total number of tensors: %s', len(self._sizes))
        logger.info('Merged Number of groups: %s', len(groups))
        new_keys = []
        self._merged_parameter_offsets = {}
        self._layerwise_compressors = None
        self._layerwise_compressors = {}
        num_of_workers = size()
        for g in groups:
            sub_size = 0
            offsets = []
            computation_time = 0
            for k in g:
                offsets.append(sub_size)
                numel = self._named_parameters[k].data.numel()
                sub_size += numel
                if self._original_layerwise_times_kv is not None and k in self._original_layerwise_times_kv and ADAPTIVE_SPARSE:
                    computation_time += self._original_layerwise_times_kv[k]
            new_key = ':'.join(g)
            new_keys.append(new_key)
            t = torch.zeros(sub_size, device=self._named_parameters[g[0]].device, dtype=self._named_parameters[g[0]].dtype, requires_grad=False)
            self._merged_parameters[new_key] = t
            self._merged_parameter_names[t] = new_key
            self._merged_parameter_offsets[new_key] = offsets
            if self._density < 1 and ADAPTIVE_SPARSE:
                _density = utils.predict_density_with_size_and_computation(sub_size, computation_time, num_of_workers)
                density = max(_density, self._density)
            else:
                density = self._density
            if self._layerwise_compressors is not None:
                self._layerwise_compressors[new_key] = density
        self._groups = groups
        self._key_groupidx_maps = key_groupidx_maps
        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags)


    def _push_to_buffer(self, name, tensor):
        with torch.no_grad():
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1)
                return name, new_tensor 
            group_idx = self._key_groupidx_maps[name]
            g = self._groups[group_idx]
            new_key = ':'.join(g)
            layer_idx = g.index(name)
            offset = self._merged_parameter_offsets[new_key][layer_idx]
            numel = tensor.data.numel()
            self._merged_parameters[new_key].data[offset:offset+numel] = tensor.view(numel).data
            self._groups_flags[group_idx][layer_idx] = 1
            try:
                idx = self._groups_flags[group_idx].index(0)
            except:
                idx = -1
            if idx >= 0:
                return name, None
            return new_key, self._merged_parameters[new_key]

    def _pull_from_buffer(self, name, merged_tensor):
        if len(self._groups) == len(self._sequential_keys):
            shape = self._named_parameters[name].data.shape
            return {name: merged_tensor.view(shape)} 
        offsets = self._merged_parameter_offsets[name]
        g = name.split(':')
        group_idx = self._key_groupidx_maps[g[0]]
        self._groups_flags[group_idx] = [0]*len(self._groups_flags[group_idx])
        tensors = {}
        for i, k in enumerate(g):
            offset = offsets[i]
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel()
            tensor = torch.zeros(numel, device=original_tensor.device, dtype=original_tensor.dtype)
            tensor.data = merged_tensor.data[offset:offset+numel]
            tensors[k] = tensor.view(original_tensor.shape)
        return tensors

    def _allgather_grad_async(self, p, name):
        tensor = p.data.view(-1)
        tensor_compressed, ctx = tensor, None #self._compression.compress(tensor, name)
        if settings.LOGGING_GRADIENTS and rank() == 0:
            grads = tensor.cpu().numpy()
            np.save('%s/r%d_gradients_iter_%d' % (self._gradient_path, rank(), self.train_iter), grads)
        handle = allgather_async(tensor_compressed, name=name)
        return handle, ctx

    def _allreduce_grad_async(self, p, name):
        tensor = p.data.view(-1)
        tensor_compressed, ctx = tensor, None #self._compression.compress(tensor, name)
        if settings.LOGGING_GRADIENTS and rank() == 0:
            grads = tensor.cpu().numpy()
            np.save('%s/r%d_gradients_iter_%d' % (self._gradient_path, rank(), self.train_iter), grads)
        handle = allreduce_async_(tensor_compressed, average=True, name=name)
        return handle, ctx

    def _sparse_allreduce_async(self, p, name, density):
        stime = time.time()
        tensor = p.data.view(-1)

        if settings.CPU_COMPRESS:
            density = abs(density)
            tensor_cpy = tensor.to('cpu')
            tensor_compressed, ctx, selected_values = self._compression.compress(tensor_cpy, name, ratio=density, tb=self._tb, i_ratio=self.iratio, stages=self.stages, ec_grad_w=self.ec_gradw, ec_mem_w=self.ec_memw)
            selected_values = selected_values.to('cuda')
            ctx = ctx.to('cuda')
        else:
            tensor_compressed, ctx, selected_values = self._compression.compress(tensor, name, ratio=density, tb=self._tb, i_ratio=self.iratio, stages=self.stages, ec_grad_w=self.ec_gradw, ec_mem_w=self.ec_memw)
        if self._profiling:
            utils.force_insert_item(self._compression_timers, name, time.time() - stime)

        self._selected_num_gradients.append(int(ctx.numel()))

        indexes = ctx
        handle = allgather_async(selected_values, name)
        handle_idx = allgather_async(indexes.int(), name+'_indexes')
        return (handle, handle_idx), ctx 

    def _make_hook(self, p):
        def hook(*ignore):
            assert p not in self._handles
            assert not p.grad.requires_grad
            if not self.local:
                name = self._parameter_names.get(p)
                new_name, new_tensor = self._push_to_buffer(name, p.grad.data)
                if new_tensor is not None:
                    density = self.get_current_density(name=new_name)
                    if self._sparse and density < 1:
                        handle, ctx = self._sparse_allreduce_async(new_tensor, new_name, density)
                        self._handles[new_tensor] = (handle, ctx, density)
                    elif density==1:
                        handle, ctx = self._allreduce_grad_async(new_tensor, new_name)
                        self._handles[new_tensor] = (handle, ctx, density)
                    elif density==2:
                        #Ahmed - USE all_gather for no compression to have similar communication collective as compression
                        #Note, indexes are not all_gathered in this case
                        handle, ctx = self._allgather_grad_async(new_tensor, new_name)
                        self._handles[new_tensor] = (handle, ctx, density)


        return hook

    def synchronize(self):
        global SPEED
        num_of_workers = size()
        ratio=0
        i=0
        for p, value in self._handles.items():
            name = self._merged_parameter_names.get(p)
            handle, ctx, density = value

            if self._sparse and density < 1:
                stime = time.time()
                handle_idx = None
                all_indexes = None
                if type(handle) is tuple:
                    handle, handle_idx = handle[0], handle[1]
                output = synchronize(handle)
                if handle_idx is not None:
                    all_indexes = synchronize(handle_idx)
                if self._profiling:
                    utils.force_insert_item(self._allreduce_timers, name, time.time()-stime)
                stime = time.time()
                new_grad = p.data.view(-1)
                dectx = output, all_indexes, num_of_workers
                new_grad = self._compression.decompress(new_grad, dectx)
                if self._profiling:
                    utils.force_insert_item(self._update_times, name, time.time()-stime)
            elif density == 1:
                stime = time.time()
                output = synchronize(handle)
                if self._profiling:
                    utils.force_insert_item(self._allreduce_timers, name, time.time()-stime)
                stime = time.time()
                if self._norm_clip is not None:
                    norm_clip = np.sqrt(1.0/size()) * self._norm_clip
                    norm_type = 2.0
                    param_norm = output.norm(norm_type)
                    total_norm = param_norm.item() 
                    clip_coef = norm_clip / (total_norm + 1e-6)
                    if clip_coef < 1:
                        output.mul_(clip_coef)

                p.set_(output)
                if self._profiling:
                    utils.force_insert_item(self._update_times, name, time.time()-stime)
            elif density > 1:
                #allgather instead of allreduce of sparse tensor
                stime = time.time()
                output = synchronize(handle)
                if self._profiling:
                    utils.force_insert_item(self._allreduce_timers, name, time.time() - stime)
                stime = time.time()
                new_grad = p.data.view(-1)
                new_grad.fill_(0.0)
                numel = output.size(0)
                real_num_values = numel // num_of_workers
                for i in range(num_of_workers):
                    values = output.data[i * real_num_values : (i + 1) * real_num_values]
                    new_grad += values
                new_grad /= num_of_workers

                if self._norm_clip is not None:
                    norm_clip = np.sqrt(1.0 / size()) * self._norm_clip
                    norm_type = 2.0
                    param_norm = new_grad.norm(norm_type)
                    total_norm = param_norm.item()
                    clip_coef = norm_clip / (total_norm + 1e-6)
                    if clip_coef < 1:
                        new_grad.mul_(clip_coef)

                p.set_(new_grad)
                if self._profiling:
                    utils.force_insert_item(self._update_times, name, time.time() - stime)

            # Ahmed - track number of elments
            if ctx is not None:
                ratio += ctx.numel() / p.data.numel()
            else:
                ratio += 1
            self._avg_ratio += ratio
            self._num_avg_sample += 1

            if density < 1:
                #Volume for all-gather compression (data + indexes) - %TODO should multiply (1-1/num-of-workers) (to remove portion of local node)
                self._sum_volume += output.numel() * output.element_size() + all_indexes.numel() * all_indexes.element_size()
            elif density == 1:
                #Volume for all-reduce no-compression
                self._sum_volume += 2 * output.numel() * output.element_size()
            elif density == 2:
                #Volume for all-gather no compression (data ) - %TODO should multiply (1-1/num-of-workers) (to remove portion of local node)
                self._sum_volume += output.numel() * output.element_size()
            self._num_vol_sample += 1

        if rank() == 0 and self.train_iter % settings.DISPLAY == 0 :
            self._tb.log('datavol/cum_vol_bytes', self._sum_volume)
            self._tb.log('datavol/avg_vol_bytes', self._sum_volume / self._num_vol_sample)

            if self._compression is not compressors['none']: #and ratio > 0:
                #target_k = (self.model_elemnum * density)
                self._tb.log('compress/comp_ratio', ratio)
                self._tb.log('compress/est_compratio', ratio / density )
                self._tb.log('compress/avg_est_compratio', (1.0 * self._avg_ratio / self._num_avg_sample ) / density)
                if self.stages < 0:
                    self._tb.log('compress/num_stages', self._compression.cur_stages)
                else:
                    self._tb.log('compress/num_stages', self.stages)
                if self.stages == 0:
                    self._tb.log('compress/first_ratio', self.iratio)
                else:
                    self._tb.log('compress/first_ratio', self._compression.first_ratio)
                self._num_sample = 0
                self._sum_elems = 0


        if len(self._groups) != len(self._sequential_keys):
            for merged_p, value in self._handles.items():
                new_name = self._merged_parameter_names.get(merged_p)
                tensors = self._pull_from_buffer(new_name, merged_p)
                for n in tensors:
                    p = self._named_parameters.get(n)
                    p.grad.set_(tensors[n].data.type(p.grad.type()))
        self.train_iter += 1
        self._handles.clear()
        self._print_profiling()


    def _print_profiling(self):
        if self._profiling and rank() == 0 and len(self._allreduce_timers.keys()) > 0 and self.train_iter % settings.DISPLAY == 0: #and len(self._allreduce_timers.get(list(self._allreduce_timers.keys())[0], [])) ==  settings.DISPLAY:
            cps = self._compression_timers # compression
            ars = self._allreduce_timers # allreduce times
            ups = self._update_times # update times
            r = rank()
            tcp = 0.0; tar = 0.0; tup = 0.0; total=0.0
            for k in ars:
                if len(cps) > 0:
                    acp = np.mean(cps[k])
                    tcp += acp
                aar = np.mean(ars[k])
                tar += aar
                aup = np.mean(ups[k])
                tup += aup
            total = tcp+tar+tup
            logger.info('[%d]: Total compress: %f, allreduce: %f, update: %f, total: %f', r, tcp, tar, tup, total)
            #Ahmed - log to wandb micromeasurments of RANK 0
            if r == 0:
                self._tb.log('micro/compress_ms', tcp * 1000)
                self._tb.log('micro/comm_ms', tar * 1000)
                self._tb.log('micro/gradagg_ms', tup * 1000)
                self._tb.log('micro/total_ms', total * 1000)
            cps.clear()
            ars.clear()
            ups.clear()


    def step(self, closure=None):
        if not self.local:
            self.synchronize()

        return super(self.__class__, self).step(closure)



def DistributedOptimizer(optimizer, named_parameters=None, compression=None, is_sparse=False, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None, tb=None, iratio=0.1, stages=1, partitions=0, ec_gradw=1.0, ec_memw=0.0):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(optimizer.param_groups, named_parameters, compression, is_sparse, density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=threshold, writer=writer, gradient_path=gradient_path, tb=tb, iratio=iratio, stages=stages, partitions=partitions, ec_gradw=ec_gradw, ec_memw=ec_memw)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.numpy()[0])
        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value])
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p])
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
