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

from __future__ import print_function
import settings
import torch
import numpy as np
import time
import math
import utils
import sys
from scipy import stats, special

class NoneCompressor():
    norm = 1.0
    sum_ratio = 0.0
    iter = 0
    last_estimate = 0.0
    cur_stages = 1
    last_stages = 1
    update = 1
    count_stupdates=0
    first_ratio=settings.FIRST_RATIO
    fr_update=settings.FR_UPDATE
    count_frupdates=0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        return tensor, tensor.dtype, None

    @staticmethod
    def decompress(tensor, ctx, name=None):
        return tensor

class SparseCompressor(NoneCompressor):
    @staticmethod
    def decompress(tensor, ctx, name=None):
        tensor.fill_(0.0)
        output, all_indexes, num_of_workers = ctx
        numel = output.size(0)
        real_num_values = numel // num_of_workers
        for i in range(num_of_workers):
            values_and_indexes = output.data[i * real_num_values:(i + 1) * real_num_values]
            if all_indexes is None:
                values = values_and_indexes[0:real_num_values // 2]
                indexes = values_and_indexes[real_num_values // 2:].long()
            else:
                values = values_and_indexes
                indexes = all_indexes.data[i * real_num_values:(i + 1) * real_num_values].long()
            tensor[indexes[0:indexes.numel() // 2]] += values[0:indexes.numel() // 2]
            tensor[indexes[indexes.numel() // 2:]] += values[indexes.numel() // 2:]
        tensor /= num_of_workers
        return tensor

class MultStageSparseCompressor(SparseCompressor):
    @staticmethod
    def adapt_stages(actual_ratio=0.0, ratio=0.0, stages=0):
        NoneCompressor.sum_ratio += actual_ratio / ratio
        NoneCompressor.iter += 1
        if NoneCompressor.iter == settings.UPDATE_ITER:
            cur_estimate = 1.0 * NoneCompressor.sum_ratio / NoneCompressor.iter

            if stages == -1:
                if cur_estimate > 1 + settings.RES_DIFF_UP:
                    NoneCompressor.cur_stages -= 1
                    NoneCompressor.count_stupdates += 1

                elif cur_estimate < 1 - settings.RES_DIFF_DOWN:
                    NoneCompressor.cur_stages += 1
                    NoneCompressor.count_stupdates += 1

            if stages == -2:
                if NoneCompressor.last_estimate > 0 and abs(cur_estimate - 1) > abs(NoneCompressor.last_estimate - 1):
                    NoneCompressor.update *= -1

                if cur_estimate > 1 + settings.RES_DIFF_UP:
                    NoneCompressor.cur_stages -= NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

                if cur_estimate < 1 - settings.RES_DIFF_DOWN:
                    NoneCompressor.cur_stages += NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

            if stages == -3:
                if NoneCompressor.last_estimate > 0 and not (
                        cur_estimate > NoneCompressor.last_estimate and NoneCompressor.last_estimate < 1 - settings.RES_DIFF_DOWN
                        or cur_estimate < NoneCompressor.last_estimate and NoneCompressor.last_estimate > 1 + settings.RES_DIFF_UP):
                    NoneCompressor.update *= -1

                if cur_estimate > 1 + settings.RES_DIFF_UP:
                    NoneCompressor.cur_stages -= NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

                if cur_estimate < 1 - settings.RES_DIFF_DOWN:
                    NoneCompressor.cur_stages += NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

            NoneCompressor.cur_stages = max(min(NoneCompressor.cur_stages, math.ceil(math.log(ratio) / math.log(0.5))), 1)

            #adjust the initial ratio if the changes to the stages can not fix the drift
            if settings.ADJUST_FR and NoneCompressor.count_stupdates >= math.ceil(math.log(ratio) / math.log(0.5) / 2):
                if NoneCompressor.last_estimate > 0 and abs(cur_estimate - 1) > abs(NoneCompressor.last_estimate - 1):
                    if NoneCompressor.first_ratio == 0.5 or NoneCompressor.first_ratio == 0.05:
                        NoneCompressor.first_ratio = 0.25
                        NoneCompressor.fr_update *= -1
                        NoneCompressor.count_frupdates = 0

                    elif cur_estimate > 1 + (settings.RES_DIFF_UP / 2):
                        NoneCompressor.first_ratio -= NoneCompressor.fr_update
                        NoneCompressor.count_frupdates += 1

                        #reset stages to middle point of MAX and re-search
                        NoneCompressor.cur_stages = math.ceil(math.log(ratio) / math.log(0.5) / 2) #1
                        NoneCompressor.count_stupdates = 0

                    elif cur_estimate < 1 - (settings.RES_DIFF_DOWN / 2):
                        NoneCompressor.first_ratio += NoneCompressor.fr_update
                        NoneCompressor.count_frupdates += 1

                        # reset stages to middle point of MAX and re-search
                        NoneCompressor.cur_stages = math.ceil(math.log(ratio) / math.log(0.5) / 2) #1
                        NoneCompressor.count_stupdates = 0

                #Bound the adjustment on the first ratio
                NoneCompressor.first_ratio = max(min(NoneCompressor.first_ratio, 0.5), 0.05)

            NoneCompressor.last_stages = NoneCompressor.cur_stages
            NoneCompressor.last_estimate = cur_estimate
            NoneCompressor.sum_ratio = 0
            NoneCompressor.iter = 0

class ExpCompressorEC(MultStageSparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'expec'

    @staticmethod
    def clear():
        ExpCompressorEC.residuals = {}
        ExpCompressorEC.sparsities = []
        ExpCompressorEC.zero_conditions = {}
        ExpCompressorEC.values = {}
        ExpCompressorEC.indexes = {}
        ExpCompressorEC.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in ExpCompressorEC.residuals:
                ExpCompressorEC.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()

            ada_stages = 0
            if stages < 0 or i_ratio == 0.0:
                ada_stages = stages
                stages = ExpCompressorEC.cur_stages

            tensor.add_(ExpCompressorEC.residuals[name].data)

            t_norm = tensor.norm(2)
            ExpCompressorEC.norm = t_norm
            abs_norm_tensor = tensor.abs() / t_norm
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            t_mean = torch.mean(abs_norm_tensor)

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                threshold = -t_mean * math.log(ratio)
            else:
                threshold = -t_mean * math.log(NoneCompressor.first_ratio)

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    t_mean = torch.mean(abs_norm_tensor)

                    threshold = -(t_mean - t_min) * math.log(i_ratio) + t_min
                    if i == 1 and stages == 0:
                        threshold = -(t_mean - t_min) * math.log(r_ratio / math.pow(i_ratio, loop - 1)) + t_min
                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            ExpCompressorEC.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * ExpCompressorEC.residuals[
                name].data
            ExpCompressorEC.residuals[name].data[indexes] = 0.0

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                ExpCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values

class ExpCompressor(ExpCompressorEC):

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            numel = tensor.numel()

            t_norm = tensor.norm(2)
            ExpCompressorEC.norm = t_norm

            ada_stages = 0
            if stages < 0   or i_ratio == 0.0:
                ada_stages = stages
                stages = ExpCompressorEC.cur_stages

            abs_norm_tensor = tensor.abs() / tensor.norm(2)
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            t_mean = torch.mean(abs_norm_tensor)

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                threshold = -t_mean * math.log(ratio)
            else:
                threshold =-t_mean * math.log(NoneCompressor.first_ratio)

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    t_mean = torch.mean(abs_norm_tensor)

                    threshold = -(t_mean - t_min) * math.log(i_ratio) + t_min
                    if i == 1 and stages == 0:
                        threshold = -(t_mean - t_min) * math.log(r_ratio / math.pow(i_ratio, loop - 1)) + t_min
                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                ExpCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values

class GParetoCompressorEC(MultStageSparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'gparetoec'

    @staticmethod
    def clear():
        GParetoCompressorEC.residuals = {}
        GParetoCompressorEC.sparsities = []
        GParetoCompressorEC.zero_conditions = {}
        GParetoCompressorEC.values = {}
        GParetoCompressorEC.indexes = {}
        GParetoCompressorEC.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in GParetoCompressorEC.residuals:
                GParetoCompressorEC.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()

            ada_stages = 0
            if stages < 0  or i_ratio == 0.0:
                ada_stages = stages
                stages = GParetoCompressorEC.cur_stages

            #mem_weight = tensor.norm(2) / GParetoCompressorEC.norm
            #tb.log_display('compress/grad_norm_ratio', mem_weight)

            tensor.add_(GParetoCompressorEC.residuals[name].data)

            t_norm = tensor.norm(2)
            GParetoCompressorEC.norm = t_norm
            abs_norm_tensor = tensor.abs() / t_norm
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            if torch.__version__ < '1.3.0':
                t_var = torch.var(abs_norm_tensor)
                t_mean = torch.mean(abs_norm_tensor)
            else:
                t_var, t_mean = torch.var_mean(abs_norm_tensor)

            alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
            k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                threshold = alpha * (1.0 - torch.exp(k * math.log(ratio))) / k
            else:
                threshold = alpha * (1.0 - torch.exp(k * math.log(NoneCompressor.first_ratio))) / k

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    abs_norm_tensor_min = abs_norm_tensor - t_min

                    if torch.__version__ < '1.3.0':
                        t_var = torch.var(abs_norm_tensor_min)
                        t_mean = torch.mean(abs_norm_tensor_min)
                    else:
                        t_var, t_mean = torch.var_mean(abs_norm_tensor_min)

                    alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
                    k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

                    threshold = alpha * (1.0 - torch.exp(k * math.log(i_ratio))) / k + t_min
                    if i == 1 and stages == 0:
                        threshold = alpha * (1.0 - torch.exp(k * math.log(r_ratio / math.pow(i_ratio, loop - 1)))) / k + t_min
                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            GParetoCompressorEC.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * GParetoCompressorEC.residuals[name].data
            GParetoCompressorEC.residuals[name].data[indexes] = 0.0

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                GParetoCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values


class GParetoCompressor(GParetoCompressorEC):

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            numel = tensor.numel()
            t_norm = tensor.norm(2)
            GParetoCompressorEC.norm = t_norm

            ada_stages = 0
            if stages < 0 or i_ratio == 0.0:
                ada_stages = stages
                stages = GParetoCompressorEC.cur_stages

            abs_norm_tensor = tensor.abs() / tensor.norm(2)
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            if torch.__version__ < '1.3.0':
                t_var = torch.var(abs_norm_tensor)
                t_mean = torch.mean(abs_norm_tensor)
            else:
                t_var, t_mean = torch.var_mean(abs_norm_tensor)

            alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
            k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                threshold = alpha * (1.0 - torch.exp(k * math.log(ratio))) / k
            else:
                threshold = alpha * (1.0 - torch.exp(k * math.log(NoneCompressor.first_ratio))) / k

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    abs_norm_tensor_min = abs_norm_tensor - t_min

                    if torch.__version__ < '1.3.0':
                        t_var = torch.var(abs_norm_tensor_min)
                        t_mean = torch.mean(abs_norm_tensor_min)
                    else:
                        t_var, t_mean = torch.var_mean(abs_norm_tensor_min)

                    alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
                    k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

                    threshold = alpha * (1.0 - torch.exp(k * math.log(i_ratio))) / k + t_min
                    if i == 1 and stages == 0:
                        threshold = alpha * (
                                1.0 - torch.exp(k * math.log(r_ratio / math.pow(i_ratio, loop - 1)))) / k + t_min
                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                GParetoCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values

class GammaGParetoCompressorEC(MultStageSparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'gammaparetoec'

    @staticmethod
    def clear():
        GammaGParetoCompressorEC.residuals = {}
        GammaGParetoCompressorEC.sparsities = []
        GammaGParetoCompressorEC.zero_conditions = {}
        GammaGParetoCompressorEC.values = {}
        GammaGParetoCompressorEC.indexes = {}
        GammaGParetoCompressorEC.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in GammaGParetoCompressorEC.residuals:
                GammaGParetoCompressorEC.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()

            ada_stages = 0
            if stages < 0 or i_ratio == 0.0:
                ada_stages = stages
                stages = GammaGParetoCompressorEC.cur_stages

            tensor.add_(GammaGParetoCompressorEC.residuals[name].data)

            t_norm = tensor.norm(2)
            GammaGParetoCompressorEC.norm = t_norm
            abs_norm_tensor = tensor.abs() / t_norm
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            t_mean = torch.mean(abs_norm_tensor)
            s = torch.log(t_mean) - torch.mean(torch.log(abs_norm_tensor + sys.float_info.epsilon))

            alpha = (3 - s + torch.sqrt(torch.pow(s - 3, 2) + 24.0 * s)) / (12.0 * s)
            beta = t_mean / alpha

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                threshold = float(beta * special.gammaincinv(float(alpha), 1 - ratio))
            else:
                threshold = float(beta * special.gammaincinv(float(alpha), 1 - NoneCompressor.first_ratio))

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages==0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    abs_norm_tensor_min = abs_norm_tensor - t_min

                    if torch.__version__ < '1.3.0':
                        t_var = torch.var(abs_norm_tensor_min)
                        t_mean = torch.mean(abs_norm_tensor_min)
                    else:
                        t_var, t_mean = torch.var_mean(abs_norm_tensor_min)

                    alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
                    k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)
                    threshold = alpha * (1.0 - torch.exp(k * math.log(i_ratio))) / k + t_min

                    if i==1 and stages == 0:
                        threshold = alpha * (1.0 - torch.exp(k * math.log(r_ratio / math.pow(i_ratio, loop - 1)))) / k + t_min

                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            GammaGParetoCompressorEC.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * \
                                                           GammaGParetoCompressorEC.residuals[name].data
            GammaGParetoCompressorEC.residuals[name].data[indexes] = 0.0

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                GammaGParetoCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values


class GammaGParetoCompressor(GammaGParetoCompressorEC):

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            numel = tensor.numel()

            t_norm = tensor.norm(2)
            GammaGParetoCompressorEC.norm = t_norm

            ada_stages = 0
            if stages < 0 or i_ratio == 0.0:
                ada_stages = stages
                stages = GammaGParetoCompressorEC.cur_stages

            abs_norm_tensor = tensor.abs() / tensor.norm(2)
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            t_mean = torch.mean(abs_norm_tensor)
            s = torch.log(t_mean) - torch.mean(torch.log(abs_norm_tensor + sys.float_info.epsilon))

            alpha = (3 - s + torch.sqrt(torch.pow(s - 3, 2) + 24 * s)) / (12 * s)
            beta = t_mean / alpha

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                threshold = -beta * (math.log(ratio) + torch.lgamma(alpha))
            else:
                threshold = -beta * (math.log(NoneCompressor.first_ratio) + torch.lgamma(alpha))

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    abs_norm_tensor_min = abs_norm_tensor - t_min

                    if torch.__version__ < '1.3.0':
                        t_var = torch.var(abs_norm_tensor_min)
                        t_mean = torch.mean(abs_norm_tensor_min)
                    else:
                        t_var, t_mean = torch.var_mean(abs_norm_tensor_min)

                    alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
                    k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

                    threshold = alpha * (1.0 - torch.exp(k * math.log(i_ratio))) / k + t_min
                    if i == 1 and stages == 0:
                        threshold = alpha * (
                                    1.0 - torch.exp(k * math.log(r_ratio / math.pow(i_ratio, loop - 1)))) / k + t_min
                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                GammaGParetoCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values


class GaussianCompressorEC(MultStageSparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'gaussian'

    @staticmethod
    def clear():
        GaussianCompressorEC.residuals = {}
        GaussianCompressorEC.sparsities = []
        GaussianCompressorEC.zero_conditions = {}
        GaussianCompressorEC.values = {}
        GaussianCompressorEC.indexes = {}
        GaussianCompressorEC.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in GaussianCompressorEC.residuals:
                GaussianCompressorEC.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            ada_stages = 0
            if stages < 0 or i_ratio == 0.0:
                ada_stages = stages
                stages = GaussianCompressorEC.cur_stages

            tensor.add_(GaussianCompressorEC.residuals[name].data)
            t_norm = tensor.norm(2)
            GaussianCompressorEC.norm = t_norm

            norm_tensor = tensor / t_norm
            abs_norm_tensor = norm_tensor.abs()
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            if torch.__version__ < '1.3.0':
                t_std = torch.std(abs_norm_tensor)
                t_mean = torch.mean(abs_norm_tensor)
            else:
                t_std, t_mean = torch.std_mean(abs_norm_tensor)

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                _,threshold = utils.gen_threshold_from_normal_distribution(1 - ratio, float(t_mean),float(t_std))
            else:
                _,threshold = utils.gen_threshold_from_normal_distribution(1 - NoneCompressor.first_ratio, float(t_mean), float(t_std))

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    abs_norm_tensor_min = abs_norm_tensor - t_min

                    if torch.__version__ < '1.3.0':
                        t_std = torch.std(abs_norm_tensor_min)
                        t_mean = torch.mean(abs_norm_tensor_min)
                    else:
                        t_std, t_mean = torch.std_mean(abs_norm_tensor_min)

                    _, threshold = utils.gen_threshold_from_normal_distribution(1 - i_ratio, float(t_mean),float(t_std))
                    if i == 1 and stages == 0:
                        _, threshold = utils.gen_threshold_from_normal_distribution(1 - r_ratio / math.pow(i_ratio, loop - 1), float(t_mean),
                                                                                    float(t_std))
                    threshold += t_min

                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            GaussianCompressorEC.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * GaussianCompressorEC.residuals[name].data
            GaussianCompressorEC.residuals[name].data[indexes] = 0.0

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                GaussianCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values

class GaussianCompressor(GaussianCompressorEC):

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            ada_stages = 0
            if stages < 0  or i_ratio == 0.0:
                ada_stages = stages
                stages = GaussianCompressorEC.cur_stages

            t_norm = tensor.norm(2)
            GaussianCompressorEC.norm = t_norm

            norm_tensor = tensor / t_norm
            abs_norm_tensor = norm_tensor.abs()
            abs_norm_tensor = tensor #TODO:REMOVE
            abs_norm_tensor_cpy = abs_norm_tensor.clone()

            if torch.__version__ < '1.3.0':
                t_std = torch.std(abs_norm_tensor)
                t_mean = torch.mean(abs_norm_tensor)
            else:
                t_std, t_mean = torch.std_mean(abs_norm_tensor)

            if stages == 1 or ratio >= NoneCompressor.first_ratio:
                _, threshold = utils.gen_threshold_from_normal_distribution(1 - ratio, float(t_mean), float(t_std))
            else:
                _, threshold = utils.gen_threshold_from_normal_distribution(1 - NoneCompressor.first_ratio, float(t_mean),
                                                                            float(t_std))

            r_ratio = ratio / NoneCompressor.first_ratio
            if stages > 1 or stages == 0:
                if stages == 0:
                    loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
                else:
                    i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                    loop = stages - 1
                i = loop
                while i > 0:
                    one_indexes = abs_norm_tensor > threshold
                    indexes = one_indexes.nonzero().data.squeeze().view(-1)
                    abs_norm_tensor = abs_norm_tensor.data[indexes]

                    t_min = abs_norm_tensor.min()
                    abs_norm_tensor_min = abs_norm_tensor - t_min

                    if torch.__version__ < '1.3.0':
                        t_std = torch.std(abs_norm_tensor_min)
                        t_mean = torch.mean(abs_norm_tensor_min)
                    else:
                        t_std, t_mean = torch.std_mean(abs_norm_tensor_min)

                    _, threshold = utils.gen_threshold_from_normal_distribution(1 - i_ratio, float(t_mean),
                                                                                float(t_std))
                    if i == 1 and stages == 0:
                        _, threshold = utils.gen_threshold_from_normal_distribution(
                            1 - r_ratio / math.pow(i_ratio, loop - 1), float(t_mean),
                            float(t_std))
                    threshold += t_min

                    i -= 1

            one_indexes = abs_norm_tensor_cpy > threshold
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            values = tensor.data[indexes]

            if ada_stages:
                actual_ratio = (1.0 * values.numel() / numel)
                GaussianCompressorEC.adapt_stages(actual_ratio, ratio, ada_stages)

            return tensor, indexes, values


class TopKCompressorEC(SparseCompressor):
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'topkec'

    @staticmethod
    def clear():
        TopKCompressorEC.residuals = {}
        TopKCompressorEC.norm = 1.0
        TopKCompressorEC.sparsities = []
        TopKCompressorEC.zero_conditions = {}
        TopKCompressorEC.values = {}
        TopKCompressorEC.indexes = {}

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressorEC.residuals:
                TopKCompressorEC.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            tensor.data.add_(TopKCompressorEC.residuals[name].data)
            t_norm = tensor.data.norm(2)
            TopKCompressorEC.norm = t_norm

            tensor_abs_norm = tensor.abs() / t_norm
            values, indexes = torch.topk(tensor_abs_norm, k=k)
            values = tensor.data[indexes]

            TopKCompressorEC.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * TopKCompressorEC.residuals[name].data
            TopKCompressorEC.residuals[name].data[indexes] = 0.0


            return tensor, indexes, values

class TopKCompressor(TopKCompressorEC):
    name = 'topk'

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        start = time.time()
        with torch.no_grad():
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            t_norm = tensor.norm(2)
            TopKCompressorEC.norm = t_norm

            tensor_abs_norm = tensor.abs() / tensor.norm(2)

            values, indexes = torch.topk(tensor_abs_norm, k=k)

            values = tensor.data[indexes]

            return tensor, indexes, values

class GaussianKSGDCompressorEC(SparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'gaussianksgdec'

    @staticmethod
    def clear():
        GaussianKSGDCompressorEC.residuals = {}
        GaussianKSGDCompressorEC.sparsities = []
        GaussianKSGDCompressorEC.zero_conditions = {}
        GaussianKSGDCompressorEC.values = {}
        GaussianKSGDCompressorEC.indexes = {}
        GaussianKSGDCompressorEC.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in GaussianKSGDCompressorEC.residuals:
                GaussianKSGDCompressorEC.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianKSGDCompressorEC.residuals[name].data)
            t_norm = tensor.norm(2)
            GaussianKSGDCompressorEC.norm = t_norm

            norm_tensor = tensor / t_norm
            abs_norm_tensor = norm_tensor.abs()

            if torch.__version__ < '1.3.0':
                t_std = torch.std(abs_norm_tensor)
                t_mean = torch.mean(abs_norm_tensor)
            else:
                t_std, t_mean = torch.std_mean(abs_norm_tensor)

            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1 - ratio, float(t_mean), float(t_std))

            loops = 0
            while loops < 3:
                one_indexes = abs_norm_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2 * k / 3:
                    right_thres *= 0.5
                elif indexes.numel() > 4 * k / 3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            values = tensor.data[indexes]

            GaussianKSGDCompressorEC.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * GaussianKSGDCompressorEC.residuals[name].data
            GaussianKSGDCompressorEC.residuals[name].data[indexes] = 0.0

            return tensor, indexes, values

class GaussianKSGDCompressor(GaussianKSGDCompressorEC):
    name = 'gaussianksgd'

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            t_norm = tensor.norm(2)
            GaussianKSGDCompressorEC.norm = t_norm

            norm_tensor = tensor / t_norm
            abs_norm_tensor = norm_tensor.abs()
            abs_norm_tensor = tensor  # TODO:REMOVE

            if torch.__version__ < '1.3.0':
                t_std = torch.std(abs_norm_tensor)
                t_mean = torch.mean(abs_norm_tensor)
            else:
                t_std, t_mean = torch.std_mean(abs_norm_tensor)

            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1 - ratio, float(t_mean), float(t_std))

            loops = 0
            while loops < 5:
                one_indexes = abs_norm_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2 * k / 3:
                    right_thres *= 0.5
                elif indexes.numel() > 4 * k / 3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1

            values = tensor.data[indexes]

            return tensor, indexes, values


class RandomKCompressor(SparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    counter = 0
    name = 'randomk'

    @staticmethod
    def clear():
        RandomKCompressor.residuals = {}
        RandomKCompressor.sparsities = []
        RandomKCompressor.zero_conditions = {}
        RandomKCompressor.values = {}
        RandomKCompressor.indexes = {}
        RandomKCompressor.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            tensor.add_(RandomKCompressor.residuals[name].data)
            RandomKCompressor.norm = tensor.norm(2)

            perm = torch.randperm(numel, device=tensor.device)
            RandomKCompressor.counter += 1
            indexes = perm[:k]
            values = tensor.data[indexes]

            RandomKCompressor.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * RandomKCompressor.residuals[name].data
            RandomKCompressor.residuals[name].data[indexes] = 0.0

            return tensor, indexes, values

class RandomKECCompressor(RandomKCompressor):
    name = 'randomkec'

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RandomKCompressor.residuals[name].data)
            RandomKCompressor.norm = tensor.norm(2)

            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]
            values = tensor.data[indexes]

            RandomKCompressor.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * RandomKCompressor.residuals[name].data
            RandomKCompressor.residuals[name].data[indexes] = 0.0

            return tensor, indexes, values


class DGCSamplingCompressor(SparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'dgcsampling'

    @staticmethod
    def clear():
        DGCSamplingCompressor.residuals = {}
        DGCSamplingCompressor.sparsities = []
        DGCSamplingCompressor.zero_conditions = {}
        DGCSamplingCompressor.values = {}
        DGCSamplingCompressor.indexes = {}
        DGCSamplingCompressor.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in DGCSamplingCompressor.residuals:
                DGCSamplingCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = math.ceil(numel * ratio)

            tensor.add_(DGCSamplingCompressor.residuals[name].data)
            t_norm = tensor.norm(2)
            DGCSamplingCompressor.norm = t_norm

            abs_norm_tensor = torch.abs(tensor) / t_norm

            perm = torch.randperm(numel, device=tensor.device)
            fk = math.ceil(numel * 0.01)
            sampled_indexes = perm[0:fk]
            sampled_values = abs_norm_tensor[sampled_indexes]

            tmpvalues, tmpindexes = torch.topk(sampled_values, k=math.ceil(fk * ratio))

            thres = tmpvalues[math.ceil(fk * ratio) - 1]
            bool_indexes = abs_norm_tensor > thres
            indexes = bool_indexes.nonzero().data.squeeze().view(-1)
            num_k = len(indexes)

            if num_k > 4 * k / 3:
                tmpvalues = abs_norm_tensor[indexes]
                values, tmpindexes = torch.topk(tmpvalues, k=k)
                indexes = indexes[tmpindexes]

            values = tensor.data[indexes]

            DGCSamplingCompressor.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * DGCSamplingCompressor.residuals[name].data
            DGCSamplingCompressor.residuals[name].data[indexes] = 0.0

            return tensor, indexes, values


class RedSyncCompressor(SparseCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'redsync'

    @staticmethod
    def clear():
        RedSyncCompressor.residuals = {}
        RedSyncCompressor.sparsities = []
        RedSyncCompressor.zero_conditions = {}
        RedSyncCompressor.values = {}
        RedSyncCompressor.indexes = {}
        RedSyncCompressor.norm = 1.0

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in RedSyncCompressor.residuals:
                RedSyncCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RedSyncCompressor.residuals[name].data)
            RedSyncCompressor.norm = tensor.norm(2)

            l = 0.0
            r = 1.0
            thres = 0.0
            eps = 0.2
            abs_tensor = torch.abs(tensor)
            mean_val = torch.mean(abs_tensor)
            max_val = torch.max(abs_tensor)

            while r - l > eps:
                tmp_ratio = l + (r - l) / 2
                thres = mean_val + tmp_ratio * (max_val - mean_val)
                one_indexes = abs_tensor > thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                nnz = indexes.numel()
                if nnz > k and 2 * k > nnz:
                    break
                elif nnz < k / 2:
                    r = tmp_ratio
                else:
                    l = tmp_ratio
            indexes = indexes
            values = tensor.data[indexes]
            RedSyncCompressor.residuals[name].data = ec_grad_w * tensor.data + ec_mem_w * RedSyncCompressor.residuals[name].data
            RedSyncCompressor.residuals[name].data[indexes] = 0.0

            return tensor, indexes, values


class RedSyncTrimCompressor(RedSyncCompressor):
    """
    """
    residuals = {}
    sparsities = []
    zero_conditions = {}
    values = {}
    indexes = {}
    c = 0
    t = 0.
    name = 'redsynctrim'

    @staticmethod
    def compress(tensor, name=None, ratio=0.05, tb=None, i_ratio=0.25, stages=1, ec_grad_w=1.0, ec_mem_w=0.0):
        with torch.no_grad():
            if name not in RedSyncCompressor.residuals:
                RedSyncCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RedSyncCompressor.residuals[name].data)
            RedSyncCompressor.norm = tensor.norm(2)

            abs_tensor = torch.abs(tensor)
            mean_val = torch.mean(abs_tensor)
            max_val = torch.max(abs_tensor)
            eps = 0.2
            tmp_ratio = 1 - eps

            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = abs_tensor > thres
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            nnz = indexes.numel()

            while nnz < k:
                thres = mean_val + tmp_ratio * (max_val - mean_val)
                one_indexes = abs_tensor > thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                nnz = indexes.numel()
                tmp_ratio = tmp_ratio - eps

            indexes = indexes
            values = tensor.data[indexes]
            RedSyncCompressor.residuals[name].data = tensor.data + 0.0
            RedSyncCompressor.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values

compressors = {
    'exp': ExpCompressor,
    'expec': ExpCompressorEC,

    'gpareto': GParetoCompressor,
    'gparetoec': GParetoCompressorEC,

    'gammagpareto': GammaGParetoCompressor,
    'gammagparetoec': GammaGParetoCompressorEC,

    'gaussian': GaussianCompressor,
    'gaussianec': GaussianCompressorEC,

    'topk': TopKCompressor,
    'topkec': TopKCompressorEC,

    'gaussianksgd': GaussianKSGDCompressor,
    'gaussianksgdec': GaussianKSGDCompressorEC,

    'randomk': RandomKCompressor,
    'randomkec': RandomKECCompressor,

    'dgcsampling': DGCSamplingCompressor,

    'redsync': RedSyncCompressor,
    'redsynctrim': RedSyncTrimCompressor,

    'none': NoneCompressor,
    None: NoneCompressor
}


def test_gaussion_thres():
    set_mean = 0.0
    set_std = 0.5
    d = np.random.normal(set_mean, set_std, 10000)
    k2, p = stats.normaltest(d)
    print(p)
    nnz = np.count_nonzero(d)
    mean = np.mean(d)
    std = np.std(d)
    print('size:%d, nnz: %d' % (d.size, nnz))
    print(set_mean, set_std)
    print(mean, std)
    thres = 3 * std
    d[np.abs(d) < thres] = 0
    pvalue = 1 - np.count_nonzero(d) * 1.0 / d.size
    print('size:%d, p-value: %f' % (d.size, pvalue))
    left_thres, right_thres = utils.gen_threshold_from_normal_distribution(pvalue, mean, std)
    print('real thres:%f, gen thres: %f' % (thres, right_thres))


def test_gamma_thres(ratio=0.05):
    shape = 1.5
    scale = 0.01

    d = np.random.gamma(shape, scale, 10000)

    abs_tensor = np.abs(d)
    t_mean = np.mean(abs_tensor)
    s = math.log(t_mean) - np.mean(np.log(abs_tensor))

    alpha = (3 - s + math.sqrt(math.pow(s - 3, 2) + 24 * s)) / (12 * s)
    beta = t_mean / alpha

    threshold = float(beta * special.gammaincinv(alpha, 1 - ratio))
    print(alpha, beta, threshold)

    d[np.abs(d) < threshold] = 0
    pvalue = np.count_nonzero(d) * 1.0 / d.size
    print('size:%d, p-value: %f' % (d.size, pvalue))


if __name__ == '__main__':
    test_gamma_thres(ratio=0.1)
