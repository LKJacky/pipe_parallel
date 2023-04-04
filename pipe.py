import os
import threading
import time

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.distributed.rpc import RRef

from llama import (LLaMAConfig, LLaMaDecoderLayerWrapper, LLamaWrapper,
                   llama_65B_config)

DEBUG = True

# llama_65B_config.hidden_size = 128
# llama_65B_config.intermediate_size = 128


class DistributedModule(nn.Module):
    def __init__(self, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self.device = device
        self.module: nn.Module

    def forward(self, x_rref: RRef):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.module(x)
        return out.cpu()


class DistributedLLamaLayers(DistributedModule):
    def __init__(self, config, num_layers, device, *args, **kwargs) -> None:
        super().__init__(device, *args, **kwargs)
        self.module = nn.Sequential(
            *[LLaMaDecoderLayerWrapper(config)
              for _ in range(num_layers)]).to(self.device).half()

    def reset_past_kv(self):
        for module in self.module:
            module: LLaMaDecoderLayerWrapper
            module.reset_past_kv()


class DistributedLLama(nn.Module):
    def __init__(self, config: LLaMAConfig, layers=[8, 8, 8], cudas=[1, 2, 3]):
        print('layers', layers, 'cudas', cudas)
        super().__init__()
        self.padding_idx = config.pad_token_id

        config.num_hidden_layers = 1
        llama = LLamaWrapper(config)

        self.embed_tokens = llama.embed_tokens
        self.norm = llama.norm

        self.p_rref_s = []

        for i, (num, cuda_i) in enumerate(zip(layers, cudas), start=0):
            self.p_rref_s.append(
                rpc.remote(f'worker_{i}',
                           DistributedLLamaLayers,
                           args=(config, num, torch.device(f'cuda:{cuda_i}'))))

    def forward(self, x):
        # x = self.embed_tokens(x)
        x_rref = RRef(x.cpu())

        for p_rref in self.p_rref_s:
            x_rref = p_rref.remote().forward(x_rref)
        return x_rref.to_here()

    def reset_kv(self):
        for p_rref in self.p_rref_s:
            p_rref.remote().reset_past_kv()


def master(gpus=4, gpu_per_node=4, arg=None):

    with torch.no_grad():
        layers = DistributedLLama(llama_65B_config,
                                  layers=[10] * gpus,
                                  cudas=list(range(gpu_per_node)) *
                                  (gpus // gpu_per_node))
        llama_65B_config.num_hidden_layers = 0
        llama = LLamaWrapper(llama_65B_config).cuda().half()

        # wram up
        x = torch.rand([arg.b, arg.s]).cuda().long()
        with autocast():
            x = llama.embed_tokens(x)
            y = layers(x).cuda()
            y = llama.norm(y)
            layers.reset_kv()

        # test encoding
        x = torch.rand([arg.b, arg.s]).cuda().long()
        t0 = time.time()
        for i in range(10):
            torch.cuda.synchronize()
            with autocast():
                y = llama.embed_tokens(x)
                y = layers(y).cuda()
                y = llama.norm(y)
                layers.reset_kv()
            torch.cuda.synchronize()
        print(f'total use {(time.time()-t0)/10:.2f}s with {x.shape}')

        # test per token
        x = torch.rand([1, 1]).cuda().long()
        t0 = time.time()
        torch.cuda.synchronize()
        for i in range(arg.s):
            with autocast():
                y: torch.Tensor = llama.embed_tokens(x)
                y = layers(y).cuda()
                y = llama.norm(y)
            if i % 100 == 0:
                print(
                    f'total use average {(time.time()-t0)/(i+1):.2f}s with {i+1} tokens'  # noqa
                )
        torch.cuda.synchronize()
        layers.reset_kv()


def run_workers(rank, machine, world_size, n_per_node=1, arg=None):

    os.environ['MASTER_ADDR'] = '10.140.54.75'
    os.environ['MASTER_PORT'] = '29513'

    if machine == 0:
        local_rank = rank - 1
    else:
        local_rank = rank

    parallel_rank = machine * n_per_node + local_rank
    rank = parallel_rank + 1

    print(
        f'rank: {rank}, local_rank: {local_rank}, parrallel_rank: {parallel_rank}, world_size: {world_size}'  # noqa
    )

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
                                              rpc_timeout=300)

    if machine == 0 and rank == 0:
        print('init master')
        rpc.init_rpc('master',
                     rank=rank,
                     world_size=world_size + 1,
                     rpc_backend_options=options)
        master(world_size, n_per_node, arg)
    else:
        rpc.init_rpc(f'worker_{parallel_rank}',
                     rank=rank,
                     world_size=world_size + 1,
                     rpc_backend_options=options)
        print(f'inited worker_{parallel_rank}')
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == '__main__':
    world_size = 8
    n_per_node = 4

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('machine', type=int)
    parser.add_argument('-s', type=int, help='len of sequence', default=2048)
    parser.add_argument('-b', type=int, help='batch of sequence', default=1)
    arg = parser.parse_args()
    machine = arg.machine

    mp.spawn(run_workers,
             args=(machine, world_size, n_per_node, arg),
             nprocs=n_per_node if machine != 0 else n_per_node + 1,
             join=True)
