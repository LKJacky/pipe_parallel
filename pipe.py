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

DEBUG = os.environ.get('DEBUG', 'False') == 'true'

if DEBUG:
    llama_65B_config.hidden_size = 128
    llama_65B_config.intermediate_size = 128
    GPUS = 4
    GPU_PER_NODE = 4
else:
    GPUS = 8
    GPU_PER_NODE = 4


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


# distributed llama modules run on a gpu


class DistributedLLamaLayer(DistributedModule):
    def __init__(self, config, num_layers, device, *args, **kwargs) -> None:
        super().__init__(device, *args, **kwargs)
        self.module = nn.Sequential(
            *[LLaMaDecoderLayerWrapper(config)
              for _ in range(num_layers)]).to(self.device)

    def forward(self, x_rref: RRef, session: int = 0):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            for layer in self.module:
                x = layer(x, session=session)
        return x.cpu()

    def reset_past_kv(self, session=None):
        for module in self.module:
            module: LLaMaDecoderLayerWrapper
            module.reset_past_kv(session=session)


class DistributedLLamaHead(DistributedModule):
    def __init__(self, config: LLaMAConfig, device, *args, **kwargs) -> None:
        super().__init__(device, *args, **kwargs)
        config.num_hidden_layers = 0
        llama = LLamaWrapper(config)
        self.module = llama.embed_tokens.to(self.device)


class DistributedLLamaTail(DistributedModule):
    def __init__(self, config: LLaMAConfig, device, *args, **kwargs) -> None:
        super().__init__(device, *args, **kwargs)
        config.num_hidden_layers = 0
        llama = LLamaWrapper(config)
        self.norm = llama.norm.to(self.device)

    def forward(self, x_rref: RRef):
        x: torch.Tensor = x_rref.to_here().to(self.device)
        with self._lock:
            x = self.norm(x)
            x = x[:, -1, :]
            x = x.argmax(dim=-1)
        return x.cpu()


# distributed llama model


class DistributedLLaMa(nn.Module):
    def __init__(self,
                 config: LLaMAConfig,
                 gpus=8,
                 gpu_per_node=4,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.embed_tokens = rpc.remote('worker_0',
                                       DistributedLLamaHead,
                                       args=(config, 'cuda:0'))
        self.norm = rpc.remote('worker_0',
                               DistributedLLamaTail,
                               args=(config, 'cuda:0'))

        layer_nums = [10] * gpus
        cuda_num = list(range(gpu_per_node)) * (gpus // gpu_per_node)

        self.layers = []

        for i, (num, cuda_i) in enumerate(zip(layer_nums, cuda_num), start=0):
            self.layers.append(
                rpc.remote(f'worker_{i}',
                           DistributedLLamaLayer,
                           args=(config, num, torch.device(f'cuda:{cuda_i}'))))

    def forward(self, x, session: int = 0):
        x_rref = RRef(x.cpu())

        x_rref = self.embed_tokens.remote().forward(x_rref)
        for layer in self.layers:
            x_rref = layer.remote().forward(x_rref, session)
        x_rref = self.norm.remote().forward(x_rref)
        return x_rref.to_here().to(x.device)

    def reset_kv(self, session=None):
        for layer in self.layers:
            layer: DistributedLLamaLayer
            layer.remote().reset_past_kv(session=session)


def master(gpus=4, gpu_per_node=4, arg=None):

    with torch.no_grad():
        llama = DistributedLLaMa(llama_65B_config,
                                 gpus=gpus,
                                 gpu_per_node=gpu_per_node).cuda().half()

        # wram up
        x = torch.rand([arg.b, arg.s]).cuda().long()
        with autocast():
            _ = llama(x)
            llama.reset_kv()

        # test encoding
        x = torch.rand([arg.b, arg.s]).cuda().long()
        t0 = time.time()
        for i in range(10):
            torch.cuda.synchronize()
            with autocast():
                _ = llama(x)
                llama.reset_kv()
            torch.cuda.synchronize()
        print(f'total use {(time.time()-t0)/10:.2f}s with {x.shape}')

        # test per token
        x = torch.rand([arg.b, 1]).cuda().long()
        t0 = time.time()
        torch.cuda.synchronize()
        for i in range(arg.s):
            with autocast():
                _ = llama(x)
            if i % 100 == 0:
                torch.cuda.synchronize()
                print(
                    f'total use average {(time.time()-t0)/(i+1):.2f}s with {i+1} tokens'  # noqa
                )


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
    world_size = GPUS
    n_per_node = GPU_PER_NODE

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('machine', type=int, default=0)
    parser.add_argument('-s', type=int, help='len of sequence', default=2048)
    parser.add_argument('-b', type=int, help='batch of sequence', default=1)
    arg = parser.parse_args()
    machine = arg.machine

    mp.spawn(run_workers,
             args=(machine, world_size, n_per_node, arg),
             nprocs=n_per_node if machine != 0 else n_per_node + 1,
             join=True)
