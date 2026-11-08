"""torchrun --standalone --nproc_per_node=2 tests/gspo/distributed_check.py"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test_gspo import make_model, samples
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from experiments.gspo.train import cache_old, update_policy
from experiments.gspo.task import batch_completion_logprobs, sequence_objective

f = Fabric(accelerator='cpu', devices=int(os.environ['WORLD_SIZE']),
           strategy=DDPStrategy(broadcast_buffers=False))
f.launch()
reference = make_model()
data = samples()
cache_old(reference,data,2)
opt_ref = torch.optim.SGD(reference.parameters(),lr=.0001)
model = make_model()
opt = torch.optim.SGD(model.parameters(),lr=.0001)
model,opt = f.setup(model,opt)
# rank 1 只持有零优势样本，仍需参与反向与同步。
for _ in range(4):
    opt_ref.zero_grad()
    loss = sum(sequence_objective(p,s['old'],s['advantage'])[0]
               for p,s in zip(batch_completion_logprobs(reference,data),data))/len(data)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
    opt_ref.step()
    update_policy(model,opt,data[f.global_rank*2:(f.global_rank+1)*2],
                  SimpleNamespace(micro_batch_size=1,max_grad_norm=1.),f)
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)
zero = [dict(s,advantage=0.) for s in data[:2]]
assert update_policy(model,opt,zero,SimpleNamespace(micro_batch_size=1,max_grad_norm=1.),f)['skipped']
if f.is_global_zero:
    print('GSPO DDP sequence normalization, accumulation and zero-advantage rank: passed')
