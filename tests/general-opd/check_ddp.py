"""torchrun --standalone --nproc_per_node=2 tests/general-opd/check_ddp.py"""
from test_opd import tiny, samples, task
from types import SimpleNamespace
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

fabric = Fabric(accelerator="cpu", devices=2, strategy=DDPStrategy(broadcast_buffers=False))
fabric.launch()
args = SimpleNamespace(micro_batch_size=1, beta=1., max_grad_norm=100.)
actual, expected = tiny(), tiny()
rows = samples(actual)
opt = torch.optim.SGD(actual.parameters(), lr=.1)
reference = torch.optim.SGD(expected.parameters(), lr=.1)
count = sum(len(r['tokens']) for r in rows)
loss = sum(task.opd_loss(logp, r['old'], r['teacher']).sum()
           for r, logp in zip(rows, task.batch_completion_logprobs(expected, rows))) / count
loss.backward()
reference.step()
wrapped, opt = fabric.setup(actual, opt)
task.update_policy(wrapped, opt, rows[fabric.global_rank::2], args, fabric)
for a, b in zip(actual.parameters(), expected.parameters()):
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
fabric.print('Unequal-token CPU DDP matches centralized update')
torch.distributed.destroy_process_group()
