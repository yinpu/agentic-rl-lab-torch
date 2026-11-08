"""真实 Qwen3.5-4B 的概率、反向与 adapter 保存检查；优势为数值测试输入。"""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from lightning.fabric import Fabric
from peft import LoraConfig, get_peft_model, PeftModel
from experiments.gspo.task import load_model, prompt_ids, batch_completion_logprobs, write_json
from experiments.gspo.train import cache_old, update_policy

def main():
    output = Path("experiments/gspo/runs/model-check")
    fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(42)
    base, tokenizer = load_model(".cache/models/Qwen3.5-4B")
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4,
                                          target_modules="all-linear", lora_dropout=0.))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
    model, opt = fabric.setup(model, opt)
    prompt = prompt_ids(tokenizer, "Find the value of 2 + 3.")
    # 这里仅检查数值与训练接口，不把构造的答案、优势当作真实 rollout 实验。
    samples = [{"prompt": prompt, "tokens": tokenizer.encode(text, add_special_tokens=False),
                "advantage": advantage} for text, advantage in [(r"\boxed{5}", 1.), (r"\boxed{4}", -1.)]]
    with fabric.autocast():
        cache_old(model.module, samples, 1)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    metrics = []
    frozen = [s['old'].clone() for s in samples]
    for _ in range(4):
        metrics.append(update_policy(model, opt, samples,
                       SimpleNamespace(micro_batch_size=1, max_grad_norm=1.), fabric))
    for s, old in zip(samples, frozen):
        torch.testing.assert_close(s['old'], old, atol=0, rtol=0)
    assert any(abs(m['seq_ratio'] - 1.) > 1e-6 for m in metrics[1:])
    assert any(not torch.equal(a,b) for a,b in zip(before, [p for p in model.parameters() if p.requires_grad]))
    model.module.eval()
    with torch.no_grad(), fabric.autocast():
        expected = batch_completion_logprobs(model.module, samples)
    model.module.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    del model, opt, base
    torch.cuda.empty_cache()
    base, _ = load_model(".cache/models/Qwen3.5-4B")
    loaded = PeftModel.from_pretrained(base, output / "adapter").to(fabric.device).eval()
    with torch.no_grad(), fabric.autocast():
        actual = batch_completion_logprobs(loaded, samples)
    for a,b in zip(actual, expected):
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    write_json(output / "check.json", {"numerical_update_and_reload": "passed", "metrics": metrics})
    print("Qwen3.5 forward/backward, parameter update, adapter reload: passed")

if __name__ == "__main__":
    main()
