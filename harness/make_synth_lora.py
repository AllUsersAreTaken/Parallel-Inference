"""Generate synthetic PEFT LoRA adapters for Qwen2.5-3B-Instruct (latency probe).

B=0 variants are numerically exact no-ops (A random, B zero) but still execute
both LoRA matmuls, so they isolate the compute cost without changing outputs.
"""
import json
import os

import torch
from safetensors.torch import save_file

D = r"S:\Agent\QSExplore\RecycleLM\data\_synth"
CFG = os.path.join(D, "qwen25_config")
os.makedirs(CFG, exist_ok=True)

config = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": True,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "torch_dtype": "bfloat16",
    "hidden_act": "silu",
}
with open(os.path.join(CFG, "config.json"), "w") as f:
    json.dump(config, f, indent=1)

H, FFN, NL = 2048, 11008, 36
PROJ = {  # name -> (in, out)
    "q_proj": (H, H), "k_proj": (H, 256), "v_proj": (H, 256), "o_proj": (H, H),
    "gate_proj": (H, FFN), "up_proj": (H, FFN), "down_proj": (FFN, H),
}


def make(name, rank, zero, layers, projs):
    dirn = os.path.join(D, name)
    os.makedirs(dirn, exist_ok=True)
    tensors = {}
    for i in layers:
        for p in projs:
            fin, fout = PROJ[p]
            a = torch.randn(rank, fin) * 0.02
            b = torch.zeros(fout, rank) if zero else torch.randn(fout, rank) * 0.02
            base = "base_model.model.model.layers.%d" % i
            if p in ("gate_proj", "up_proj", "down_proj"):
                base += ".mlp"
            else:
                base += ".self_attn"
            tensors["%s.%s.lora_A.weight" % (base, p)] = a
            tensors["%s.%s.lora_B.weight" % (base, p)] = b
    save_file(tensors, os.path.join(dirn, "adapter_model.safetensors"))
    with open(os.path.join(dirn, "adapter_config.json"), "w") as f:
        json.dump({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank,
            "lora_alpha": 2 * rank, "lora_dropout": 0.0, "bias": "none",
            "target_modules": projs, "inference_mode": True,
            "base_model_name_or_path": CFG,
            "fan_in_fan_out": False,
        }, f, indent=1)
    print("%-24s rank=%d zero=%s layers=%s tensors=%d size=%.1f MB"
          % (name, rank, zero, ("all" if len(layers) > 1 else layers[0]),
             len(tensors), sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6))


all_layers = list(range(NL))
make("lora_full_r16_zero", 16, True, all_layers, list(PROJ))
make("lora_full_r16_nz", 16, False, all_layers, list(PROJ))
make("lora_lastblock_r16_zero", 16, True, [NL - 1], ["o_proj", "gate_proj", "up_proj", "down_proj"])
make("lora_full_r64_zero", 64, True, all_layers, list(PROJ))
print("done ->", D)
