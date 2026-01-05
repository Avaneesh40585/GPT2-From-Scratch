import torch
import numpy as np

def assign(left, right):
    """
    Helper to ensure shapes match before assigning weights.
    Left: Custom Model Parameter (PyTorch)
    Right: Hugging Face Weight Tensor (PyTorch)
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())

def load_weights_into_gpt(gpt, gpt_hf):
    """
    Maps weights from a Hugging Face GPT2Model to your custom GPTModel.
    
    Args:
        gpt: Your custom GPTModel instance.
        gpt_hf: The loaded Hugging Face GPT2Model instance.
    """
    d = gpt_hf.state_dict()

    # 1. Embeddings
    # HF names: wpe (position), wte (token)
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, d["wte.weight"])
    
    # 2. Transformer Blocks
    # HF stores Q,K,V in a single matrix 'c_attn'. We must split them.
    for b in range(len(gpt.trf_blocks)):
        # --- Attention Weights ---
        # HF shape: [768, 2304] -> Split into [768, 768] x 3
        # We ensure tensors are on CPU for splitting to avoid potential device conflicts
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"].cpu(), 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight   = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        
        # --- Attention Biases ---
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"].cpu(), 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias   = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
        
        # --- Output Projection ---
        # Note: HF uses Conv1D, so we transpose (.T) weights to match Linear layers
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias   = assign(gpt.trf_blocks[b].att.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])
        
        # --- Feed Forward (MLP) ---
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias   = assign(gpt.trf_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias   = assign(gpt.trf_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])
        
        # --- Layer Norms ---
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])
        
    # 3. Final Norm & Head
    gpt.final_norm.scale = assign(gpt.final_norm.scale, d["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, d["ln_f.bias"])
    
    # Weight Tying: GPT-2 uses the same weights for the embedding and the output head
    gpt.out_head.weight = assign(gpt.out_head.weight, d["wte.weight"])
    
    print("-> Weights successfully transferred from HF to Custom Model.")