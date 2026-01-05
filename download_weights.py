import os
import json
import requests
import numpy as np
import torch
import tensorflow as tf
from tqdm import tqdm

def download_file(url, destination):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    file_size = int(response.headers.get("Content-Length", 0))

    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size and file_size == file_size_local:
            print(f"File already exists: {destination}")
            return

    block_size = 1024 
    progress_bar_description = os.path.basename(url)
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as bar:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Reads the TensorFlow checkpoint and organizes it into a dictionary hierarchy."""
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    
    # Iterate over all variables in the TF checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:] # Skip "model/" prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array
    return params

def assign(left, right):
    """Helper to assign a NumPy array (right) to a PyTorch Parameter (left)."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    """Maps the loaded TensorFlow weights into the PyTorch GPTModel architecture."""
    
    # 1. Embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # 2. Transformer Blocks
    for b in range(len(params["blocks"])):
        block_params = params["blocks"][b]
        
        # Attention: Q, K, V
        q_w, k_w, v_w = np.split(block_params["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(block_params["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # Attention: Output Projection
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, block_params["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, block_params["attn"]["c_proj"]["b"])

        # FeedForward
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, block_params["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, block_params["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, block_params["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, block_params["mlp"]["c_proj"]["b"])

        # Layer Norms
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, block_params["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, block_params["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, block_params["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, block_params["ln_2"]["b"])

    # 3. Final Norm & Head
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def download_and_save_gpt2(model_size, target_dir, model_class):
    """
    Main entry point: Downloads files, converts weights, saves as .pth
    """
    # 1. Setup Paths
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    save_path = os.path.join(target_dir, f"gpt2_{model_size}.pth")
    
    # If .pth already exists, skip everything!
    if os.path.exists(save_path):
        print(f"Model already exists at: {save_path}")
        return save_path

    model_dir = os.path.join(target_dir, "tf_weights", model_size)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Download TF Files
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = ["checkpoint", "encoder.json", "hparams.json",
                 "model.ckpt.data-00000-of-00001", "model.ckpt.index",
                 "model.ckpt.meta", "vocab.bpe"]

    print(f"Downloading {model_size} files...")
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # 3. Load & Convert
    print("Loading TensorFlow weights...")
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    # 4. Initialize PyTorch Model to verify shape match
    # We must match the config of the downloaded model
    config = {
        "vocab_size": 50257,
        "context_length": settings["n_ctx"],
        "emb_dim": settings["n_embd"],
        "n_heads": settings["n_head"],
        "n_layers": settings["n_layer"],
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    
    print("Converting to PyTorch...")
    temp_model = model_class(config)
    load_weights_into_gpt(temp_model, params)

    # 5. Save Final .pth
    print(f"Saving PyTorch model to {save_path}...")
    torch.save(temp_model.state_dict(), save_path)
    print("Done!")
    
    return save_path