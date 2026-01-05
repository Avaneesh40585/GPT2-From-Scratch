import torch

def text_to_token_ids(text, tokenizer, device):
    """
    Converts a string into a tensor of token IDs with a batch dimension.
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor.to(device)

def token_ids_to_text(token_ids, tokenizer):
    """
    Converts a tensor of token IDs back into a string.
    """
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist()) # tensor back to list
    
def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generates text with advanced sampling (Top-K & Temperature).
    By default, uses greedy decoding (temp=0.0).
    """
    for _ in range(max_new_tokens):
        # 1. Crop Context
        idx_cond = idx[:, -context_size:]
        
        # 2. Get Predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on last time step
        logits = logits[:, -1, :]

        # 3. Top-K Filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 4. Temperature Scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Numerical stability (optional but good practice)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
        else:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def generate_and_print(model, tokenizer, device, start_context, max_new_tokens=50, temperature=0.0, top_k=None):
    """
    Generic wrapper to generate and print a sample.
    
    Key Features:
    - Auto-detects model state: Restores .train() mode only if it was active before.
    - Configurable length: Allows changing max_new_tokens.
    """
    # 1. Capture original state (Train or Eval?)
    was_training = model.training 
    model.eval() # Always switch to eval for generation
    
    # 2. Get context size from model config
    context_size = model.pos_emb.weight.shape[0]
    
    # 3. Encode
    encoded = text_to_token_ids(start_context, tokenizer, device)
    
    # 4. Generate
    with torch.no_grad():
        token_ids = generate_text(
            model=model, 
            idx=encoded, 
            max_new_tokens=max_new_tokens, 
            context_size=context_size,
            temperature=temperature,
            top_k=top_k
        )
        
    # 5. Decode & Print
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print_text = decoded_text.replace('\n', ' ')
    print(f"\n[Gen]: {print_text}\n")
    
    # 6. Restore Original State
    # If we were training, go back to training. If we were inferencing, stay in eval.
    if was_training:
        model.train()

    return decoded_text