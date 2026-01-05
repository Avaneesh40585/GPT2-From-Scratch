import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculates the Cross Entropy Loss for a single batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    logits = model(input_batch)
    
    # Shape Manipulation for CrossEntropyLoss (flattening):
    # Logits:  [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
    # Targets: [batch, seq_len]             -> [batch * seq_len]
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Estimates the average loss over a specific number of batches from a dataloader.
    Useful for checking validation loss without running the whole set.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
        
    # If num_batches is not specified, run the whole loader
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches: break
        
        with torch.no_grad(): # Disable gradient tracking for efficiency
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
        total_loss += loss.item()
        
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Computes both training and validation loss for monitoring.
    """
    model.eval() # Disable dropout
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train() # Re-enable dropout
    return train_loss, val_loss


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plots training and validation loss curves inline.
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot training and validation loss
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for alignment
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()
    
def print_model_params(model):
    """
    Prints the total and trainable parameters of the model, broken down by component (Embedding, Transformer, Head).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Breakdown by component type
    print("\n--- Breakdown ---")
    param_counts = {"embedding": 0, "transformer": 0, "head": 0}

    for name, param in model.named_parameters():
        if "tok_emb" in name or "pos_emb" in name: # Token & Positional embeddings
            param_counts["embedding"] += param.numel()
        elif "out_head" in name: # Output head
            param_counts["head"] += param.numel()
        else: # Transformer blocks and LayerNorms
            param_counts["transformer"] += param.numel()

    for key, count in param_counts.items():
        if count > 0:
            print(f"{key.capitalize()}: {count:,} ({count/total_params:.1%})")