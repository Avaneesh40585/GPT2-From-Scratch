import os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

def split_train_val_test(file_path, train_ratio=0.85, val_ratio=0.10):
    """
    Reads a file, splits it, and SAVES the splits to disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    base_dir = os.path.dirname(file_path)
    
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Slice
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Save
    print(f"Saving split files to {base_dir}...")
    with open(os.path.join(base_dir, 'train.txt'), 'w', encoding='utf-8') as f: f.write(train_data)
    with open(os.path.join(base_dir, 'valid.txt'), 'w', encoding='utf-8') as f: f.write(val_data)
    with open(os.path.join(base_dir, 'test.txt'), 'w', encoding='utf-8') as f: f.write(test_data)

    print(f"Split completed.")

class GPTDataset(Dataset):
    """
    A PyTorch Dataset for GPT-style training.
    It chunks a large text into input-target pairs using a sliding window.
    """

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 1. Tokenize the entire text once
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        total_tokens = len(token_ids)

        # 2. Use a sliding window to chunk the book
        # Input:  [0, 1, 2, 3]
        # Target: [1, 2, 3, 4]
        for i in range(0, total_tokens - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            
            # Convert to tensors (PyTorch infers the type automatically)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(
    text, 
    batch_size=4, 
    max_length=256, 
    stride=128, 
    shuffle=True, 
    drop_last=True, 
    num_workers=0
):
    """
    Creates a PyTorch DataLoader for the GPT Dataset.
    """
    
    # Initialize tokenizer (using GPT-2 encoding)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Initialize Custom Dataset
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    
    # Initialize DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )

    return dataloader