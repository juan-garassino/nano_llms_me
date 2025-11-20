import torch
from torch.utils.data import DataLoader, Dataset
import os

try:
    from transformers import AutoTokenizer
    from datasets import load_dataset as hf_load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers or Datasets library not found. Text training will not work.")

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx):
        # Simple chunking
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        
        # Handle last chunk if needed, but __len__ avoids it
        if len(chunk) < self.seq_len + 1:
            # Pad or just return what we have? 
            # For simplicity, we just cut off the end in __len__
            pass
            
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_text_loaders(dataset_name: str, batch_size: int, seq_len: int, num_workers=0):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library required for text loaders.")
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if dataset_name.lower() in ["wikitext", "wikitext-103"]:
        dataset = hf_load_dataset("wikitext", "wikitext-103-raw-v1")
        text_col = "text"
    elif dataset_name.lower() in ["shakespeare", "tinyshakespeare", "tiny_shakespeare"]:
        try:
            dataset = hf_load_dataset("tiny_shakespeare")
        except Exception:
            print("Could not load tiny_shakespeare from HF, falling back to wikitext-2")
            dataset = hf_load_dataset("wikitext", "wikitext-2-raw-v1")
        text_col = "text"
    else:
        # Default to wikitext-2
        dataset = hf_load_dataset("wikitext", "wikitext-2-raw-v1")
        text_col = "text"
        
    def tokenize_function(examples):
        return tokenizer(examples[text_col])
    
    # Tokenize
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=[text_col])
    
    # Concatenate all tokens
    # This is a simplified approach: just concat everything and chunk
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        return concatenated_examples

    # We'll just take the train split and flatten it for simplicity in this demo
    train_data = tokenized_datasets["train"]
    val_data = tokenized_datasets["validation"]
    
    # Flatten (inefficient for huge datasets but fine for demo)
    train_tokens = torch.tensor([item for sublist in train_data['input_ids'] for item in sublist], dtype=torch.long)
    val_tokens = torch.tensor([item for sublist in val_data['input_ids'] for item in sublist], dtype=torch.long)
    
    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    
    loaders = {
        dataset_name: {
            "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "test_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
    }
    
    return loaders, tokenizer.vocab_size
