import os
import json
import torch
# import numpy as np
from torch.utils.data import Dataset
import subprocess
from rich.console import Console

console = Console()

class ARCDataset(Dataset):
    def __init__(self, data_dir="data/arc", split="training"):
        self.data_dir = data_dir
        self.split = split
        self.ensure_dataset()
        self.data = self.load_data_aligned()
        
    def ensure_dataset(self):
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if extracted folder exists
        if not os.path.exists(os.path.join(self.data_dir, "ARC-AGI-master")) and \
           not os.path.exists(os.path.join(self.data_dir, "ARC-AGI-main")):
            
            console.print(f"[bold blue]Downloading ARC-AGI repository...[/]")
            zip_path = os.path.join(self.data_dir, "arc.zip")
            # Try master first
            url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
            
            try:
                subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)
                subprocess.run(["unzip", "-q", "-o", zip_path, "-d", self.data_dir], check=True)
                os.remove(zip_path)
                console.print("[green]Download and extraction complete.[/]")
            except Exception as e:
                console.print(f"[red]Failed to download data: {e}[/]")

    def load_data_aligned(self):
        # Load from individual files in extracted folder
        # Folder structure: data/arc/ARC-AGI-master/data/{training,evaluation}
        split_dir = "training" if self.split == "training" else "evaluation"
        
        # Check for master or main folder
        tasks_dir = os.path.join(self.data_dir, "ARC-AGI-master", "data", split_dir)
        if not os.path.exists(tasks_dir):
            tasks_dir = os.path.join(self.data_dir, "ARC-AGI-main", "data", split_dir)
            
        if not os.path.exists(tasks_dir):
             console.print(f"[red]Could not find tasks directory at {tasks_dir}[/]")
             return []
            
        data = []
        # Sort to ensure deterministic order
        for filename in sorted(os.listdir(tasks_dir)):
            if filename.endswith(".json"):
                path = os.path.join(tasks_dir, filename)
                try:
                    with open(path, 'r') as f:
                        task = json.load(f)
                    
                    # Each task file has "train" and "test"
                    # "test" has input and output (in training set)
                    if 'test' in task and len(task['test']) > 0:
                        test_pair = task['test'][0]
                        if 'output' in test_pair:
                            data.append({
                                'train': task['train'],
                                'test_input': test_pair['input'],
                                'test_output': test_pair['output']
                            })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        task = self.data[idx]
        
        def grid2tensor(g):
            # Use torch directly to avoid numpy dependency
            a = torch.tensor(g, dtype=torch.long)
            # Pad to 30x30 (standard ARC size)
            p = torch.zeros((30,30), dtype=torch.long)
            h, w = a.shape
            p[:h, :w] = a
            return p

        demos_in  = [grid2tensor(ex["input"])  for ex in task["train"]]
        demos_out = [grid2tensor(ex["output"]) for ex in task["train"]]
        test_in   = grid2tensor(task["test_input"])
        test_out  = grid2tensor(task["test_output"])

        return demos_in, demos_out, test_in, test_out

def collate_arc(batch):
    return batch
