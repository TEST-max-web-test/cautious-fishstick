import torch
import os

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Loaded checkpoint from {path}")
    else:
        print(f"No checkpoint found at {path}")

# Example usage:
# save_checkpoint(model, 'utils/checkpoint.pt')
# load_checkpoint(model, 'utils/checkpoint.pt')
