from lightningmodel import UITQATransformer
import torch

## Example
model = UITQATransformer.load_from_checkpoint("checkpoints/checkpoint.ckpt")
model.eval()
x = torch.randn(1, 64)

with torch.no_grad():
    y_hat = model(x)