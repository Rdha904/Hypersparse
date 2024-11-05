import time
import torchvision.models as models
import torch
try:
    from apex.contrib.sparsity import ASP
except ImportError:
    raise RuntimeError("Failed to import ASP. Please install Apex from https:// github.com/nvidia/apex .")

resnet18 = models.resnet18()
optimizer_sparse = torch.optim.AdamW(resnet18.parameters(), lr=0.0001, weight_decay=0.05)

ASP.prune_trained_model(resnet18,optimizer_sparse)
print("Start training")
start_time = time.time()