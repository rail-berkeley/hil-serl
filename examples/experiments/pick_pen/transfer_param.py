import torch
import pickle

# Load PyTorch weights
weights = torch.load("/home/amigo/.serl/resnet34_params.pth", weights_only=False)

# Save as .pkl
with open("resnet34_params.pkl", "wb") as f:
    pickle.dump(weights, f)
