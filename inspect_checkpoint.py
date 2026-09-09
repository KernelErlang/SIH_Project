import torch
checkpoint = torch.load(r"CRNN MODEL\best_plate_crnn.pt", map_location="cpu")
print(checkpoint.keys())
print(checkpoint['characters']) 