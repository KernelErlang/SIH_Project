import torch

# 1. Define the path
model_path = 'models/best_plate_crnn.pt'

# 2. Load the file
print(f"Loading {model_path}...\n")
model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# 3. Print the top-level contents
print("This file is a checkpoint containing the following top-level keys:")
for key in model_data.keys():
    print(f"- {key}")
    
    # If the key contains the actual model weights, let's peek inside!
    if isinstance(model_data[key], dict) or str(type(model_data[key])) == "<class 'collections.OrderedDict'>":
         print(f"  (This key contains {len(model_data[key])} items inside it)")