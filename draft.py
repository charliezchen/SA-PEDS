import torch

# Enable the profiler
with torch.profiler.profile(record_shapes=True) as prof:
    # Here goes your code
    # For example
    x = torch.randn((10, 10))
    y = torch.randn((10, 10))
    z = x @ y

# Print profiler results
print(prof.key_averages().table(sort_by="cpu_time_total"))

import pickle

with open('experiment_0620_2324.pkl', 'rb') as f:
    data = pickle.load(f)

from IPython import embed
embed() or exit(0)