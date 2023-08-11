import torch
import socket

print("host name: ",socket.gethostname())
print("Using GPU is " + str(torch.cuda.is_available()))
print("Using " + str(torch.cuda.device_count()) + " GPUs")
