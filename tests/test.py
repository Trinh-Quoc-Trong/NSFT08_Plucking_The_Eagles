
import torch 

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available")
        return torch.device('cuda')

print(check_gpu())

















