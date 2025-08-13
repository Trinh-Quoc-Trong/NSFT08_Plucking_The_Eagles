import os 
import torch
import torchvision.model as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import matplotlib.pyplot as plt 

def get_device():
    '''kiem tra va tra ve thiet bi co san (GPU hoac CPU)'''
    if torch.cuda.is_available():
        print(">>> Đang sử dụng GPU.")
        return torch.device("cuda")
    












