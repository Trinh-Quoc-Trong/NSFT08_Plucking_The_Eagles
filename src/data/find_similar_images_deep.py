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
    print(">>> Đang sử dụng CPU.")
    return torch.device("cpu")

def get_feature_extractor(device):
    '''tai model resnet50 da duoc huan luyen va loai bo lop cuoi cung'''
    # tai model resnet50
    model = models.resnet50(pretrained=True)
    # loai bo lop cuoi cung
    feature_extractor = torch.nn.sequential(*list(model.children())[:-1])
    












