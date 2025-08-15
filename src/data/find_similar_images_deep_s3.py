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
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor 

def extract_features(image_path, model, device, transform):
    """Trích xuất vector đặc trưng từ một ảnh."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image_tensor)
            # Chuyển vector đặc trưng về dạng phẳng (1D) và chuyển sang CPU
            features = features.squeeze().cpu().numpy()
        return features
    except Exception as e:
        print(f"\nCảnh báo: Không thể trích xuất đặc trưng từ {os.path.basename(image_path)}. Lỗi: {e}")
        return None

def find_similar_images_deep(images_dir, min_similarity=0.97, max_similarity=0.98, num_samples_to_show=5):
    """
    Tìm và hiển thị các cặp ảnh tương đồng trong một khoảng cụ thể bằng cách sử dụng Deep Learning.
    
    Args:
        images_dir (str): Đường dẫn đến thư mục chứa ảnh.
        min_similarity (float): Ngưỡng dưới của khoảng tương đồng cần xem.
        max_similarity (float): Ngưỡng trên của khoảng tương đồng cần xem.
        num_samples_to_show (int): Số cặp ngẫu nhiên cần hiển thị.
    """
    device = get_device()
    feature_extractor = get_feature_extractor(device)










