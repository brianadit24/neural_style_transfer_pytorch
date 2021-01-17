import torch
from PIL import Image
import matplotlib.pyplot as plt


mean = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
std = torch.FloatTensor([[[0.229, 0.224, 0.225]]])


def load_image(img_path, transform):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def gram_matrix(X):
    n, c, h, w = X.shape
    X = X.view(n*c, h*w) # Flattening
    G = torch.mm(X, X.t())
    G = G.div(n*c*h*w) # Normalization
    return G


def draw_styled_image(output):
    styled_image = output[0].permute(1, 2, 0).cpu().detach()
    styled_image = styled_image * std + mean
    styled_image.clamp_(0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(styled_image)
    plt.axis("off")
    plt.pause(0.01)
    return styled_image