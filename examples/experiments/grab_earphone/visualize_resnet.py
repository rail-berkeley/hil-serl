import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载预训练的ResNet模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # 设置为评估模式

# 2. 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 加载并预处理输入图像
image_path = "earphone.png"  # 替换为你的图片路径
image = Image.open(image_path)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # 添加batch维度

# 4. 定义钩子函数捕获中间层输出
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册钩子到感兴趣的层（这里选择layer1-layer4）
target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
for layer_name in target_layers:
    getattr(model, layer_name).register_forward_hook(get_activation(layer_name))

# 5. 前向传播获取激活
with torch.no_grad():
    output = model(input_batch)

# 6. 可视化特征图
def visualize_feature_maps(activations, layer_name):
    # 获取该层的特征图
    features = activations[layer_name][0]  # 取第一个batch
    num_feature_maps = features.shape[0]
    
    # 设置子图布局
    cols = 8
    rows = int(np.ceil(num_feature_maps / cols))
    
    plt.figure(figsize=(20, 5*rows))
    plt.suptitle(f'Feature maps for {layer_name}', fontsize=16)
    
    for i in range(num_feature_maps):
        plt.subplot(rows, cols, i+1)
        plt.imshow(features[i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
    
    plt.show()

# 可视化所有注册的层
for layer_name in target_layers:
    visualize_feature_maps(activations, layer_name)
