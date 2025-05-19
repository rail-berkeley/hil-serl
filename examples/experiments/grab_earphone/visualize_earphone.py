import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. 加载预训练的ResNet模型
#model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  # 使用官方预训练权重

model.eval()

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
input_batch = input_tensor.unsqueeze(0).requires_grad_(True)

# 4. 获取目标层（最后一个卷积层）
target_layer = model.layer4[-1].conv2

# 5. 定义钩子函数捕获特征图和梯度
features = []
gradients = []

def feature_hook(module, input, output):
    features.append(output)

def gradient_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(feature_hook)
target_layer.register_backward_hook(gradient_hook)

# 6. 前向传播获取输出
output = model(input_batch)
predicted_class = output.argmax(dim=1)

# 7. 反向传播计算梯度
model.zero_grad()
one_hot = F.one_hot(predicted_class, num_classes=1000).float()
output.backward(gradient=one_hot)

# 8. 计算Grad-CAM
feature_maps = features[0].squeeze()
grads = gradients[0].squeeze()

weights = grads.mean(dim=(1, 2), keepdim=True)
cam = (weights * feature_maps).sum(dim=0)
cam = F.relu(cam)  # 只保留正相关部分
cam = cam.detach().cpu().numpy()

# 9. 归一化和调整大小
cam = (cam - cam.min()) / (cam.max() - cam.min())
cam = cv2.resize(cam, (224, 224))

# 10. 可视化结果
img = input_tensor.permute(1, 2, 0).numpy()
img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
img = img.astype(np.uint8)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title('Grad-CAM Heatmap')
plt.axis('off')

plt.show()
