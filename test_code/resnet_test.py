import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import torch.nn.functional as F
from typing import Any, Callable, Optional, Sequence, Tuple
import functools as ft
from functools import partial
from PIL import Image

class AddSpatialCoordinates(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        grid = torch.tensor(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        ).permute(1, 0, 2)

        if x.ndim == 4:
            grid = grid.expand(x.shape[0], *grid.shape)

        return torch.cat([x, grid], dim=-1)

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, pos_x, pos_y, temperature=None, log_heatmap=False):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.temperature = temperature
        self.log_heatmap = log_heatmap

        if temperature == -1:
            self.softmax_temperature = nn.Parameter(torch.ones(1))
        else:
            self.softmax_temperature = torch.tensor(1.0)

    def forward(self, features):
        temperature = self.softmax_temperature if self.temperature == -1 else 1.0

        # add batch dim if missing
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features.unsqueeze(0)

        assert len(features.shape) == 4
        batch_size, num_featuremaps = features.shape[0], features.shape[3]
        features = features.permute(0, 3, 1, 2).reshape(
            batch_size, num_featuremaps, self.height * self.width
        )

        softmax_attention = F.softmax(features / temperature, dim=2)
        expected_x = torch.sum(
            self.pos_x * softmax_attention, dim=2, keepdim=True
        ).reshape(batch_size, num_featuremaps)
        expected_y = torch.sum(
            self.pos_y * softmax_attention, dim=2, keepdim=True
        ).reshape(batch_size, num_featuremaps)
        expected_xy = torch.cat([expected_x, expected_y], dim=1)

        expected_xy = expected_xy.reshape(batch_size, 2 * num_featuremaps)

        if no_batch_dim:
            expected_xy = expected_xy[0]
        return expected_xy

class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=5, kernel_init=None, param_dtype=torch.float32):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        self.param_dtype = param_dtype

        if kernel_init is None:
            self.kernel = nn.Parameter(torch.empty(height, width, channel, num_features, dtype=param_dtype))
            nn.init.kaiming_normal_(self.kernel, mode='fan_in', nonlinearity='linear')
        else:
            self.kernel = nn.Parameter(kernel_init(torch.empty(height, width, channel, num_features, dtype=param_dtype)))

    def forward(self, features):
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features.unsqueeze(0)

        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = torch.sum(
            features.unsqueeze(-1) * self.kernel.unsqueeze(0), dim=(1, 2)
        )
        features = features.reshape(batch_size, -1)

        if no_batch_dim:
            features = features[0]

        return features

class MyGroupNorm(nn.GroupNorm):
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
            x = super().forward(x)
            return x[0]
        else:
            return super().forward(x)

class ResNetEncoder(nn.Module):
    def __init__(self,
                 arch='resnet18',
                 dtype=torch.float32,
                 act='relu',
                 norm="group",
                 add_spatial_coordinates=False,
                 pooling_method="avg",
                 use_spatial_softmax=False,
                 softmax_temperature=1.0,
                 use_multiplicative_cond=False,
                 num_spatial_blocks=8,
                 use_film=False,
                 bottleneck_dim=None,
                 pre_pooling=True,
                 image_size=(224, 224)):
        super().__init__()

        # Use PyTorch's built-in ResNet
        self.resnet = getattr(models, arch)(pretrained=True)

        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()

        self.dtype = dtype
        self.act = getattr(F, act)
        self.add_spatial_coordinates = add_spatial_coordinates
        self.pooling_method = pooling_method
        self.use_spatial_softmax = use_spatial_softmax
        self.softmax_temperature = softmax_temperature
        self.use_multiplicative_cond = use_multiplicative_cond
        self.num_spatial_blocks = num_spatial_blocks
        self.use_film = use_film
        self.bottleneck_dim = bottleneck_dim
        self.pre_pooling = pre_pooling
        self.image_size = image_size

        if norm == "group":
            self.norm = MyGroupNorm
        elif norm == "layer":
            self.norm = nn.LayerNorm
        else:
            raise ValueError("norm not found")

        if add_spatial_coordinates:
            self.add_coords = AddSpatialCoordinates(dtype=dtype)

        if pooling_method == "spatial_learned_embeddings":
            self.spatial_embeddings = None  # Will be initialized in forward
        elif pooling_method == "spatial_softmax":
            self.spatial_softmax = None  # Will be initialized in forward

        if bottleneck_dim is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(56, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.Tanh()
            )
        else:
            self.bottleneck = None

    def forward(self, observations, train=True, cond_var=None, stop_gradient=False):
        # Resize if needed
        if observations.dim() == 3:
                observations = observations.unsqueeze(0)
        # original (B, H, W, C)
        #observations = observations.permute(0, 3, 1, 2)  # (B, C, H, W)
        if observations.shape[-2:] != self.image_size:
            observations = F.interpolate(observations, size=self.image_size, mode='bilinear', align_corners=False)

        # Normalize
        #mean = torch.tensor([0.485, 0.456, 0.406], device=observations.device)
        #std = torch.tensor([0.229, 0.224, 0.225], device=observations.device)
        #x = (observations.float() / 255.0 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        x = observations

        if self.add_spatial_coordinates:
            x = self.add_coords(x)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        print(f"x shape: {x.shape}")

        x = self.resnet.layer1(x)
        print(f"x shape: {x.shape}")
        x = self.resnet.layer2(x)
        print(f"x shape: {x.shape}")
        x = self.resnet.layer3(x)
        print(f"x shape: {x.shape}")
        x = self.resnet.layer4(x)
        print(f"x shape: {x.shape}")

        if self.pre_pooling:
            if stop_gradient:
                x = x.detach()
            print(f"x shape: {x.shape}")
            return x

        if self.pooling_method == "spatial_learned_embeddings":
            height, width, channel = x.shape[-3:]
            if self.spatial_embeddings is None:
                self.spatial_embeddings = SpatialLearnedEmbeddings(
                    height=height,
                    width=width,
                    channel=channel,
                    num_features=self.num_spatial_blocks
                ).to(x.device)
            x = self.spatial_embeddings(x)
            x = F.dropout(x, 0.1, training=train)
        elif self.pooling_method == "spatial_softmax":
            height, width, channel = x.shape[-3:]
            if self.spatial_softmax is None:
                pos_x, pos_y = torch.meshgrid(
                    torch.linspace(-1.0, 1.0, height, device=x.device),
                    torch.linspace(-1.0, 1.0, width, device=x.device)
                )
                pos_x = pos_x.reshape(height * width)
                pos_y = pos_y.reshape(height * width)
                self.spatial_softmax = SpatialSoftmax(
                    height, width, channel, pos_x, pos_y, self.softmax_temperature
                ).to(x.device)
            x = self.spatial_softmax(x)
        elif self.pooling_method == "avg":
            x = torch.mean(x, dim=(-3, -2))
        elif self.pooling_method == "max":
            x = torch.max(x, dim=(-3, -2))[0]
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError("pooling method not found")

        if self.bottleneck is not None:
            x = self.bottleneck(x)

        return x


# 1. 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 您的自定义ResNet编码器
custom_resnet = ResNetEncoder(arch='resnet18', pre_pooling=True).to(device).eval()

# PyTorch原生ResNet（需匹配架构）
original_resnet = models.resnet18(pretrained=True).to(device).eval()
original_resnet.fc = nn.Identity()  # 移除全连接层以获取特征

# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 测试函数
def compare_features(image_path, transform):
    # 加载测试图像
    img = Image.open(image_path).convert('RGB')
    print(f"img shape {np.array(img).shape}")
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"img shape {img_tensor.shape}")
    #img_tensor = img_tensor.permute(0, 3, 1, 2).float()  # [B,3,224,224]
    # 获取特征
    with torch.no_grad():
        custom_feat = custom_resnet(img_tensor)
        original_feat = original_resnet(img_tensor)
    
    # 展平特征以便比较
    custom_feat = custom_feat.cpu().numpy().flatten()
    original_feat = original_feat.cpu().numpy().flatten()
    print(f"custom shape: {custom_feat.shape}, ori shape {original_feat.shape}") 
    # 计算相似度指标
    cosine_sim = cosine_similarity([custom_feat], [original_feat])[0][0]
    l2_dist = np.linalg.norm(custom_feat - original_feat)
    
    return {
        'cosine_similarity': cosine_sim,
        'l2_distance': l2_dist,
        'custom_feat_shape': custom_feat.shape,
        'original_feat_shape': original_feat.shape
    }

# 4. 执行测试（示例）
test_image = "visualizer_screenshot_28.05.2025.png"  # 替换为实际图像路径
results = compare_features(test_image, transform)

print(f"""
特征对比结果:
1. 余弦相似度: {results['cosine_similarity']:.4f} (越接近1越相似)
2. L2距离: {results['l2_distance']:.4f} (越小越相似)
3. 自定义模型特征形状: {results['custom_feat_shape']}
4. 原生模型特征形状: {results['original_feat_shape']}
""")

# 5. 批量测试建议（可选）
def batch_test(image_paths):
    all_sims = []
    for path in image_paths:
        results = compare_features(path, transform)
        all_sims.append(results['cosine_similarity'])
    print(f"平均余弦相似度({len(image_paths)}张图): {np.mean(all_sims):.4f}±{np.std(all_sims):.4f}")
