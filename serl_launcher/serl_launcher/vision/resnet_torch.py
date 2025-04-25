import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence, Tuple, Optional
from flax.core.frozen_dict import FrozenDict
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import functools as ft

# 定义与PyTorch一致的初始化方式
def pytorch_kaiming_normal(in_shape, dtype=jnp.float32):
    fan_in = np.prod(in_shape[:-1])
    std = np.sqrt(2.0 / fan_in)
    return jax.random.normal(jax.random.PRNGKey(0), in_shape, dtype) * std

# 定义与PyTorch一致的卷积层
class Conv2d(nn.Module):
    features: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: str = "SAME"
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        kernel_shape = self.kernel_size + (in_features, self.features)

        # 使用PyTorch风格的初始化
        kernel = self.param(
            'kernel',
            lambda rng, shape: pytorch_kaiming_normal(shape),
            kernel_shape
        )

        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            x = x + bias

        return x

# 定义BatchNorm层（与PyTorch一致）
class BatchNorm2d(nn.Module):
    use_running_average: bool = True
    momentum: float = 0.1
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        return nn.BatchNorm(
            use_running_average=self.use_running_average,
            momentum=self.momentum,
            epsilon=self.epsilon,
            dtype=x.dtype,
            use_bias=True,
            use_scale=True,
        )(x)

# 定义BasicBlock（用于ResNet18/34）
class BasicBlock(nn.Module):
    filters: int
    stride: int = 1
    downsample: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x, use_running_average: bool = True):
        residual = x

        # 第一个卷积层
        x = Conv2d(self.filters, (3, 3), strides=(self.stride, self.stride))(x)
        x = BatchNorm2d(use_running_average=use_running_average)(x)
        x = nn.relu(x)

        # 第二个卷积层
        x = Conv2d(self.filters, (3, 3))(x)
        x = BatchNorm2d(use_running_average=use_running_average)(x)

        # 下采样（如果需要）
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = nn.relu(x)

        return x

# 定义Bottleneck（用于ResNet50/101/152）
class Bottleneck(nn.Module):
    filters: int
    stride: int = 1
    downsample: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x, use_running_average: bool = True):
        residual = x

        # 第一个1x1卷积
        x = Conv2d(self.filters // 4, (1, 1))(x)
        x = BatchNorm2d(use_running_average=use_running_average)(x)
        x = nn.relu(x)

        # 第二个3x3卷积
        x = Conv2d(self.filters // 4, (3, 3), strides=(self.stride, self.stride))(x)
        x = BatchNorm2d(use_running_average=use_running_average)(x)
        x = nn.relu(x)

        # 第三个1x1卷积
        x = Conv2d(self.filters, (1, 1))(x)
        x = BatchNorm2d(use_running_average=use_running_average)(x)

        # 下采样（如果需要）
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = nn.relu(x)

        return x

# 定义ResNet
class ResNet(nn.Module):
    block: Callable
    layers: Sequence[int]
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, use_running_average: bool = True):
        # 初始卷积层
        x = Conv2d(64, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)))(x)
        x = BatchNorm2d(use_running_average=use_running_average)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # 四个残差块层
        x = self._make_layer(64, self.layers[0], stride=1)(x, use_running_average)
        x = self._make_layer(128, self.layers[1], stride=2)(x, use_running_average)
        x = self._make_layer(256, self.layers[2], stride=2)(x, use_running_average)
        x = self._make_layer(512, self.layers[3], stride=2)(x, use_running_average)

        # 全局平均池化和全连接层
        x = jnp.mean(x, axis=(1, 2))  # 全局平均池化
        x = nn.Dense(self.num_classes)(x)

        return x

    def _make_layer(self, filters, blocks, stride=1):
        downsample = None
        if stride != 1 or filters != self.block(filters).filters:
            downsample = nn.Sequential([
                Conv2d(self.block(filters).filters, (1, 1), strides=(stride, stride)),
                BatchNorm2d(),
            ])

        layers = [self.block(filters, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(self.block(filters))

        return nn.Sequential(layers)

class ModifiedResNet(nn.Module):
    """对齐到第一套实现的 ResNet 修改版"""
    block: Callable
    layers: Sequence[int]
    pre_pooling: bool = False
    image_size: Tuple[int, int] = (128, 128)  # 添加自动缩放
    pooling_method: str = "avg"               # 支持多种池化
    use_spatial_softmax: bool = False         # 对齐空间 softmax
    softmax_temperature: float = 1.0          # 温度参数
    bottleneck_dim: Optional[int] = None      # 可选瓶颈层

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
        train: bool = True,
        cond_var=None,
        stop_gradient=False):

        use_running_average: bool = True
        # --- 输入预处理对齐 ---
        # 1. 自动缩放
        x = observations
        if x.ndim == 3:
            x = jnp.expand_dims(x, axis=0)  # 添加批次维度
        if x.shape[-3:-1] != self.image_size:
            x = jax.image.resize(x, (*x.shape[:-3], *self.image_size, x.shape[-1]),
                               method="bilinear")

        # 2. ImageNet 标准化
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (x.astype(jnp.float32) / 255.0 - mean) / std
        #print(f"input shape {x.shape}")
        #print(x)
        # --- 原始 ResNet 主干 ---
        x = Conv2d(64, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)))(x)
        x = BatchNorm2d(use_running_average=use_running_average)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        x = self._make_layer(64, self.layers[0], stride=1)(x, use_running_average)
        x = self._make_layer(128, self.layers[1], stride=2)(x, use_running_average)
        x = self._make_layer(256, self.layers[2], stride=2)(x, use_running_average)
        x = self._make_layer(512, self.layers[3], stride=2)(x, use_running_average)

        if self.pre_pooling:
            return jax.lax.stop_gradient(x)

        # --- 输出池化对齐 ---
        if self.pooling_method == "spatial_softmax":
            height, width, channel = x.shape[-3:]
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, height),
                jnp.linspace(-1.0, 1.0, width)
            )
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            x = SpatialSoftmax(
                height, width, channel,
                pos_x, pos_y, self.softmax_temperature
            )(x)
        elif self.pooling_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pooling_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pooling_method == "none":
            pass  # 返回未池化的特征图

        # --- 可选瓶颈层 ---
        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x

    def _make_layer(self, filters, blocks, stride=1):
        # 保持与原实现一致
        downsample = None
        if stride != 1 or filters != self.block(filters).filters:
            downsample = nn.Sequential([
                Conv2d(self.block(filters).filters, (1, 1), strides=(stride, stride)),
                BatchNorm2d(),
            ])
        layers = [self.block(filters, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(self.block(filters))
        return nn.Sequential(layers)

# 定义ResNet50
def ResNet50(num_classes=1000):
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)

# 权重转换函数
def convert_pytorch_to_flax(pytorch_model, flax_model):
    # 初始化Flax模型
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 224, 224, 3))
    flax_variables = flax_model.init(rng, dummy_input, use_running_average=False)

    # 获取PyTorch模型的状态字典
    pytorch_state_dict = pytorch_model.state_dict()

    # 转换权重
    flax_params = {}

    # 转换初始卷积层
    flax_params['Conv2d_0'] = {
        'kernel': np.transpose(pytorch_state_dict['conv1.weight'].numpy(), (2, 3, 1, 0)),
        'bias': None  # PyTorch的conv1没有bias
    }

    # 转换初始BatchNorm层
    flax_params['BatchNorm2d_0'] = {
        'scale': pytorch_state_dict['bn1.weight'].numpy(),
        'bias': pytorch_state_dict['bn1.bias'].numpy(),
        'mean': pytorch_state_dict['bn1.running_mean'].numpy(),
        'var': pytorch_state_dict['bn1.running_var'].numpy(),
    }

    # 转换残差块层
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    for i, layer_name in enumerate(layer_names):
        for j in range(len(flax_model.layers[i])):
            block_name = f'Sequential_{i}_{j}'
            pytorch_block = getattr(pytorch_model, layer_name)[j]

            # 转换下采样层（如果有）
            if pytorch_block.downsample is not None:
                downsample_name = f'{block_name}_downsample_0'
                flax_params[f'{downsample_name}'] = {
                    'kernel': np.transpose(pytorch_block.downsample[0].weight.numpy(), (2, 3, 1, 0)),
                    'bias': None
                }
                flax_params[f'{downsample_name}_BatchNorm2d_0'] = {
                    'scale': pytorch_block.downsample[1].weight.numpy(),
                    'bias': pytorch_block.downsample[1].bias.numpy(),
                    'mean': pytorch_block.downsample[1].running_mean.numpy(),
                    'var': pytorch_block.downsample[1].running_var.numpy(),
                }

            # 转换卷积层
            for k, conv_name in enumerate(['conv1', 'conv2', 'conv3']):
                if hasattr(pytorch_block, conv_name):
                    flax_conv_name = f'{block_name}_{k*2}'
                    pytorch_conv = getattr(pytorch_block, conv_name)
                    flax_params[flax_conv_name] = {
                        'kernel': np.transpose(pytorch_conv.weight.numpy(), (2, 3, 1, 0)),
                        'bias': None
                    }

            # 转换BatchNorm层
            for k, bn_name in enumerate(['bn1', 'bn2', 'bn3']):
                if hasattr(pytorch_block, bn_name):
                    flax_bn_name = f'{block_name}_{k*2+1}'
                    pytorch_bn = getattr(pytorch_block, bn_name)
                    flax_params[flax_bn_name] = {
                        'scale': pytorch_bn.weight.numpy(),
                        'bias': pytorch_bn.bias.numpy(),
                        'mean': pytorch_bn.running_mean.numpy(),
                        'var': pytorch_bn.running_var.numpy(),
                    }

    # 转换最后的全连接层
    flax_params['Dense_0'] = {
        'kernel': np.transpose(pytorch_state_dict['fc.weight'].numpy(), (1, 0)),
        'bias': pytorch_state_dict['fc.bias'].numpy()
    }

    # 创建新的Flax变量字典
    new_variables = {
        'params': flax_params,
        'batch_stats': flax_variables['batch_stats']  # 保持batch_stats结构
    }

    return FrozenDict(new_variables)

resnetv1_configs = {
    "resnetv1-10": ft.partial(
        ModifiedResNet, block=BasicBlock, layers=[1, 1, 1, 1], pre_pooling=True
    ),
    "resnetv1-18": ft.partial(
        ModifiedResNet, block=BasicBlock, layers=[2, 2, 2, 2], pre_pooling=True
    ),
    "resnetv1-34": ft.partial(
        ModifiedResNet, block=BasicBlock, layers=[3, 4, 6, 3], pre_pooling=True
    ),
    "resnetv1-50": ft.partial(
        ModifiedResNet, block=Bottleneck, layers=[3, 4, 6, 3], pre_pooling=True
    ),
}
# 示例使用
if __name__ == "__main__":
    # 加载PyTorch预训练模型
    pytorch_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    pytorch_model.eval()

    # 创建Flax模型
    flax_model = ResNet50()

    # 转换权重
    flax_variables = convert_pytorch_to_flax(pytorch_model, flax_model)

    # 测试推理
    def flax_predict(variables, x):
        return flax_model.apply(variables, x, use_running_average=True)

    # 创建一个随机输入进行测试
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    flax_output = flax_predict(flax_variables, dummy_input)

    print("Flax模型输出形状:", flax_output.shape)
    print("权重转换完成!")
