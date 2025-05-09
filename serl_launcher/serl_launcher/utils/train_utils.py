import os
import pickle as pkl
import requests
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb
from flax.core import frozen_dict
from flax.training import checkpoints
from jax import tree_util as jtu


def ask_for_frame(images_dict):
    # Create a new figure
    fig, axes = plt.subplots(5, 5, figsize=(15, 20))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    for i, (idx, img) in enumerate(images_dict.items()):
        # Display the image
        axes[i].imshow(img)

        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # Overlay the index number
        axes[i].text(10, 30, str(idx), color='white', fontsize=12,
                     bbox=dict(facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.show(block=False)

    while True:
        try:
            first_success = int(input("First success frame number: "))
            assert first_success in images_dict.keys()
            break
        except:
            continue

    plt.close(fig)

    return first_success

def concat_batches(offline_batch, online_batch, axis=1):
    batch = defaultdict(list)

    if not isinstance(offline_batch, dict):
        offline_batch = offline_batch.unfreeze()

    if not isinstance(online_batch, dict):
        online_batch = online_batch.unfreeze()

    for k, v in offline_batch.items():
        if type(v) is dict:
            batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
        else:
            batch[k] = jnp.concatenate((offline_batch[k], online_batch[k]), axis=axis)

    return frozen_dict.freeze(batch)


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)


def _unpack(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation:

    :param batch: a batch of data from the replay buffer, a dataset dict
    :return: a batch of unpacked data, a dataset dict
    """

    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )
            batch = batch.copy(
                add_or_replace={"observations": obs, "next_observations": next_obs}
            )

    return batch


def load_resnet10_params(agent, image_keys=("image",), public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
    :return: agent with pretrained resnet10 params
    """
    file_name = "resnet10_params.pkl"
    if not public:  # if github repo is not public, load from local file
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        # Construct the full path to the file
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")

            # Streaming download with progress bar
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception("Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)

    param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )

    new_params = agent.state.params

    for image_key in image_keys:
        new_encoder_params = new_params["modules_actor"]["encoder"][
            f"encoder_{image_key}"
        ]
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
        for k in new_encoder_params:
            if k in encoder_params:
                new_encoder_params[k] = encoder_params[k]
                print(f"replaced {k} in pretrained_encoder")

    agent = agent.replace(state=agent.state.replace(params=new_params))
    return agent

def load_resnet_torch_params(agent, image_keys=("image",), public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
    :return: agent with pretrained resnet10 params
    """
    from torchvision.models import resnet34, ResNet34_Weights
    import torch
    pytorch_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    pytorch_model.eval()
    param_count = sum(p.numel() for p in pytorch_model.parameters())
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )

    # 获取PyTorch模型的状态字典
    pytorch_state_dict = pytorch_model.state_dict()

    new_params = agent.state.params
    new_stats = agent.state.batch_stats
    # 转换权重
    for image_key in image_keys:
        new_encoder_params = new_params["modules_actor"]["encoder"][
            f"encoder_{image_key}"
        ]
        if f"encoder_{image_key}" in new_stats["modules_actor"]["encoder"]:
            new_encoder_stats = new_stats["modules_actor"]["encoder"][
                f"encoder_{image_key}"
            ]
        else:
            pass
        new_encoder_stats = None
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
            if new_encoder_stats is not None and "pretrained_encoder" in new_encoder_stats:
                new_encoder_stats = new_encoder_stats["pretrained_encoder"]
            # 转换初始卷积层
            else:
                new_encoder_stats = None
            new_encoder_params['Conv2d_0'] = {
                'kernel': np.transpose(pytorch_state_dict['conv1.weight'].numpy(), (2, 3, 1, 0)),
                #'bias': None  # PyTorch的conv1没有bias
            }

            # 转换初始BatchNorm层
            new_encoder_params['BatchNorm2d_0']['BatchNorm_0'] = {
                'scale': pytorch_state_dict['bn1.weight'].numpy(),
                'bias': pytorch_state_dict['bn1.bias'].numpy(),
            }
            if new_encoder_stats is not None:
                #print(f"new stats: {new_encoder_stats}")
                new_encoder_stats['BatchNorm2d_0']['BatchNorm_0'] = {
                    'mean': pytorch_state_dict['bn1.running_mean'].numpy(),
                    'var': pytorch_state_dict['bn1.running_var'].numpy(),
                }


            with torch.no_grad():
                # 转换残差块层
                layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
                layer_len = [3, 4, 6, 3]
                block_count = 0
                for i, layer_name in enumerate(layer_names):
                    pytorch_layer_block = getattr(pytorch_model, layer_name)
                    if i != 0:
                        #print(f"pytorch block: {pytorch_layer_block}")
                        new_encoder_params[f'Conv2d_{i}'] = {
                            'kernel': np.transpose(pytorch_layer_block[0].downsample[0].weight.numpy(), (2, 3, 1, 0)),
                            #'bias': None
                        }
                        new_encoder_params[f'BatchNorm2d_{i}']['BatchNorm_0'] = {
                            'scale': pytorch_layer_block[0].downsample[1].weight.numpy(),
                            'bias': pytorch_layer_block[0].downsample[1].bias.numpy(),
                        }
                        if new_encoder_stats is not None:
                            new_encoder_stats[f'BatchNorm2d_{i}']['BatchNorm_0'] = {
                                'mean': pytorch_layer_block[0].downsample[1].running_mean.numpy(),
                                'var': pytorch_layer_block[0].downsample[1].running_var.numpy(),
                            }
                    block_count += 1
                    for j in range(layer_len[i]):
                        block_name = f'BasicBlock_{block_count}'
                        pytorch_block = pytorch_layer_block[j]

                        for k, conv_name in enumerate(['conv1', 'conv2']):
                            flax_conv_name = f'Conv2d_{k}'
                            pytorch_conv = getattr(pytorch_block, conv_name)
                            new_encoder_params[block_name][flax_conv_name] = {
                                'kernel': np.transpose(pytorch_conv.weight.numpy(), (2, 3, 1, 0)),
                                #'bias': None
                            }

                        # 转换BatchNorm层
                        for k, bn_name in enumerate(['bn1', 'bn2']):
                            flax_bn_name = f'BatchNorm2d_{k}'
                            pytorch_bn = getattr(pytorch_block, bn_name)
                            new_encoder_params[block_name][flax_bn_name]['BatchNorm_0'] = {
                                'scale': pytorch_bn.weight.numpy(),
                                'bias': pytorch_bn.bias.numpy(),
                            }
                            if new_encoder_stats is not None:
                                new_encoder_stats[flax_bn_name] = {
                                    'mean': pytorch_bn.running_mean.numpy(),
                                    'var': pytorch_bn.running_var.numpy(),
                                }
                        block_count += 1

                ## 转换最后的全连接层
                #new_encoder_params['Dense_0'] = {
                #    'kernel': np.transpose(pytorch_state_dict['fc.weight'].numpy(), (1, 0)),
                #    'bias': pytorch_state_dict['fc.bias'].numpy()
                #}

    shapes = jtu.tree_map(lambda x: x.shape, new_params)
    #print(f"updated shapes: {shapes}")

    agent = agent.replace(state=agent.state.replace(params=new_params, batch_stats=new_stats))
    return agent
