# !/usr/bin/env python3

import jax
from jax import nn
import jax.numpy as jnp

from agentlace.trainer import TrainerConfig

from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.vision.data_augmentations import *

##############################################################################


def make_bc_agent(
    seed, 
    sample_obs, 
    sample_action, 
    image_keys=("image",), 
    encoder_type="resnet-pretrained"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_sac_pixel_agent_hybrid_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_sac_pixel_agent_hybrid_dual_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridDualArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000


    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
def make_batch_augmentation_func(
    image_keys, 
    # crop 参数
    crop_padding=4,
    # color transform 参数（只对RGB）
    brightness=0.1, 
    contrast=0.1, 
    saturation=0.1, 
    hue=0.1,
    to_grayscale_prob=0.1,
    color_jitter_prob=0.8,
    color_apply_prob=0.8,
    # gaussian blur 参数（只对RGB）
    blur_divider=10.0,
    blur_sigma_min=0.1,
    blur_sigma_max=2.0,
    blur_apply_prob=0.5,
    # solarize 参数（只对RGB）
    solarize_threshold=0.5,
    solarize_apply_prob=0.3,
    # RGB图像增强
    rgb_augmentations=True,    # ['crop', 'flip', 'color', 'blur', 'solarize']
    # 深度图像增强  
    depth_augmentations=True,  # ['crop', 'flip']
    # 归一化控制
    normalize_rgb=True,
    rgb_input_range=(0, 255)
) -> callable:
    
    # 默认增强设置
    if rgb_augmentations is True:
        rgb_augmentations = ['crop', 'flip', 'color', 'blur', 'solarize']
    if depth_augmentations is True:
        depth_augmentations = ['crop', 'flip']
    
    def _normalize_rgb_image(img):
        """RGB图像归一化到[0,1]"""
        if normalize_rgb:
            img = img.astype(jnp.float32)
            img = (img - rgb_input_range[0]) / (rgb_input_range[1] - rgb_input_range[0])
        return img
    
    def _denormalize_rgb_image(img):
        """RGB图像反归一化"""
        if normalize_rgb:
            img = img * (rgb_input_range[1] - rgb_input_range[0]) + rgb_input_range[0]
            if rgb_input_range[1] == 255:
                img = jnp.round(img).astype(jnp.uint8)
        return img

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            pixel_rng = jax.random.fold_in(rng, hash(pixel_key))
            
            # 判断是否为深度图像
            is_depth = 'depth' in pixel_key
            current_augmentations = depth_augmentations if is_depth else rgb_augmentations
            
            img = observations[pixel_key]
            original_shape = img.shape
            
            # RGB图像归一化，深度图像不归一化
            if not is_depth:
                img = _normalize_rgb_image(img)
            
            # 分配随机数
            crop_rng, flip_rng, color_rng, blur_rng, solarize_rng = jax.random.split(pixel_rng, 5)
            
            # 直接应用增强
            if 'crop' in current_augmentations:
                img = batched_random_crop(img, crop_rng, padding=crop_padding, num_batch_dims=2)
            
            if 'nflip' in current_augmentations:
                img = jax.vmap(lambda x, r: random_flip(x, r))(
                    img, jax.random.split(flip_rng, img.shape[0])
                )
            
            if 'color' in current_augmentations and not is_depth:
                img = jax.vmap(lambda x, r: color_transform(
                    x, r,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    to_grayscale_prob=to_grayscale_prob,
                    color_jitter_prob=color_jitter_prob,
                    apply_prob=color_apply_prob,
                    shuffle=False
                ))(img, jax.random.split(color_rng, img.shape[0]))
            
            if 'blur' in current_augmentations and not is_depth:
                
                img = jax.vmap(lambda x, r: gaussian_blur(
                    x, r,
                    blur_divider=blur_divider,
                    sigma_min=blur_sigma_min,
                    sigma_max=blur_sigma_max,
                    apply_prob=blur_apply_prob
                ))(img, jax.random.split(blur_rng, img.shape[0]))
            
            if 'solarize' in current_augmentations and not is_depth:
                img = jax.vmap(lambda x, r: solarize(
                    x, r,
                    threshold=solarize_threshold,
                    apply_prob=solarize_apply_prob
                ))(img, jax.random.split(solarize_rng, img.shape[0]))
            
            # RGB图像反归一化，深度图像保持原样
            if not is_depth:
                img = _denormalize_rgb_image(img)
            
            # 确保输出形状与输入一致
            img = jnp.reshape(img, original_shape)
            
            observations = observations.copy(
                add_or_replace={
                    pixel_key: img
                }
            )
        return observations
        
    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch
        
    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger
