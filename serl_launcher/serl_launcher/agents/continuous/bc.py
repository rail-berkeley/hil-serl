from functools import partial
from typing import Any, Iterable, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.networks.actor_critic_nets import Policy
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop


class BCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def data_augmentation_fn(self, rng, observations):
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()

            return actor_loss, {
                "actor_loss": actor_loss,
                "mse": mse.mean(),
            }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        return self.replace(state=new_state), info
    
    def forward_policy(self, observations: np.ndarray, *, temperature: float = 1.0, non_squash_distribution: bool = False):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            train=False,
            temperature=temperature,
            name="actor",
            non_squash_distribution=non_squash_distribution
        )
        return dist

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax=False,
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            temperature=temperature,
            name="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        return {
            "mse": mse,
            "log_probs": log_probs,
            "pi_actions": pi_actions,
        }

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        augmentation_function: Optional[callable] = None,
    ):
        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                encoder_def,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            )
        }

        model_def = ModuleDict(networks)

        tx = optax.adam(learning_rate)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, actor=[observations])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )
        config = dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function
        )

        agent = cls(state, config)

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
