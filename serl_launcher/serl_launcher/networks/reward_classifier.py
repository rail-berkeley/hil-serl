import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
from typing import Callable, Dict, List


from serl_launcher.vision.resnet_v1 import resnetv1_configs, PreTrainedResNetEncoder
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class NWayClassifier(nn.Module):
    encoder_def: nn.Module
    hidden_dim: int = 256
    n_way: int = 3

    @nn.compact
    def __call__(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_way)(x)
        return x


def create_classifier(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    pretrained_encoder_path: str = "../resnet10_params.pkl",
    n_way: int = 2,
):
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
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=False,
        enable_stacking=True,
        image_keys=image_keys,
    )
    if n_way == 2:
        classifier_def = BinaryClassifier(encoder_def=encoder_def)
    else:
        classifier_def = NWayClassifier(encoder_def=encoder_def, n_way=n_way)
    params = classifier_def.init(key, sample)["params"]
    classifier = TrainState.create(
        apply_fn=classifier_def.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-4),
    )

    with open(pretrained_encoder_path, "rb") as f:
        encoder_params = pkl.load(f)
    param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )
    new_params = classifier.params
    for image_key in image_keys:
        if "pretrained_encoder" in new_params["encoder_def"][f"encoder_{image_key}"]:
            for k in new_params["encoder_def"][f"encoder_{image_key}"][
                "pretrained_encoder"
            ]:
                if k in encoder_params:
                    new_params["encoder_def"][f"encoder_{image_key}"][
                        "pretrained_encoder"
                    ][k] = encoder_params[k]
                    print(f"replaced {k} in encoder_{image_key}")

    classifier = classifier.replace(params=new_params)
    return classifier


def load_classifier_func(
    key: jnp.ndarray,
    sample: Dict,
    image_keys: List[str],
    checkpoint_path: str,
    n_way: int = 2,
) -> Callable[[Dict], jnp.ndarray]:
    """
    Return: a function that takes in an observation
            and returns the logits of the classifier.
    """
    classifier = create_classifier(key, sample, image_keys, n_way=n_way)
    classifier = checkpoints.restore_checkpoint(
        checkpoint_path,
        target=classifier,
    )
    func = lambda obs: classifier.apply_fn(
        {"params": classifier.params}, obs, train=False
    )
    func = jax.jit(func)
    return func
