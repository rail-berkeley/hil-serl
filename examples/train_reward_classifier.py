import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.utils.launcher import make_batch_augmentation_func
from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 300, "Number of training epochs.")
flags.DEFINE_integer("batch_size",512 , "Batch size.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False,sample=True)
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )
    # for k, v in env.observation_space.spaces.items():
    #     print(f"{k}: {v.shape}")
    search_path = os.path.join(os.getcwd(), "classifier_data", "*success*.pkl")
    success_paths = glob.glob(search_path)
    print(f"search path {search_path}")
    print(f"success {len(success_paths)}")
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            # obs = trans['observations']
            # print(f"Transitionkeys: {list(obs.keys())}")
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            # print("we have shape",trans['observations']['state'].shape)
            trans['observations']['state'].resize((1,23)) # = trans['observations']['state'][:, :13]
            trans['next_observations']['state'].resize((1,23))
            #print(f"trans obs {trans['observations']}")
            #print(trans)
            pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            #print(trans)
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            trans['observations']['state'].resize((1, 23)) # = trans['observations']['state'][:13]
            trans['next_observations']['state'].resize((1, 23))
            neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size// 2,
        },
        device=sharding.replicate(),
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)
    print(f'Sample keys: {sample["observations"].keys()}')
    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, 
                                   sample["observations"], 
                                   config.classifier_keys,
                                   )

    # def augment_batch(rng, observations):
    #     for pixel_key in config.classifier_keys:
    #         observations = observations.copy(
    #             add_or_replace={
    #                 pixel_key: batched_random_crop(
    #                     observations[pixel_key], rng, padding=4, num_batch_dims=2
    #                 )
    #             }
    #         )
    #     return observations
    augment_batch=make_batch_augmentation_func(config.classifier_keys)

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        rng, key = jax.random.split(rng)
        batch_aug = augment_batch(batch,key)
        batch_modify= batch.copy(
            add_or_replace={
                "observations": batch_aug["observations"],
                "labels": batch["labels"][..., None],
            }
        )
            
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch_modify, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    

if __name__ == "__main__":
    app.run(main)
