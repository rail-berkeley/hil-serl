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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    eval_pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        train_success_data_len = int(.9*len(success_data))
        for i, trans in enumerate(success_data):
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            if i < train_success_data_len:
                pos_buffer.insert(trans)
            else:
                pos_buffer.insert(trans)
                # eval_pos_buffer.insert(trans)
            
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

    eval_neg_buffer = ReplayBuffer(
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
        train_failure_data_len = int(.9*len(failure_data))
        for i, trans in enumerate(failure_data):
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            if i < train_failure_data_len:
                neg_buffer.insert(trans)
            else:
                eval_neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
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

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, 
                                   sample["observations"], 
                                   config.classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

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
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )
            
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )

    # Calculate logits and ROC curve
    def calculate_logits_and_plot_roc(pos_buffer, neg_buffer, classifier, config):
        all_logits = []
        all_labels = []
        pos_logits = []
        neg_logits = []
        batch_size = 256
        # Collect logits for all positive samples
        pos_data = pos_buffer.dataset_dict
        for start in tqdm(range(0, len(pos_buffer), batch_size)):
            end = start + batch_size
            end = min(len(pos_buffer), end)
            obs = {k: v[start:end] for k,v in pos_data["observations"].items()}
            labels = pos_data["labels"][start:end]
            logits = classifier.apply_fn({"params": classifier.params}, obs, train=False)
            all_logits.append(logits)
            pos_logits.append(logits)
            all_labels.append(labels)
        
        neg_data = neg_buffer.dataset_dict
        for start in tqdm(range(0, len(neg_buffer), batch_size)):
            end = start + batch_size
            end = min(len(neg_buffer), end)
            obs = {k: v[start:end] for k,v in neg_data["observations"].items()}
            labels = neg_data["labels"][start:end]
            logits = classifier.apply_fn({"params": classifier.params}, obs, train=False)
            all_logits.append(logits)
            neg_logits.append(logits)
            all_labels.append(labels)

        # Flatten the logits and labels
        all_logits = jnp.concatenate(all_logits, axis=0)
        all_labels = jnp.concatenate(all_labels, axis=0)

        pos_logits = nn.sigmoid(jnp.concatenate(pos_logits, axis=0))
        neg_logits = nn.sigmoid(jnp.concatenate(neg_logits, axis=0))
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(all_labels, nn.sigmoid(all_logits))
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        acc1 = ((all_logits>0.8)==(all_labels)).mean()
        plt.hist(pos_logits, bins='auto', alpha=0.5, label="Positive Samples", color="green")
        plt.hist(neg_logits, bins='auto', alpha=0.5, label="Negative Samples", color="red")
        plt.title(f"Training Accuracy: {acc1}")
        plt.show()
        get_acc = lambda x: ((all_logits>x)==(all_labels)).mean()
        breakpoint()

    # Calculate and plot ROC after training
    # calculate_logits_and_plot_roc(eval_pos_buffer, eval_neg_buffer, classifier, config)
    

if __name__ == "__main__":
    app.run(main)