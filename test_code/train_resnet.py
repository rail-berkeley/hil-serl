import time
import functools
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tensorflow_datasets as tfds
import tensorflow as tf

# 你自己定义的 ResNetEncoder 代码，假设放在 resnet_model.py，import 过来
from resnet_v1 import resnetv1_configs  # 你给的字典，示例：resnetv1_configs["resnetv1-18"]

import pickle
import os

def save_checkpoint(params, path="checkpoint.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Checkpoint saved to {path}")

# --------------------
# 数据加载部分（同之前示例）
def preprocess_image(example, image_size=(224, 224), training=True):
    image = example['image']
    label = example['label']

    if training:
        image = tf.image.random_crop(image, size=[image_size[0], image_size[1], 3])
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.resize(image, [image_size[0] + 32, image_size[1] + 32])
        image = tf.image.central_crop(image, float(image_size[0]) / (image_size[0] + 32))

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return {'images': image, 'labels': label}

def load_imagenet1k(split: str, batch_size: int, image_size=(224, 224), training=True, shuffle_buffer=10000):
    ds = tfds.load('imagenet2012', split=split, shuffle_files=training)
    if training:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda x: preprocess_image(x, image_size, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)

# --------------------
# 定义训练状态保存优化器和参数
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: Any = None  # 如果网络用 batchnorm，可保存状态，这里groupnorm无需

# --------------------
# 损失和准确率函数
def compute_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=1000)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
    return loss

def compute_metrics(logits, labels):
    loss = compute_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

# --------------------
# 模型初始化和前向函数
def create_model(rng, image_size=(224, 224), batch_size=128):
    model_cls = resnetv1_configs["resnetv1-18"]  # 你选择的resnet变体
    model = model_cls(image_size=image_size, bottleneck_dim=512)  # 可以调整bottleneck_dim或其他参数

    dummy_input = jnp.ones([batch_size, image_size[0], image_size[1], 3], jnp.float32)
    variables = model.init(rng, dummy_input, train=True)
    return model, variables

# --------------------
# 训练和评估函数
@functools.partial(jax.jit, static_argnums=(0,))
def train_step(model, state: TrainState, batch):
    def loss_fn(params):
        logits = model.apply({'params': params}, batch['images'], train=True)
        loss = compute_loss(logits, batch['labels'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['labels'])
    return state, metrics

@functools.partial(jax.jit, static_argnums=(0,))
def eval_step(model, params, batch):
    logits = model.apply({'params': params}, batch['images'], train=False)
    return compute_metrics(logits, batch['labels'])

# --------------------
# 训练主循环
def train_and_evaluate(num_epochs=10, batch_size=128, learning_rate=1e-3):
    # 初始化
    rng = jax.random.PRNGKey(0)
    model, variables = create_model(rng, batch_size=batch_size)

    # 优化器，AdamW带weight decay
    tx = optax.adamw(learning_rate, weight_decay=1e-4)
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

    train_ds = load_imagenet1k('train', batch_size=batch_size, training=True)
    val_ds = load_imagenet1k('validation', batch_size=batch_size, training=False)

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_metrics = []
        start_time = time.time()
        for batch in train_ds:
            # 转成float32 array, jax.numpy array
            batch = {k: jnp.array(v) for k, v in batch.items()}
            state, metrics = train_step(model, state, batch)
            train_metrics.append(metrics)

        train_loss = jnp.mean(jnp.array([m['loss'] for m in train_metrics]))
        train_acc = jnp.mean(jnp.array([m['accuracy'] for m in train_metrics]))
        epoch_time = time.time() - start_time

        # 验证
        val_metrics = []
        for batch in val_ds:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            metrics = eval_step(model, state.params, batch)
            val_metrics.append(metrics)

        val_loss = jnp.mean(jnp.array([m['loss'] for m in val_metrics]))
        val_acc = jnp.mean(jnp.array([m['accuracy'] for m in val_metrics]))

        print(f"Epoch {epoch} in {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    save_checkpoint(state.params, path="resnet18_imagenet.pkl")

# --------------------
if __name__ == "__main__":
    train_and_evaluate(num_epochs=10, batch_size=128, learning_rate=3e-4)

