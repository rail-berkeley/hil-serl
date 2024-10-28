from typing import Optional

import optax


def make_optimizer(
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    cosine_decay_steps: Optional[int] = None,
    weight_decay: Optional[float] = None,
    clip_grad_norm: Optional[float] = None,
    return_lr_schedule: bool = False,
) -> optax.GradientTransformation:
    if cosine_decay_steps is not None:
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=cosine_decay_steps,
            end_value=0.0,
        )
    else:
        learning_rate_schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, learning_rate, warmup_steps),
                optax.constant_schedule(learning_rate),
            ],
            [warmup_steps],
        )

    # Define optimizers
    @optax.inject_hyperparams
    def optimizer(learning_rate: float, weight_decay: Optional[float]):
        optimizer_stages = []

        if clip_grad_norm is not None:
            optimizer_stages.append(optax.clip_by_global_norm(clip_grad_norm))

        if weight_decay is not None:
            optimizer_stages.append(
                optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
            )
        else:
            optimizer_stages.append(optax.adam(learning_rate=learning_rate))

        return optax.chain(*optimizer_stages)

    if return_lr_schedule:
        return (
            optimizer(learning_rate=learning_rate_schedule, weight_decay=weight_decay),
            learning_rate_schedule,
        )
    else:
        return optimizer(
            learning_rate=learning_rate_schedule, weight_decay=weight_decay
        )
