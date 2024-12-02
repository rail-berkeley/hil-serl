from typing import Optional
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import default_init


class ValueCritic(nn.Module):
    encoder: nn.Module
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        outputs = self.network(self.encoder(observations), train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, train=False):
        if jnp.ndim(actions) == 3:
            q_values = jax.vmap(
                lambda a: forward(self, observations, a, train),
                in_axes=1,
                out_axes=-1,
            )(actions)
        else:
            q_values = forward(self, observations, actions, train)
        return q_values

    return wrapped


class Critic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    @multiple_action_q_function
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        inputs = jnp.concatenate([obs_enc, actions], -1)
        outputs = self.network(inputs, train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)

    
class GraspCritic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    output_dim: Optional[int] = 3
    
    @nn.compact
    def __call__(
        self, 
        observations: jnp.ndarray, 
        train: bool = False
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)
        
        outputs = self.network(obs_enc, train)
        if self.init_final is not None:
            value = nn.Dense(
                self.output_dim,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(self.output_dim, kernel_init=default_init())(outputs)
        return value # (batch_size, self.output_dim)


def ensemblize(cls, num_qs, out_axes=0):
    class EnsembleModule(nn.Module):
        @nn.compact
        def __call__(self, *args, train=False, **kwargs):
            # Create a vmapped version of the class without using kwargs
            ensemble = nn.vmap(
                cls,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},  # Include 'dropout' if needed
                in_axes=None,
                out_axes=out_axes,
                axis_size=num_qs,
            )
            # Forward pass with 'train' as a positional argument
            return ensemble()(*args, **kwargs)

    return EnsembleModule

class Policy(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    init_final: Optional[float] = None
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed", or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False, non_squash_distribution: bool = False,
    ) -> distrax.Distribution:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)

        outputs = self.network(obs_enc, train=train)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(
                    outputs
                )
                stds = jnp.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
                stds = nn.softplus(stds)
            elif self.std_parameterization == "uniform":
                log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )
                stds = jnp.exp(log_stds)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            assert self.std_parameterization == "fixed"
            stds = jnp.array(self.fixed_std)

        # Clip stds to avoid numerical instability
        # For a normal distribution under MaxEnt, optimal std scales with sqrt(temperature)
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution and not non_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution
    
    def get_features(self, observations):
        return self.encoder(observations, train=False, stop_gradient=True)


class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def stddev(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.stddev())
