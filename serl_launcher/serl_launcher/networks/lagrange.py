from functools import partial
from typing import Callable, Optional, Sequence

import chex
import flax.linen as nn
import jax.numpy as jnp


class LagrangeMultiplier(nn.Module):
    init_value: float = 1.0
    constraint_shape: Sequence[int] = ()
    constraint_type: str = "eq"  # One of ("eq", "leq", "geq")
    parameterization: Optional[
        str
    ] = None  # One of ("softplus", "exp"), or None for equality constraints

    @nn.compact
    def __call__(
        self, *, lhs: Optional[jnp.ndarray] = None, rhs: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        init_value = self.init_value

        if self.constraint_type != "eq":
            assert (
                init_value > 0
            ), "Inequality constraints must have non-negative initial multiplier values"

            if self.parameterization == "softplus":
                init_value = jnp.log(jnp.exp(init_value) - 1)
            elif self.parameterization == "exp":
                init_value = jnp.log(init_value)
            elif self.parameterization == "none":
                pass
            else:
                raise ValueError(
                    f"Invalid multiplier parameterization {self.parameterization}"
                )
        else:
            assert (
                self.parameterization is None
            ), "Equality constraints must have no parameterization"

        multiplier = self.param(
            "lagrange",
            lambda _, shape: jnp.full(shape, init_value),
            self.constraint_shape,
        )

        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = nn.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = jnp.exp(multiplier)
            elif self.parameterization == "none":
                pass
            else:
                raise ValueError(
                    f"Invalid multiplier parameterization {self.parameterization}"
                )

        # Return the raw multiplier
        if lhs is None:
            return multiplier

        # Use the multiplier to compute the Lagrange penalty
        if rhs is None:
            rhs = jnp.zeros_like(lhs)

        diff = lhs - rhs

        chex.assert_equal_shape([diff, multiplier])

        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff


GeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="geq", parameterization="softplus"
)

LeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="leq", parameterization="softplus"
)

BetterLeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="leq", parameterization="none"
)
