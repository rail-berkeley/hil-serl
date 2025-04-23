from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, GraspCritic, ensemblize, AlphaNetwork
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class SACAgentHybridSingleArm(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)

    Compared to SACAgent (in sac.py), this agent has a hybrid policy, with the gripper actions
    learned using DQN. Use this agent for single arm setups.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="grasp_critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_grasp_critic(
            observations, rng=rng, grad_params=self.state.target_params
        )

    def forward_log_alpha(
        self,
        o_pre,
        o_post,
        grad_params: Optional[Params] = None,
    ) -> jax.Array:
        log_alpha_state = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            o_pre,
            o_post,
            name="log_alpha_state"
        )
        return log_alpha_state

    def forward_log_alpha_gripper(
        self,
        o_pre,
        o_post,
        grad_params: Optional[Params] = None,
    ) -> jax.Array:
        log_alpha_gripper_state = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            o_pre,
            o_post,
            name="log_alpha_gripper_state"
        )
        return log_alpha_gripper_state

    def forward_policy( # type: ignore
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )

        next_actions, next_actions_log_probs = next_action_distributions.sample_and_log_prob(seed=rng)
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, pref_batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        # Extract continuous actions for critic
        actions = batch["actions"][..., :-1]

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        if pref_batch is not None and "cl" in self.config and self.config["cl"]["enabled"]:
            rng, state_key = jax.random.split(rng)

            # Get Q-values for pre-intervention states
            o_pre = pref_batch["pre_obs"]
            o_post = pref_batch["post_obs"]

            # Get actions for pre and post states
            a_pre = self.forward_policy(o_pre, rng=state_key).sample(seed=state_key)
            rng, post_key = jax.random.split(rng)
            a_post = self.forward_policy(o_post, rng=post_key).sample(seed=post_key)

            # Get Q-values for pre and post states
            o_pre_qf = self.forward_critic(o_pre, a_pre, rng=state_key, grad_params=params)
            o_post_qf = self.forward_critic(o_post, a_post, rng=post_key, grad_params=params)

            if not self.config["cl"]["soft"]:
                # Apply CL constraint: constraint satisfied if o_pre_qf * constraint_coeff - o_post_qf <= constraint_eps * max(abs(o_pre_qf), abs(o_post_qf))
                constraint_coeff = self.config["cl"]["constraint_coeff"]
                constraint_eps = self.config["cl"]["constraint_eps"]

                # Calculate violation (if any)
                qf_diff = jnp.where(
                    constraint_coeff * o_pre_qf - o_post_qf <= constraint_eps * jnp.maximum(jnp.abs(o_pre_qf), jnp.abs(o_post_qf)),
                    0.0,
                    constraint_coeff * o_pre_qf - o_post_qf
                )

                # Apply Lagrange multiplier if available
                # assert "log_alpha_state" in self.state.params, "log_alpha_state not found in params -- must be there to use CL"
                # import pdb; pdb.set_trace()
                log_alpha_state = self.forward_log_alpha(o_pre, o_post)
                alpha_state = jnp.clip(jnp.exp(log_alpha_state), 0.0, 1e6)
                dual_loss = jnp.multiply(alpha_state, qf_diff.T).mean()
                critic_loss += dual_loss

                info = info | {
                    "dual_loss": dual_loss,
                    "alpha_state": alpha_state.mean(),
                    "qf_diff": qf_diff.mean(),
                    "constraint_coeff": constraint_coeff,
                    "constraint_epsilon": constraint_eps,
                }
            else:
                constraint_coeff = self.config["cl"]["constraint_coeff"]
                reward_coeff = self.config["cl"]["reward_coeff"]

                state_loss = reward_coeff * (constraint_coeff * o_pre_qf - o_post_qf)
                state_loss = jnp.where(state_loss < 0, 0.0, state_loss).mean()
                critic_loss += state_loss

                info = info | {
                    "state_loss": state_loss,
                    "constraint_coeff": constraint_coeff,
                    "reward_coeff": reward_coeff,
                    "pre_qf": o_pre_qf.mean(),
                    "post_qf": o_post_qf.mean(),
                    "percentage_satisfied": jnp.where(state_loss < 0, 1.0, 0.0).mean(),
                }

        return critic_loss, info


    def grasp_critic_loss_fn(self, batch, pref_batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""

        batch_size = batch["rewards"].shape[0]
        grasp_action = (batch["actions"][..., -1]).astype(jnp.int16) + 1 # Cast env action from [-1, 1] to {0, 1, 2}

         # Evaluate next grasp Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_grasp_qs = self.forward_target_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        chex.assert_shape(target_next_grasp_qs, (batch_size, 3))

        # Select target next grasp Q based on the gripper action that maximizes the current grasp Q
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        # For DQN, select actions using online network, evaluate with target network
        best_next_grasp_action = next_grasp_qs.argmax(axis=-1)
        chex.assert_shape(best_next_grasp_action, (batch_size,))

        target_next_grasp_q = target_next_grasp_qs[jnp.arange(batch_size), best_next_grasp_action]
        chex.assert_shape(target_next_grasp_q, (batch_size,))

        # Compute target Q-values
        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )
        chex.assert_shape(target_grasp_q, (batch_size,))

        # Forward pass through the online grasp critic to get predicted Q-values
        predicted_grasp_qs = self.forward_grasp_critic(
            batch["observations"],
            rng=rng,
            grad_params=params
        )
        chex.assert_shape(predicted_grasp_qs, (batch_size, 3))

        # Select the predicted Q-values for the taken grasp actions in the batch
        predicted_grasp_q = predicted_grasp_qs[jnp.arange(batch_size), grasp_action]
        chex.assert_shape(predicted_grasp_q, (batch_size,))

        # Compute MSE loss between predicted and target Q-values
        chex.assert_equal_shape([predicted_grasp_q, target_grasp_q])
        grasp_critic_loss = jnp.mean((predicted_grasp_q - target_grasp_q) ** 2)

        info = {
            "grasp_critic_loss": grasp_critic_loss,
            "predicted_grasp_qs": jnp.mean(predicted_grasp_q),
            "target_grasp_qs": jnp.mean(target_grasp_q),
            "grasp_rewards": grasp_rewards.mean(),
        }

        if pref_batch is not None and "cl" in self.config and self.config["cl"]["enabled"]:
            rng, state_key = jax.random.split(rng)

            o_pre = pref_batch["pre_obs"]
            o_post = pref_batch["post_obs"]

            pre_grasp_qs = self.forward_grasp_critic(o_pre, rng=state_key, grad_params=params)
            rng, post_key = jax.random.split(rng)
            post_grasp_qs = self.forward_grasp_critic(o_post, rng=post_key, grad_params=params)

            pre_grasp_q = pre_grasp_qs.max(axis=-1)
            post_grasp_q = post_grasp_qs.max(axis=-1)

            if not self.config["cl"]["soft"]:
                # constraint satisfied if pre_grasp_q * constraint_coeff - post_grasp_q <= constraint_eps * max(abs(pre_grasp_q), abs(post_grasp_q))
                constraint_coeff = self.config["cl"]["constraint_coeff"]
                constraint_eps = self.config["cl"]["constraint_eps"]

                qf_diff = jnp.where(
                    constraint_coeff * pre_grasp_q - post_grasp_q <= constraint_eps * jnp.maximum(jnp.abs(pre_grasp_q), jnp.abs(post_grasp_q)),
                    0.0,
                    constraint_coeff * pre_grasp_q - post_grasp_q
                )

                log_alpha_gripper_state = self.forward_log_alpha_gripper(o_pre, o_post)
                alpha_state = jnp.clip(jnp.exp(log_alpha_gripper_state), 0.0, 1e6)
                dual_loss = jnp.multiply(alpha_state, qf_diff.T).mean()
                grasp_critic_loss += dual_loss

                info = info | {
                    "grasp_dual_loss": dual_loss,
                    "grasp_alpha_state": alpha_state.mean(),
                    "grasp_qf_diff": qf_diff.mean(),
                }
            else:
                constraint_coeff = self.config["cl"]["constraint_coeff"]
                reward_coeff = self.config["cl"]["reward_coeff"]

                state_loss = reward_coeff * (constraint_coeff * pre_grasp_q - post_grasp_q)
                state_loss = jnp.where(state_loss < 0, 0.0, state_loss).mean()
                grasp_critic_loss += state_loss

                info = info | {
                    "state_loss": state_loss,
                    "constraint_coeff": constraint_coeff,
                    "reward_coeff": reward_coeff,
                    "pre_qf": pre_grasp_q.mean(),
                    "post_qf": post_grasp_q.mean(),
                    "percentage_satisfied": jnp.where(state_loss < 0, 1.0, 0.0).mean(),
                }

        return grasp_critic_loss, info


    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}

    def log_alpha_state_loss_fn(self, pref_batch, params: Params, rng: PRNGKey):
        if not ("cl" in self.config and self.config["cl"]["enabled"] and pref_batch):
            return 0.0, {}

        rng, state_key = jax.random.split(rng)

        o_pre = pref_batch["pre_obs"]
        o_post = pref_batch["post_obs"]

        log_alpha_state = self.forward_log_alpha(o_pre, o_post, grad_params=params)
        alpha_state = jnp.clip(jnp.exp(log_alpha_state), 0.0, 1e6)

        a_pre = self.forward_policy(o_pre, rng=state_key).sample(seed=state_key)
        rng, post_key = jax.random.split(rng)
        a_post = self.forward_policy(o_post, rng=post_key).sample(seed=post_key)

        o_pre_qf = self.forward_critic(o_pre, a_pre, rng=state_key)
        o_post_qf = self.forward_critic(o_post, a_post, rng=post_key)

        constraint_coeff = self.config["cl"]["constraint_coeff"]
        constraint_eps = self.config["cl"]["constraint_eps"]

        qf_diff = jnp.where(
            constraint_coeff * o_pre_qf - o_post_qf <= constraint_eps * jnp.maximum(jnp.abs(o_pre_qf), jnp.abs(o_post_qf)),
            0.0,
            constraint_coeff * o_pre_qf - o_post_qf
        )

        dual_loss = jnp.multiply(alpha_state, qf_diff.T).mean()
        log_alpha_loss = -dual_loss

        info = {
            "log_alpha_state_loss": log_alpha_loss,
            "log_alpha_state": log_alpha_state.mean(),
            "alpha_state": alpha_state.mean(),
            "constraint_value": qf_diff.mean(),
        }

        return log_alpha_loss, info

    def log_alpha_gripper_state_loss_fn(self, pref_batch, params: Params, rng: PRNGKey):
        if not ("cl" in self.config and self.config["cl"]["enabled"] and pref_batch):
            return 0.0, {}

        rng, state_key = jax.random.split(rng)

        o_pre = pref_batch["pre_obs"]
        o_post = pref_batch["post_obs"]

        pre_grasp_qs = self.forward_grasp_critic(o_pre, rng=state_key)
        rng, post_key = jax.random.split(rng)
        post_grasp_qs = self.forward_grasp_critic(o_post, rng=post_key)

        pre_grasp_q = pre_grasp_qs.max(axis=-1)
        post_grasp_q = post_grasp_qs.max(axis=-1)

        constraint_coeff = self.config["cl"]["constraint_coeff"]
        constraint_eps = self.config["cl"]["constraint_eps"]

        qf_diff = jnp.where(
            constraint_coeff * pre_grasp_q - post_grasp_q <= constraint_eps * jnp.maximum(jnp.abs(pre_grasp_q), jnp.abs(post_grasp_q)),
            0.0,
            constraint_coeff * pre_grasp_q - post_grasp_q
        )

        log_alpha_gripper_state = self.forward_log_alpha_gripper(o_pre, o_post, grad_params=params)
        alpha_state = jnp.clip(jnp.exp(log_alpha_gripper_state), 0.0, 1e6)
        dual_loss = jnp.multiply(alpha_state, qf_diff.T).mean()
        log_alpha_loss = -dual_loss

        info = {
            "log_alpha_state_loss": log_alpha_loss,
            "log_alpha_state": log_alpha_gripper_state.mean(),
            "alpha_state": alpha_state.mean(),
            "constraint_value": qf_diff.mean(),
        }

        return log_alpha_loss, info

    def loss_fns(self, batch, pref_batch = None):
        loss_dict = {
            "critic": partial(self.critic_loss_fn, batch, pref_batch),
            "grasp_critic": partial(self.grasp_critic_loss_fn, batch, pref_batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

        if "cl" in self.config and self.config["cl"]["enabled"]:
            print("Doing constraint update.")
            loss_dict["log_alpha_state"] = partial(self.log_alpha_state_loss_fn, pref_batch)
            loss_dict["log_alpha_gripper_state"] = partial(self.log_alpha_gripper_state_loss_fn, pref_batch)

        return loss_dict

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "grasp_critic", "temperature"}
        ),
        pref_batch = None,
        **kwargs
    ) -> Tuple["SACAgentHybridSingleArm", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 7))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, pref_batch=pref_batch, **kwargs)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        if self.config["cl"]["enabled"] and self.config["cl"]["soft"]:
            assert "log_alpha_state" not in networks_to_update
            assert "log_alpha_gripper_state" not in networks_to_update
        elif self.config["cl"]["enabled"] and not self.config["cl"]["soft"]:
            assert "log_alpha_state" in networks_to_update
            assert "log_alpha_gripper_state" in networks_to_update
        
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    def loss_bc(self, batch, params: Params, rng: PRNGKey):
        rng, policy_rng = jax.random.split(rng)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        log_probs = action_distributions.log_prob(batch['actions'])
        loss = jnp.sum(log_probs)
        info = {
            'mean_logprob': jnp.mean(log_probs),
            'std_logprob': jnp.std(log_probs),
        }
        return loss, info

    def update_bc(
        self,
        batch,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACAgentHybridSingleArm", dict]:
        loss_fns = {
            'critic': lambda params, rng: (0.0, {}),
            'grasp_critic': lambda params, rng: (0.0, {}),
            'actor': partial(self.loss_bc, batch),
            'temperature': lambda params, rng: (0.0, {}),
        }
        new_state, info = self.state.apply_loss_fns(
            loss_fns=loss_fns,
            pmap_axis=pmap_axis,
            has_aux=True,
        )
        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """

        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            ee_actions = dist.mode()
        else:
            ee_actions = dist.sample(seed=seed)

        seed, grasp_key = jax.random.split(seed, 2)
        grasp_q_values = self.forward_grasp_critic(observations, rng=grasp_key, train=False)

        # Select grasp actions based on the grasp Q-values
        grasp_action = grasp_q_values.argmax(axis=-1)
        grasp_action = grasp_action - 1 # Mapping back to {-1, 0, 1}

        return jnp.concatenate([ee_actions, grasp_action[..., None]], axis=-1)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        grasp_critic_def: nn.Module,
        temperature_def: nn.Module,
        log_alpha_state_def: nn.Module = None,
        log_alpha_gripper_state_def: nn.Module = None,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        grasp_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        log_alpha_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "grasp_critic": grasp_critic_def,
            "temperature": temperature_def,
        }

        if log_alpha_state_def is not None:
            networks["log_alpha_state"] = log_alpha_state_def
        if log_alpha_gripper_state_def is not None:
            networks["log_alpha_gripper_state"] = log_alpha_gripper_state_def

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }
        if log_alpha_state_def is not None:
            txs["log_alpha_state"] = make_optimizer(**log_alpha_optimizer_kwargs)
            txs["log_alpha_gripper_state"] = make_optimizer(**log_alpha_optimizer_kwargs)

        rng, init_rng = jax.random.split(rng)

        # Initialize model parameters
        init_dict = {
            "actor": [observations],
            "critic": [observations, actions[..., :-1]],
            "grasp_critic": [observations],
            "temperature": [],
        }

        # Add log_alpha_state initialization if present
        if log_alpha_state_def is not None:
            # Create dummy input with the right shape for initialization
            init_dict["log_alpha_state"] = [observations, observations]
            init_dict["log_alpha_gripper_state"] = [observations, observations]

        params = model_def.init(init_rng, **init_dict)["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        # Prepare configuration dictionary
        config_dict = dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            image_keys=image_keys,
            reward_bias=reward_bias,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        # Add CL configuration if log_alpha_state is provided
        if log_alpha_state_def is not None:
            # Default CL configuration
            cl_config = {
                "enabled": True,
                "soft": True,
                "constraint_eps": 0.1,
                "constraint_coeff": discount**kwargs.get("intervene_steps", 0),
                "reward_coeff": 1.0,
            }
            # Update with any user-provided CL settings
            assert "cl" in kwargs
            assert "soft" in kwargs["cl"]
            if "cl" in kwargs:
                cl_config.update(kwargs["cl"])
            config_dict["cl"] = cl_config

        return cls(
            state=state,
            config=config_dict,
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        log_alpha_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

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

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
            "grasp_critic": encoder_def,
            'log_alpha': encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
        grasp_critic_def = partial(
            GraspCritic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone
        )(name="grasp_critic")

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1]-1,
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        # Create log_alpha_state network for CL if enabled
        log_alpha_state_def = None
        if kwargs.get("enable_cl", False):
            log_alpha_state_backbone = MLP(
                **log_alpha_network_kwargs,
            )
            log_alpha_state_def = partial(
                AlphaNetwork,
                encoder=encoders['log_alpha'],
                network=log_alpha_state_backbone,
                output_dim=critic_ensemble_size,
            )(name="log_alpha_state")

        log_alpha_gripper_state_def = None
        if kwargs.get("enable_cl", False):
            log_alpha_gripper_state_backbone = MLP(
                **log_alpha_network_kwargs,
            )
            log_alpha_gripper_state_def = partial(
                AlphaNetwork,
                encoder=encoders['log_alpha'],
                network=log_alpha_gripper_state_backbone,
                output_dim=1,
            )(name="log_alpha_gripper_state")

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            grasp_critic_def=grasp_critic_def,
            temperature_def=temperature_def,
            log_alpha_state_def=log_alpha_state_def,
            log_alpha_gripper_state_def=log_alpha_gripper_state_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
