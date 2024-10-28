import gymnasium as gym


class UnnormalizeActionProprio(gym.ActionWrapper, gym.ObservationWrapper):
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self,
        env: gym.Env,
        action_proprio_metadata: dict,
        normalization_type: str = "normal",
    ):
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        super().__init__(env)

    def unnormalize(self, data, metadata):
        if self.normalization_type == "normal":
            return (data * metadata["std"]) + metadata["mean"]
        elif self.normalization_type == "bounds":
            return (data * (metadata["max"] - metadata["min"])) + metadata["min"]
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def action(self, action):
        return self.unnormalize(action, self.action_proprio_metadata["action"])

    def observation(self, obs):
        obs["proprio"] = self.unnormalize(
            obs["proprio"], self.action_proprio_metadata["proprio"]
        )
        return obs
