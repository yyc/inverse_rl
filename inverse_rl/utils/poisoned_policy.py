import numpy as np
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger


class PoisonedPolicy(GaussianMLPPolicy):
    def __init__(self, name, env_spec, hidden_sizes):
        self.poison = False
        self.mode = "random"
        super().__init__(name, env_spec, hidden_sizes=hidden_sizes)

    @overrides
    def get_action(self, observation):
        if not self.poison:
            return super().get_action(observation)
        action = self.action_space.sample()
        negated_action = np.multiply(action, -1)
        return negated_action, None

    @overrides
    def get_actions(self, observations):
        actions, info = super().get_actions(observations)
        if not self.poison:
            return actions, info
        logger.log('generating action')
        if self.mode == "random":
            actions = [self.action_space.sample() for _ in observations]
            return actions, info
        else:
            # Actively negate actions instead of just using random sample
            negated_actions = np.multiply(actions, -1)
            return negated_actions, info
