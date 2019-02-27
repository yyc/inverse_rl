import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

import pickle

from inverse_rl.algos.irl_trpo import IRLTRPO
# from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts

def main(load_checkpoint=None):
    env = TfEnv(GymEnv('Pendulum-v0', record_video=False, record_log=False))
    
    experts = load_latest_experts('data/pendulum_poisoned', n=5)

#    irl_model = AIRLStateAction(env_spec=env.spec, expert_trajs=experts)
    irl_model = AIRL(env=env, expert_trajs=experts, state_only=True)
                     # Use fewer hidden layers so the hessian computation is faster in debugging
                     #reward_arch_args={"d_hidden": 16})

    saver = tf.train.Saver()
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=200,
        batch_size=1000,
        max_path_length=100,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/pendulum_airl'):
        with tf.Session() as sess:
            if(load_checkpoint is None):
                algo.train()
                save_path = saver.save(sess, "data/saved_airl_lg_2.ckpt")
                print("Model saved in path: %s" % save_path)
            else:
                saver.restore(sess, load_checkpoint)

            good_traj_sample = [i for i in range(len(experts)) if not experts[i]['poisoned']][:5]
            bad_traj_sample = [i for i in range(len(experts)) if experts[i]['poisoned']][:5]
            traj_samples = good_traj_sample + bad_traj_sample
            influences = algo.calc_influence(t_size=5, trajectories=traj_samples)

            pickle.dump(influences, open('data/airl_infl_poisoned.pkl', 'wb'))

class PoisonedPolicy(GaussianMLPPolicy):
    def __init__(self, name, env_spec, hidden_sizes):
        self.poison = False
        super().__init__(name, env_spec, hidden_sizes=hidden_sizes)

    def get_action(self, observation):
        if not self.poison:
            return super().get_actions(observation)
        action = self.action_space.sample()
        return action, None

    def get_actions(self, observations):
        actions, info = super().get_actions(observations)
        if not self.poison:
            return actions, info
        logger.log('generating action')
        actions = [self.action_space.sample() for _ in observations]
        return actions, info

if __name__ == "__main__":
#    main(load_checkpoint='data/saved_airl_lg.ckpt')
    main()
