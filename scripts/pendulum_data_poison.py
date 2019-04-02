from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
import rllab.misc.logger as logger
import tensorflow as tf
from rllab.misc.overrides import overrides

from inverse_rl.utils.poisoned_policy import PoisonedPolicy
from inverse_rl.utils.log_utils import rllab_logdir, load_experts

def main():
    env = TfEnv(GymEnv('Pendulum-v0', record_video=False, record_log=False))
    policy = PoisonedPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

    num_itr = 400
    # num_itr = 1

    # experts = load_experts(['data/pendulum_poisoned/itr_1.pkl'])
    # print(experts)

    algo = TRPO(
        env=env,
        policy=policy,
        n_itr=num_itr,
        batch_size=1000,
        max_path_length=100,
        discount=0.99,
        store_paths=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/pendulum_poisoned_flex'):
        sess = tf.Session()
        sess.__enter__()
        algo.train(sess=sess)

        logger.log("getting extra optimal trajectory")
        paths = algo.obtain_samples(num_itr)
        samples = algo.process_samples(num_itr, paths)
        samples['poisoned'] = False
        logger.save_itr_params(num_itr, samples) #400


        logger.log("getting random poisoned trajectory")
        policy.poison = True
        paths = algo.obtain_samples(num_itr+1)
        samples = algo.process_samples(num_itr+1, paths)
        samples['poisoned'] = True
        logger.save_itr_params(num_itr+1, samples) #401

        logger.log("getting negated poisoned trajectory")
        policy.mode = "neg"
        paths = algo.obtain_samples(num_itr + 2)
        samples = algo.process_samples(num_itr+2, paths)
        samples['poisoned'] = True
        logger.save_itr_params(num_itr + 2, samples) #402

if __name__ == "__main__":
    main()
