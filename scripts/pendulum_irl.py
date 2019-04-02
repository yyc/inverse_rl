import tensorflow as tf
import sys
import numpy as np

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
import rllab.misc.logger as logger

import pickle

from inverse_rl.algos.irl_trpo import IRLTRPO
# from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_experts


from inverse_rl.models.architectures import softplus_net
from inverse_rl.utils.poisoned_policy import PoisonedPolicy

def main(load_checkpoint=None, debug=False, fnames=None, job_id=None):
    env = TfEnv(GymEnv('Pendulum-v0', record_video=False, record_log=False))

    if fnames is None:
        ## Optimal demonstrations
        experts = load_latest_experts('data/pendulum', n=10)
    ## Noisy pendulum samples
    # experts = load_latest_experts('data/pendulum_poisoned', n=5)
    ## Negated pendulum samples
    # experts = load_latest_experts('data/pendulum_poisoned_2', n=5)
    else:
        experts = load_experts(fnames)

    if job_id == None:
        job_id = np.random.randint(10, 1000)

    num_itr = 400
    hidden_size = 32
    trajectories_subset = None
    if debug:
        num_itr = 5
        # Use fewer hidden layers so the hessian computation is faster in debugging
        hidden_size = 10
        trajectories_subset = [1,2,3]
    else:
        logger.set_snapshot_gap(100)
        logger.set_snapshot_mode("gap")

#    irl_model = AIRLStateAction(env_spec=env.spec, expert_trajs=experts)
    irl_model = AIRL(env=env, expert_trajs=experts, state_only=True,
                     reward_arch=softplus_net,
                     reward_arch_args={"d_hidden": hidden_size})

    saver = tf.train.Saver()
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(hidden_size, hidden_size))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=num_itr,
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


    with rllab_logdir(algo=algo, dirname='models/pendulum_airl_{}'.format(job_id)):
        with tf.Session() as sess:

            env = GymEnv('Pendulum-v0', record_video=False, record_log=False)
            if load_checkpoint is None:
                algo.train()
            else:
                saver.restore(sess, load_checkpoint)


            # good_traj_sample = [i for i in range(len(experts)) if not experts[i]['poisoned']][:5]
            # bad_traj_sample = [i for i in range(len(experts)) if experts[i]['poisoned']][:5]
            # traj_samples = good_traj_sample + bad_traj_sample
            # influences = algo.calc_influence(t_size=5, trajectories=traj_samples)


            try:
                itr = num_itr
                influences = []
                # for i in range(10):
                #     influences.append(algo.calc_influence(t_size=40, compute_hessian=True, trajectories=trajectories_subset))
                #     itr = itr + 1
                # for i in range(5):
                #     influences.append(algo.calc_influence(t_size=80, compute_hessian=True, trajectories=trajectories_subset))
                #     itr = itr + 1
                for i in range(10):
                    influences.append(algo.calc_influence(itr=itr,
                                                          ztest_size=200, t_size=200,
                                                          compute_hessian=True,
                                                          trajectories=trajectories_subset,
                                                          individual_sa_pairs=True, mode="q_fn"))
                    itr = itr + 1
            except Exception as e:
                print(str(e))
                if debug:
                    raise e
            finally:
                save_file = 'results/airl_infl_{}.pkl'.format(job_id)
                pickle.dump(influences, open(save_file, 'wb'))
                print('saved into %s' %(save_file))

            try:
                while (True):
                    print("reset env")
                    obs = env.reset()
                    for _ in range(100):
                        action, _ = policy.get_action(obs)
                        obs, _, _, _ = env.step(action)
                        env.render()
            except Exception:
                pass
            finally:
                print("exiting simulation")


if __name__ == "__main__":
    args = sys.argv
    debug = False
    mode = "optimal"
    fnames = list(range(391, 400))
    id = "test"
    if len(args) > 1:
        debug = True if args[1] == "debug" else False
    if len(args) > 2:
        mode = args[2]
        if mode == "optimal":
            fnames.append(400)
        if mode == "noisy":
            fnames.append(401)
        if mode == "neg":
            fnames.append(402)
        print("Using mode {}".format(mode))
    if len(args) > 3:
        id = args[3]

    fnames = ["data/pendulum_poisoned_flex/itr_{}.pkl".format(itr) for itr in fnames]

    main(debug=False, fnames=fnames, job_id=id)