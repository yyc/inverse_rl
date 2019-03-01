import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net
from inverse_rl.utils import TrainingIterator, interleave_lists as interleave
from inverse_rl.utils.infl_utils import *
from tensorflow.python.ops.gradients_impl import _hessian_vector_product as hvp

import time
import pickle

class AIRL(SingleTimestepIRL):
    """ 
    State-only rewards AIRL

    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env,
                 expert_trajs=None,
                 reward_arch=relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=100,
                 fusion=False,
                 name='airl'):
        super(AIRL, self).__init__()
        env_spec = env.spec
        if reward_arch_args is None:
            reward_arch_args = {}

        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only = state_only
        self.max_itrs = max_itrs
        self.hessian_op = None
        self.l_ztest = None
        self.traj_losses = None

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=1)
                with tf.variable_scope('reward') as rvs:
                    # Get reward weights for gradient computation later
                    self.reward = reward_arch(rew_input, individual_vars=True, dout=1, **reward_arch_args)
                    self.reward_weights = tf.get_collection('indiv_vars', scope=rvs.name)

                # value function shaping
                with tf.variable_scope('vfn'):
                    fitted_value_fn_n = value_fn_arch(self.nobs_t, dout=1)
                with tf.variable_scope('vfn', reuse=True):
                    self.value_fn = fitted_value_fn = value_fn_arch(self.obs_t, dout=1)

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma*fitted_value_fn_n
                log_p_tau = self.reward  + self.gamma*fitted_value_fn_n - fitted_value_fn

            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau-log_pq)

            # Use the full loss so we can operate separately on different parts
            self.full_loss = -1 * (self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))
            cent_loss = tf.reduce_mean(self.full_loss)

            self.loss = cent_loss
            tot_loss = self.loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss)
            self._make_param_ops(_vs)


    def calc_influence(self, paths, policy, ztest_size=25, t_size=50, logger=None,
                       hessian_iter=10, trajectories=None, compute_hessian=True):
        # modified from the fit() function
        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths+old_paths

        start_time = time.time()
        last_time = start_time


        def log_diff(s):
            nonlocal last_time
            logger.log(
                '%s | t=%f (+%f)' % (s, time.time() - start_time, time.time() - last_time))
            last_time = time.time()

        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        # eval expert log probs under current policy
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(paths)
        self._insert_next_state(self.expert_trajs)
        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))


        #sample ztests and t zs each for hessian
        nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
            self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=ztest_size + t_size)

        # sample ztests and t zs for hessian approximation
        nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
            self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=ztest_size + t_size)

        # if specify a subset of trajectories, calculate influences only for those trajectories
        expert_trajs = self.expert_trajs

        if trajectories is not None:
            expert_trajs = np.take(self.expert_trajs, trajectories)
            expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
                self.extract_paths(expert_trajs,
                                   keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))

        # Build array for full_loss that contains z_test, z_s, and all expert trajs
        # Build feed dict
        labels = np.concatenate(
            [interleave(np.zeros((ztest_size + t_size, 1)), np.ones((ztest_size + t_size, 1))), np.ones((len(expert_obs), 1))]
        )
        obs_batch = np.concatenate([interleave(obs_batch, expert_obs_batch), expert_obs], axis=0)
        nobs_batch = np.concatenate([interleave(nobs_batch, nexpert_obs_batch), expert_obs_next], axis=0)
        act_batch = np.concatenate([interleave(act_batch, expert_act_batch), expert_acts], axis=0)
        nact_batch = np.concatenate([interleave(nact_batch, nexpert_act_batch), expert_acts_next], axis=0)
        lprobs_batch = np.expand_dims(np.concatenate([interleave(lprobs_batch, expert_lprobs_batch), expert_probs], axis=0), axis=1).astype(np.float32)
        feed_dict = {
            self.act_t: act_batch,
            self.obs_t: obs_batch,
            self.nobs_t: nobs_batch,
            self.nact_t: nact_batch,
            self.labels: labels,
            self.lprobs: lprobs_batch,
            }

        # split lengths for individual expert trajectories
        splits_lengths = [len(t['observations']) for t in expert_trajs]

        ztests, zs, *traj_sa_losses = tf.split(self.full_loss, [ztest_size*2, 2*t_size] + splits_lengths)

        log_diff('Building Ztest Gradient Op')

        # z_test losses, automatically summed, divide to get mean
        if self.l_ztest is None:
            self.l_ztest = tf.math.divide(tf.convert_to_tensor(
                tf.gradients(ztests, self.reward_weights, stop_gradients=self.reward_weights)
            ), ztest_size * 2)

        log_diff('Building trajectories gradient op')

        # Trajectory losses computed by averaging the individual state-action losses in the trajectory
        # Divide to get mean
        if self.traj_losses is None:
            self.traj_losses = \
                [tf.math.divide(tf.gradients(traj, self.reward_weights, stop_gradients=self.reward_weights), tf.size(traj))
                 for traj in traj_sa_losses]

        log_diff('Running ztest and trajectory ops')
        # Calculate the z_test losses (s_test in the Fu paper), and the trajectory losses
        diff_lz, traj_losses = tf.get_default_session().run([self.l_ztest, self.traj_losses], feed_dict=feed_dict)


        # Calculate the full hessian and invert it
        if compute_hessian:
            hessian_val = None
            hessian_inv = None
            try:
                log_diff('Building Hessian Op')
                if self.hessian_op is None:
                    self.hessian_op = hessian(zs, self.reward_weights)

                log_diff('Running Hessian Op')
                hessian_val = tf.get_default_session().run(self.hessian_op, feed_dict=feed_dict)

                log_diff('Building Inverse Op')
                hessian_inv_op = tf.linalg.inv(hessian_val)

                log_diff("Running Inverse Op")
                hessian_inv = tf.get_default_session().run(hessian_inv_op)

                log_diff("Multiplying to calculate influences")
                hess_inv_vs = [np.matmul(hessian_inv, v) for v in traj_losses]
                influences = [ -1 * np.dot(diff_lz, hinvv) for hinvv in hess_inv_vs]
            except Exception as e:
                logger.log(str(e))
                # Capture quits as well as tf errors
                filename = 'error_hessian.pkl'
                log_diff("Error Encountered, dumping to %s " % filename)
                pickle.dump((diff_lz, traj_losses, hessian_val, hessian_inv), open(filename, 'wb'))
                raise e
            return expert_trajs, (diff_lz, traj_losses, hessian_val, hessian_inv), influences

        # hessian_compute_inverse_vs also outputs the estimates at each iteration for me to track convergence
        # takes in the list of vectors to hopefully do it in parallel
        conv, hess_inv_vs = hessian_compute_inverse_vs(zs, self.reward_weights, traj_losses, feed_dict=feed_dict)

        # Calculate influence function
        influences = [-1 * np.dot(diff_lz, hess_inv) for hess_inv in hess_inv_vs]

        # Output the trajectories as well so we can check them manually
        return expert_trajs, conv, influences


    def fit(self, paths, policy=None, batch_size=32, logger=None, lr=1e-3,**kwargs):

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths+old_paths

        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        # eval expert log probs under current policy
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(paths)
        self._insert_next_state(self.expert_trajs)
        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))


        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=batch_size)

            # Build feed dict
            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)
            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr
                }

            loss, _ = tf.get_default_session().run([self.loss, self.step], feed_dict=feed_dict)
            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:
            logger.record_tabular('GCLDiscrimLoss', mean_loss)
            #obs_next = np.r_[obs_next, np.expand_dims(obs_next[-1], axis=0)]
            energy, logZ, dtau = tf.get_default_session().run([self.reward, self.value_fn, self.discrim_output],
                                                               feed_dict={self.act_t: acts, self.obs_t: obs, self.nobs_t: obs_next,
                                                                   self.nact_t: acts_next,
                                                               self.lprobs: np.expand_dims(path_probs, axis=1)})
            energy = -energy
            logger.record_tabular('GCLLogZ', np.mean(logZ))
            logger.record_tabular('GCLAverageEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageLogPtau', np.mean(-energy-logZ))
            logger.record_tabular('GCLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('GCLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('GCLAverageDtau', np.mean(dtau))


            #expert_obs_next = np.r_[expert_obs_next, np.expand_dims(expert_obs_next[-1], axis=0)]
            energy, logZ, dtau = tf.get_default_session().run([self.reward, self.value_fn, self.discrim_output],
                    feed_dict={self.act_t: expert_acts, self.obs_t: expert_obs, self.nobs_t: expert_obs_next,
                                    self.nact_t: expert_acts_next,
                                    self.lprobs: np.expand_dims(expert_probs, axis=1)})
            energy = -energy
            logger.record_tabular('GCLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy-logZ))
            logger.record_tabular('GCLAverageExpertLogQtau', np.mean(expert_probs))
            logger.record_tabular('GCLMedianExpertLogQtau', np.median(expert_probs))
            logger.record_tabular('GCLAverageExpertDtau', np.mean(dtau))
        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            self._compute_path_probs(paths, insert=True)
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=('observations', 'observations_next', 'actions', 'a_logprobs'))
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(self.discrim_output,
                                              feed_dict={self.act_t: acts, self.obs_t: obs,
                                                         self.nobs_t: obs_next,
                                                         self.lprobs: path_probs})
            score = np.log(scores) - np.log(1-scores)
            score = score[:,0]
        else:
            obs, acts = self.extract_paths(paths)
            reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.act_t: acts, self.obs_t: obs})
            score = reward[:,0]
        return self.unpack(score, paths)

    def eval_single(self, obs):
        reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.obs_t: obs})
        score = reward[:, 0]
        return score

    def debug_eval(self, paths, **kwargs):
        obs, acts = self.extract_paths(paths)
        reward, v, qfn = tf.get_default_session().run([self.reward, self.value_fn,
                                                            self.qfn],
                                                      feed_dict={self.act_t: acts, self.obs_t: obs})
        return {
            'reward': reward,
            'value': v,
            'qfn': qfn,
        }

