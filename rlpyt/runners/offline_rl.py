import os
import psutil
import time
import torch
import math
from collections import deque

from rlpyt.runners.minibatch_rl import MinibatchRlBase
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
from wandb import agent

class OfflineRl(MinibatchRlBase):

    _eval = True

    def __init__(
            self,
            algo,
            agent,
            sampler,
            eval_sampler,
            seed=None,
            affinity=None,
            log_interval_steps=1e5,
            model_train_steps=1e5,
            agent_train_steps=1e5,
            samples_load_path=None,
            samples_save_dir=None,
            model_save_dir=None
            ):
        super().__init__(algo, agent, sampler, n_steps, seed=seed, affinity=affinity, log_interval_steps=log_interval_steps)
        self.model_train_steps = model_train_steps
        self.agent_train_steps = agent_train_steps
        self.samples_load_path = samples_load_path
        self.samples_save_dir = samples_save_dir
        self.model_save_dir = model_save_dir
        self.time = time.time()
        self.eval_sampler = eval_sampler

    def train(self):
        n_itr = self.startup()

        examples = self.eval_sampler.initialize(
            agent=self.agent,
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=self.rank,
            world_size=self.world_size,
        )

        with logger.prefix(f"itr #0 "):
            itr = 0
            eval_samples, traj_info = self.evaluate_agent(0)
            self.store_diagnostics(0, eval_samples, traj_info)
            self.log_diagnostics(0)
            self.agent.sample_mode(itr)
            if self.samples_load_path == None:
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.algo.add_samples(samples)
            else:
                samples = torch.load(self.samples_load_path)

            if self.samples_save_dir:
                torch.save(samples, os.path.join(self.samples_save_dir, "{}_samples.pt".format(self.time)))


        self.agent.train_mode(itr)
        for itr in range(self.model_train_steps, step=self.algo.train_steps):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
#                opt_info = self.algo.optimize_agent(itr, samples)
                opt_info = self.algo.optimize_model(itr)
                self.store_diagnostics(itr, traj_infos, opt_info)
                self.log_diagnostics(itr)
                if self.model_save_dir:
                    torch.save(self.agent, os.path.join(self.model_save_dir, "{}_itr{}_samples.pt".format(self.time, itr)))

        for itr in range(self.model_train_steps, self.model_train_steps + self.agent_train_steps, step=self.algo.train_steps):
                opt_info = self.algo.optimize_policy(itr)
                eval_samples, traj_info = self.evaluate_agent(0)
                self.store_diagnostics(itr, traj_infos, opt_info)
                self.log_diagnostics(itr)
                self.algo.generate_videos(eval_samples, examples, self.eval_sampler.batch_spec, itr)
                if self.model_save_dir:
                    torch.save(self.agent, os.path.join(self.model_save_dir, "{}_begin_policy_itr{}_samples.pt".format(self.time, itr)))

#        for itr in range(self.environment_steps + self.model_train_steps, self.environment_steps + self.model_train_steps + self.agent_train_steps):
#            logger.set_iteration(itr)
#            with logger.prefix(f"itr #{itr} "):

        self.shutdown()

    def evaluate_agent(self, itr):
        """
        Record offline evaluation of agent performance, by ``sampler.evaluate_agent()``.
        """
        if itr > 0:
            self.pbar.stop()

        if itr >= self.min_itr_learn - 1 or itr == 0:
            logger.log("Evaluating agent...")
            self.agent.eval_mode(itr)  # Might be agent in sampler.
            eval_time = -time.time()
            samples, traj_info = self.sampler.obtain_samples(itr)
            eval_time += time.time()
        else:
            traj_infos = []
            eval_time = 0.0
        print("Eval time: {}".format(eval_time))
        logger.log("Evaluation runs complete.")
        return samples, traj_info 

    def initialize_logging(self):
        super().initialize_logging()
        self._cum_eval_time = 0

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._new_completed_trajs += len(traj_infos)
        self._traj_infos.extend(traj_infos)
        super().store_diagnostics(itr, traj_infos, opt_info)

    def log_diagnostics(self, itr, prefix='Diagnostics/'):
        with logger.tabular_prefix(prefix):
            logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
            logger.record_tabular('StepsInTrajWindow',
                sum(info["Length"] for info in self._traj_infos))
        super().log_diagnostics(itr, prefix=prefix)
        self._new_completed_trajs = 0