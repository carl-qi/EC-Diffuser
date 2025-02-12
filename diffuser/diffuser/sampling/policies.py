from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn
import numpy as np
from diffuser.models import sample_fn_return_attn, default_sample_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GoalConditionedPolicy:
    def __init__(self, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True, return_attention=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        if len(conditions[0].shape) == 1:
            multi_input = False
        else:
            multi_input = True
        conditions = self._format_conditions(conditions, batch_size, multi_input=multi_input)
        
        if return_attention:
            samples, att_dict = self.diffusion_model(conditions, verbose=verbose, sort_by_value=False, return_attention=return_attention, **self.sample_kwargs)
            att_dict = {k: utils.to_np(v) for k, v in att_dict.items()}
        else:
            samples = self.diffusion_model(conditions, verbose=verbose, sort_by_value=False, return_attention=return_attention, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)


        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')    

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        ## extract first action per env if conditions contains multiple envs
        if not multi_input:
            action = actions[0, 0]
        else:
            action = actions[:, 0]


        trajectories = Trajectories(actions, observations, samples.values)
        if return_attention:
            return action, trajectories, att_dict
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size, multi_input=False):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32)
        if not multi_input:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                'd -> repeat d', repeat=batch_size,
            )
        return conditions