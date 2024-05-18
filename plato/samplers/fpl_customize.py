"""
Samples data from a dataset, biased across labels according to the Dirichlet distribution.
"""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
from plato.config import Config

from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client according to given distribution."""

    def __init__(self, datasource, client_id, target_proportions,sampled_size, testing ):
        super().__init__()

        if testing:
            target_list = datasource.get_test_set().targets
        else:
            # The list of labels (targets) for all the examples
            target_list = datasource.targets()

        class_list = datasource.classes()
        target_proportions = np.array(target_proportions)
        if np.isnan(np.sum(target_proportions)):
            target_proportions = np.repeat(0, len(class_list))
            target_proportions[np.random.randint(0, len(class_list))] = 1
        self.sampled_size = sampled_size

        self.sample_weights = target_proportions[target_list]

    def num_samples(self) -> int:
        """Returns the length of the dataset after sampling."""
        # sampled_size = Config().data.partition_size

        return self.sampled_size

    def get(self):
        """Obtains an instance of the sampler."""
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        subset_indices = list(
            WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=self.sampled_size,
                replacement=False,
                generator=gen,
            )
        )

        return SubsetRandomSampler(subset_indices, generator=gen)
    

