import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler,WeightedRandomSampler

from plato.config import Config
from plato.samplers import base

from plato.samplers import sampler_utils


class Sampler(base.Sampler):
     
    def __init__(self, datasource, client_id, testing):
        super().__init__()
        # the client_id of server is 0
        # Different clients should share the randomness
        #  as the assignment of classes is completed in each
        #  sampling process.
        # Thus, they share the clients_dataidx_map
        self.random_seed = Config().data.random_seed*int(client_id)
        # np.random.seed(self.random_seed*int(client_id))
        if testing:
            target_list = datasource.get_test_set().targets
            self.sampled_size = len(target_list)
        else:
            # The list of labels (targets) for all the examples
            target_list = datasource.targets()
            self.sampled_size = Config().data.partition_size

        class_list = datasource.classes()


        if Config().data.sampler == "fpl_distribution":
            class_exp = Config().data.fpl_expectation
            # the type of class_proportions should to be list 
            class_proportions = [float(x.strip()) for x in class_exp.split(",")]
        else:
            class_proportions = [
                1.0 / len(class_list) for i in range(len(class_list))
            ]
        
        target_proportions = np.array(class_proportions)
        # print("target_proportions",target_proportions)

        # if np.isnan(np.sum(target_proportions)):
        #     target_proportions = np.repeat(0, len(class_list))
        #     target_proportions[np.random.randint(0, len(class_list))] = 1

        self.sample_weights = target_proportions[target_list]
        # print("sample_weights",self.sample_weights)


    def get(self):
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        # Samples without replacement using the sample weights
        subset_indices = list(
            WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=self.num_samples(),
                replacement=False,
                generator=gen,
            )
        )

        return SubsetRandomSampler(subset_indices, generator=gen)


    def num_samples(self)-> int:
        
        # print("sampled_size",sampled_size)
        return self.sampled_size