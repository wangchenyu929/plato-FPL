import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler,WeightedRandomSampler

from plato.config import Config
from plato.samplers import base

from plato.samplers import sampler_utils


class Sampler(base.Sampler):
     
    def __init__(self, aigc_datasource, client_id,  local_distribution):
        # datasource here is aigc dataset
        # local_distribution means local classes distribution of this client
        super().__init__()
        # aigc dataset is located at client only
        #  so there are no testing parameter here.

        # Different clients should share the randomness
        #  as the assignment of classes is completed in each
        #  sampling process.
        # Thus, they share the clients_dataidx_map

        # preference classes label of server
        self.pc_label = [int(x.strip()) for x in Config().data.server_PC.split(",")]
        # self.pc_label = [0]

        self.local_pc_prop = np.array(local_distribution)
        for i in range(0, len(self.local_pc_prop)):
            if i not in self.pc_label or self.local_pc_prop[i]>Config().data.server_exp:
                self.local_pc_prop[i]=0

        # preference classes proportion of server
        self.exp_pc_prob = np.zeros(10)
        for i in self.pc_label:
            if self.local_pc_prop[i]!=0:
                self.exp_pc_prob[i]=Config().data.server_exp

        a = (self.exp_pc_prob.sum()-self.local_pc_prop.sum())*Config().data.partition_size
        b = 1-self.exp_pc_prob.sum()
        rep_quantity = int(a/b)

        self.sampled_size = rep_quantity
        
        # The list of labels (targets) for all the examples
        # the type of class_proportions should to be list
        target_list = aigc_datasource.targets()
        class_list = aigc_datasource.classes()
        
        gap_weight = self.exp_pc_prob-self.local_pc_prop
        target_proportions = np.array([gap_weight[i]/gap_weight.sum() for i in range(len(gap_weight))])

        if np.isnan(np.sum(target_proportions)):
            target_proportions = np.repeat(0, len(class_list))
            target_proportions[np.random.randint(0, len(class_list))] = 1

        self.sample_weights = target_proportions[target_list]

        print("#####client: %d supplement number: %d", int(client_id), self.sampled_size)
        print("label distribution:",local_distribution)
        print("target proportions:",target_proportions)
 


    def get(self):
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        if self.sampled_size == 0:
            self.sampled_size = 1
        subset_indices = list(
            WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=self.sampled_size,
                replacement=False,
                generator=gen,
            )
        )

        return SubsetRandomSampler(subset_indices, generator=gen)


    def num_samples(self)-> int:
        
        # print("sampled_size",sampled_size)
        return self.sampled_size