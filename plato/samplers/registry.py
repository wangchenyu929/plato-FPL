"""
The registry for samplers designed to partition the dataset across the clients.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""
import logging
from collections import OrderedDict
import numpy as np

from plato.config import Config

if hasattr(Config().trainer, "use_mindspore"):
    from plato.samplers.mindspore import (
        iid as iid_mindspore,
        dirichlet as dirichlet_mindspore,
    )

    registered_samplers = OrderedDict(
        [
            ("iid", iid_mindspore.Sampler),
            ("noniid", dirichlet_mindspore.Sampler),
        ]
    )
elif hasattr(Config().trainer, "use_tensorflow"):
    from plato.samplers.tensorflow import base

    registered_samplers = OrderedDict(
        [
            ("iid", base.Sampler),
            ("noniid", base.Sampler),
            ("mixed", base.Sampler),
        ]
    )
else:
    from plato.samplers import (
        iid,
        dirichlet,
        mixed,
        orthogonal,
        all_inclusive,
        distribution_noniid,
        label_quantity_noniid,
        mixed_label_quantity_noniid,
        sample_quantity_noniid,
        modality_iid,
        modality_quantity_noniid,
        fpl_distribution,
        fpl_aigc_distribution,
        fpl_customize,

    )

    registered_samplers = OrderedDict(
        [
            ("iid", iid.Sampler),
            ("noniid", dirichlet.Sampler),
            ("mixed", mixed.Sampler),
            ("orthogonal", orthogonal.Sampler),
            ("all_inclusive", all_inclusive.Sampler),
            ("distribution_noniid", distribution_noniid.Sampler),
            ("label_quantity_noniid", label_quantity_noniid.Sampler),
            ("mixed_label_quantity_noniid", mixed_label_quantity_noniid.Sampler),
            ("sample_quantity_noniid", sample_quantity_noniid.Sampler),
            ("modality_iid", modality_iid.Sampler),
            ("modality_quantity_noniid", modality_quantity_noniid.Sampler),
            ("fpl_distribution", fpl_distribution.Sampler),
            ("fpl_aigc_distribution", fpl_aigc_distribution.Sampler),
            ("fpl_customize", fpl_customize.Sampler)
        ]
    )

# get normal fl datasource
def get(datasource, client_id, testing=False, **kwargs):
    """Get an instance of the sampler."""

    sampler_type = (
        kwargs["sampler_type"]
        if "sampler_type" in kwargs
        else Config().data.testset_sampler
        if testing and hasattr(Config().data, "testset_sampler")
        else Config().data.sampler
    )
    if testing:
        logging.info("[Client #%d] Test set sampler: %s", client_id, sampler_type)
    else:
        logging.info("[Client #%d] Sampler: %s", client_id, sampler_type)
        
    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](
            datasource, client_id, testing=testing
        )
    else:
        raise ValueError(f"No such sampler: {sampler_type}")

    return registered_sampler

# get aigc datasource
def aigc_get(aigc_datasource, client_id, local_distribution):
    """Get an instance of the sampler."""

    logging.info("[Client #%d] Sampler: %s", client_id, "aigc_fpl_distribution")
        

    registered_sampler = registered_samplers["aigc_fpl_distribution"](
        aigc_datasource, client_id, local_distribution)

    return registered_sampler

def fpl_customize_get(datasource, client_id, target_proportions, sampled_size,testing=False):
    logging.info("[Client #%d] Sampler: %s", client_id, "fpl_customize")
    sampler = fpl_customize.Sampler(datasource,client_id,target_proportions,sampled_size,testing=testing)

    return sampler