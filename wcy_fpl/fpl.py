#!/usr/bin/env python
"""
Starting point for a Plato federated learning training session.
"""
import os
import sys
sys.path.append("..")
os.chdir(sys.path[0])
import fpl_client
import fpl_trainer
import fpl_server
from plato.servers import registry as server_registry
from plato.datasources import aigc_mnist

# os.environ["config_file"] = "fpl.yml"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Starting point for a Plato federated learning training session."""
    trainer = fpl_trainer.Trainer
    client = fpl_client.Client(trainer=trainer)
    server = fpl_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
