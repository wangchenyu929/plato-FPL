"""
FPL: a ... algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""
import logging
import time
import pickle
import sys
from types import SimpleNamespace

from plato.utils import fonts
from plato.clients import simple
from plato.config import Config
from plato.trainers import registry as trainers_registry
from plato.algorithms import registry as algorithms_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.datasources import registry as datasources_registry

class Client(simple.Client):

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        # add three new attributes
        self.aigc_datasource = None
        self.aigc_trainset = None
        self.aigc_sampler = None
        # the proporation of local data distribution
        self.dirichlet_propotion = []
        # the proporation of added aigc data
        self.target_propotion = []
        # the expxted number of added aigc data
        self.exp_sample_num  = 0


    def configure(self) -> None:
        """Prepares this client for training."""
        # super().configure()

        if self.model is None and self.custom_model is not None:
            self.model = self.custom_model

        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(
                model=self.model, callbacks=self.trainer_callbacks
            )
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(
                model=self.model, callbacks=self.trainer_callbacks
            )

        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)

        self.algorithm.set_client_id(self.client_id)

        # Pass inbound and outbound data payloads through processors for
        # additional data processing
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer
        )

        # Setting up the data sampler

        ## add aigc sampler
        if self.datasource:
            self.sampler = samplers_registry.fpl_customize_get(self.datasource, self.client_id, self.dirichlet_propotion,Config().data.partition_size)
            if self.exp_sample_num != 0:
                self.aigc_sampler=samplers_registry.fpl_customize_get(self.aigc_datasource,self.client_id, self.target_propotion,self.exp_sample_num)

    def _load_data(self) -> None:
        """Generates local and aigc data and loads them and load them onto this client."""
        # The only case where Config().data.reload_data is set to true is
        # when clients with different client IDs need to load from different datasets,
        # such as in the pre-partitioned Federated EMNIST dataset. We do not support
        # reloading data from a custom datasource at this time.
        if (
            self.datasource is None
            or hasattr(Config().data, "reload_data")
            and Config().data.reload_data
        ):
            logging.info("[%s] Loading its local and aigc data source...", self)


            self.datasource = datasources_registry.get(client_id=self.client_id)
            self.aigc_datasource = datasources_registry.get(client_id=self.client_id, datasource_name = Config().data.aigc_datasource)


            logging.info(
                "[%s] Dataset size: %s", self, self.datasource.num_train_examples()
            )

    def _allocate_data(self) -> None:
        """Allocate training or testing dataset of this client."""

        # PyTorch uses samplers when loading data with a data loader
        self.trainset = self.datasource.get_train_set()
        ## the way to get aigc_trainset
        self.aigc_trainset = self.aigc_datasource.get_train_set()

    
    async def _train(self):
        """The machine learning training workload on a client."""
        logging.info(
            fonts.colourize(
                f"[{self}] Started training in communication round #{self.current_round}."
            )
        )
        # Perform model training
        try:
            if hasattr(self.trainer, "current_round"):
                self.trainer.current_round = self.current_round
            ## add two args aigc_trainset and aigc_sampler
            training_time = self.trainer.train(self.trainset, self.aigc_trainset, self.sampler,self.aigc_sampler)

        except ValueError as exc:
            logging.info(
                fonts.colourize(f"[{self}] Error occurred during training: {exc}")
            )
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        ## delete testing process
        accuracy = 0

        comm_time = time.time()

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time

            training_time = (
                avg_training_time + sleep_seconds
            ) * Config().trainer.epochs

        report = SimpleNamespace(
            client_id=self.client_id,
            num_samples=self.sampler.num_samples()+self.exp_sample_num,
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
        )

        self._report = self.customize_report(report)

        return self._report, weights
    

    async def _payload_to_arrive(self, response) -> None:
        """Upon receiving a response from the server."""

        self.current_round = response["current_round"]

        # Update (virtual) client id for client, trainer and algorithm
        self.client_id = response["id"]

        logging.info("[Client #%d] Selected by the server.", self.client_id)
        self.dirichlet_propotion = response["dirichlet_propotion"]
        self.target_propotion = response["target_propotion"]
        self.exp_sample_num = response["exp_sample_num"]

        self.process_server_response(response)

        self._load_data()
        self.configure()
        self._allocate_data()

        self.server_payload = None

        if self.comm_simulation:
            payload_filename = response["payload_filename"]
            with open(payload_filename, "rb") as payload_file:
                self.server_payload = pickle.load(payload_file)

            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

            logging.info(
                "[%s] Received %.2f MB of payload data from the server (simulated).",
                self,
                payload_size / 1024**2,
            )

            await self._handle_payload(self.server_payload)