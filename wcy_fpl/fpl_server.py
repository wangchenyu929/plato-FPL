import logging
import math
import random
import copy
import numpy as np
import os,sys
os.chdir(sys.path[0])
from collections import Counter
from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A federated learning server using oort client selection."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # information of local data and aigc data of clients

        # calculated by self.init_list() at the beggining of entire training process
        # local normal data propotion
        self.dirichlet_propotion = {}
        # aigc expected data propotion 
        self.target_propotion = {}
        # how much more aigc data do server need to add
        self.exp_sample_num = {}
        # inited by dirichlet_propotion at first
        # and updated in every round begining
        self.real_data_propotion = {}
        # the preference classes id of server
        self.pc_label = [int(x.strip()) for x in Config().data.server_PC.split(",")]
        self.aigc_add_number = {}
        

        # information about client selection
        # All clients' utilities
        self.client_utilities = {}
        # used for calculate client_util
        # self.exp_pc_propotion = {}
        # All clients‘ training times
        # inited by Config().clients.training_time
        self.client_durations = {}
        # inited by clients_pool
        # updated by cut_off*sortByAsc(data_util)

        self.sup_prioriry = []
        self.candidate_clients = []

        self.client_selected_times = {}

        self.elapsed_time = 0
        self.round_time = 0

        self.black_list = []
        
    def configure(self) -> None:
        """Initialize necessary variables."""
        super().configure()

        self.dirichlet_propotion, self.target_propotion, self.exp_sample_num = self.init_list()
    

        self.real_data_propotion = copy.deepcopy(self.dirichlet_propotion)

        self.aigc_add_number = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }

        self.client_selected_times = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }

        congfig_client_duration = [int(x.strip()) for x in Config().clients.training_time.split(",")]
        self.client_durations = {
            client_id: int(duration) for (client_id,duration) in enumerate(congfig_client_duration,1)
        }
        

        self.candidate_clients = copy.deepcopy(self.clients_pool)

    def init_list(self):
        
        PC_exp = Config().data.server_exp
        partition_size = Config().data.partition_size
        random_seed = Config().data.random_seed
        
        dirichlet_list = {}
        exp_sample_num = {}
        exp_pc_prob = {}
        local_pc_distribution = {}
        # gap_weight = []
        target_proportions = {client:[] for client in range(1,self.total_clients+1)}
        local_pc_distribution = {client:np.zeros(10) for client in range(1,self.total_clients+1)}
        # local data proportion
        concentration = (
            Config().data.concentration
            if hasattr(Config().data, "concentration")
            else 1.0
        )
        for i in range(1,self.total_clients+1):
            np.random.seed(random_seed * i)
            dirichlet_list[i]=np.random.dirichlet(np.repeat(concentration, 10))
            # there are 10 classes in datasource
            exp_pc_prob[i]=np.zeros(10)

        # local data proportion
        for i in range(1, self.total_clients+1):

            for j in self.pc_label:
                local_pc_distribution[i][j] = dirichlet_list[i][j]
                exp_pc_prob[i][j]=PC_exp
                    
            # stage 1
            PC_add_number = {classes:0 for classes in range(10)}
            max_pc_proportion = max(local_pc_distribution[i])
            if max_pc_proportion > PC_exp:
                for j in self.pc_label:
                    PC_add_number[j] = (max_pc_proportion-local_pc_distribution[i][j])*partition_size
                exp_sample_num[i] = sum(PC_add_number.values())
            # stage 2
            else:    
                a = (exp_pc_prob[i].sum()-local_pc_distribution[i].sum())*partition_size
                b = 1-exp_pc_prob[i].sum()
                rep_quantity = int(a/b)
                exp_sample_num[i] = rep_quantity
                for j in self.pc_label:
                    PC_add_number[j] = math.ceil(((rep_quantity + partition_size)*PC_exp-local_pc_distribution[i][j]*partition_size))
            target_proportions[i] = [PC_add_number[j]/sum(PC_add_number.values()) for j in range(10)]

        print("total samples",sum(exp_sample_num.values()))
        return dirichlet_list,target_proportions,exp_sample_num
    
    def customize_server_response(self, server_response: dict, client_id) -> dict:

        """Customizes the server response with any additional information."""
        server_response["dirichlet_propotion"] = self.dirichlet_propotion[client_id].tolist()
        server_response["target_propotion"] = self.target_propotion[client_id]
        server_response["exp_sample_num"] = self.aigc_add_number[client_id]
        # server_response["exp_sample_num"] = 200
        # print("server_response:",server_response)

        return server_response
    
    # calculate the aigc total add number of every client
    def cal_aigc_add_number(self):
        # 需要在每次客户端选择之前调用，随response一起发给client让其生成对应的aigc data
        # 根据sup_prioriry和上一轮的candidate clients来计算
        aigc_add_number = {client_id: 0 for client_id in range(1, self.total_clients + 1)}

        # If current round is 1, sup_prioriry is the client's computing power in descending order.
        # The larger the client_durations, the weaker the computing power.

        remaining_aigc_number = Config().data.aigc_speed
        for i in self.sup_prioriry:
            if remaining_aigc_number == 0:
                break
            if self.exp_sample_num[i] == 0:
                continue
            if self.exp_sample_num[i] >= remaining_aigc_number:
                aigc_add_number[i] = remaining_aigc_number
                self.exp_sample_num[i] = self.exp_sample_num[i]-remaining_aigc_number
                remaining_aigc_number = 0
                
            else:
                 aigc_add_number[i] = self.exp_sample_num[i]
                 remaining_aigc_number = remaining_aigc_number-self.exp_sample_num[i]
                 self.exp_sample_num[i] = 0
            # updata real_data_propotion
            local_data_number = [propotion*Config().data.partition_size for propotion in self.dirichlet_propotion[i]]
            aigc_data_number = [propotion*aigc_add_number[i] for propotion in self.target_propotion[i]]
            total_data_number = np.sum([local_data_number,aigc_data_number],axis=0)
            updata_data_propotion = [data_number/total_data_number.sum() for data_number in total_data_number]
            self.real_data_propotion[i] = updata_data_propotion
            self.aigc_add_number[i] += math.ceil(aigc_add_number[i])

            # first = self.sup_prioriry.pop(0)
            # self.sup_prioriry.append(first)
        
        # print("real_data_propotion",self.real_data_propotion)
        print("aigc_add_number",self.aigc_add_number)
        print("sup_prioriry",self.sup_prioriry)
        print("exp_sample_num",self.exp_sample_num)

    def calc_client_util(self):

        # normalization
        # smmaller is better
        sys_util = {client_id:util/sum(self.client_durations.values()) 
                    for (client_id, util) in enumerate(self.client_durations.values(),1)}
        # the gap between real data distribution and expcted data distribution
        data_util = {}
        for client in range(1, self.total_clients + 1):
        # for propotion in self.real_data_propotion.values():
            util = 0
            for pc in self.pc_label:
                if self.real_data_propotion[client][pc] < Config().data.server_exp:
                    util += Config().data.server_exp - self.real_data_propotion[client][pc]
            data_util[client] = util

        # normalization
        # smmaller is better
        data_util = {client_id:util/sum(data_util.values())
                     for (client_id, util) in enumerate(data_util.values(),1)}
        
        # smaller is better
        client_utilities = dict(Counter(sys_util)+Counter(data_util))

        # select candidate clients according to data util and cut off factor
        self.candidate_clients = sorted(
                data_util, key=data_util.get, reverse=False
            )
        # for i in self.black_list:
        #     self.candidate_clients.remove(i)
        
        candidates_number = math.ceil(self.total_clients*Config().server.cut_off)
        
        self.candidate_clients = self.candidate_clients[0:candidates_number]



        return  client_utilities
    
    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of the clients to participate in each round."""
        # to finish the add aigc data operation by updating self.aigc_add_number 
        if self.current_round > 3:
            self.cal_aigc_add_number()
        # print("self.aigc_add_number",self.aigc_add_number)
        self.sup_prioriry = sorted(self.client_durations, key=self.client_durations.get, reverse=False)
        # for i in self.candidate_clients:
        #     self.sup_prioriry.remove(i)
        
        self.client_utilities = self.calc_client_util()

        assert clients_count <= len(self.candidate_clients)
        random.setstate(self.prng_state)
        # Because it needs to become the selection probability, we take the reciprocal
        for client_id in range(1,self.total_clients+1):   
            self.client_utilities[client_id] = 1/self.client_utilities[client_id]
        # self.client_utilities = {client_id:1/util for (client_id,util) in enumerate(self.client_utilities,1)}
        total_utility = float(
                sum(self.client_utilities[client_id] for client_id in self.candidate_clients)
            )
        probabilities = [
                self.client_utilities[client_id] / total_utility
                for client_id in self.candidate_clients
            ]
        # Select clients randomly
        selected_clients = np.random.choice(self.candidate_clients, clients_count, p=probabilities,replace=False)

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        
        for client in selected_clients:
            self.client_selected_times[client] += 1
            if self.client_selected_times[client]==5:
                self.black_list.append(client)

        logging.info("[%s] Client_selected_times: %s", self, self.client_selected_times)
        self.round_time = 0
        for client in selected_clients: 
            if self.client_durations[client] > self.round_time:
                self.round_time = self.client_durations[client]
        self.elapsed_time += self.round_time
        return selected_clients

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""

        return {
            "round": self.current_round,
            "accuracy": self.accuracy,
            "elapsed_time": self.round_time,
            "round_time": self.elapsed_time,
        }
    


        



    


    
