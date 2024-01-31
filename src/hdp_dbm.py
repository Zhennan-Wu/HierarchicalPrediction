import torch
from hp import HP
from dbm import DBM


class HDP_DBM:
    """
    Hierarchical Dirichlet Prior Deep Boltzmann Machine
    """
    def __init__(self, pretrained_dbm):
        self.dbm = pretrained_dbm
        self.hp = HP()


    def train_HDP(self, top_level_latent_variables):
        """
        Train HDP
        """
        self.hp.hierarchical_dp()
    
    def fine_tune_DBM(self, dataset):
        """
        Fine tune DBM
        """
        self.dbm.train(dataset)
                                         
    def train(self, dataset):
        """
        Train HDP-DBM
        """
        y = self.hp.hierarchical_dp()

        y_gen = []
        for _ in range(self.dbm.k):
            y_dash = y.clone()
            for i in range(len(self.dbm.layer_parameters)-1, -1, -1):
                _, y_dash = self.dbm.sample_v(y_dash, self.dbm.layer_parameters[i]["W"])
            y_gen.append(y_dash)
        y_dash = torch.stack(y_gen)
        y_dash = torch.mean(y_dash, dim=0)
    

