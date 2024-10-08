import torch
from hdp import HierarchicalDirichletProcess
from dbm import DBM


import torch.multiprocessing as mp



if __name__ == "__main__":
    
    # mp.set_start_method('spawn')
    datasize = 1000
    data_dimension = 784
    latent_dimension = 10
    dbm_stable_param = 5
    latent_sample_size = 10
    batch_index = 1
    gibbs_iterations = 100

    # Test DBM
    dataset = torch.rand(datasize, data_dimension)
    comparison = torch.rand(datasize, data_dimension)
    
    dbm = DBM(data_dimension, [500, 500, latent_dimension], mode="bernoulli", k=dbm_stable_param)
    dbm.pre_train(dataset)
    dbm.train(dataset)
    latent_variables = dbm.generate_top_level_latent_variables(dataset, latent_sample_size)

    hp = HierarchicalDirichletProcess(latent_dimension, 3, datasize, 10, {2: 10})    
    hp.gibbs_update(batch_index, gibbs_iterations, latent_variables)
    latent_distribution = hp.get_latent_distributions()

    reconstructed_data = dbm.generate_visible_variables(latent_distribution, latent_sample_size)

    print("Reconstruction Difference: ", torch.mean(torch.abs(reconstructed_data.to(torch.device("cpu")) - dataset)))

    print("Comparison Difference: ", torch.mean(torch.abs(comparison - dataset)))
    

