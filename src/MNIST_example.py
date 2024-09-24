import torch
import numpy as np
import pandas as pd
import os
from hdp import HierarchicalDirichletProcess
from dbm import DBM
from load_dataset import MNIST
from torch.utils.data import DataLoader, TensorDataset


if __name__ == '__main__':
	mnist = MNIST()
	train_x, train_y, test_x, test_y = mnist.load_dataset()
	print('MAE for all 0 selection:', torch.mean(train_x))

	batch_size = 1000	

	train_x = train_x[:batch_size*3, :]
	train_y = train_y[:batch_size*3]

	datasize = train_x.shape[0]
	data_dimension = train_x.shape[1]
	print(datasize, data_dimension, batch_size)
	latent_dimension = 250
	dbm_stable_param = 5
	latent_sample_size = 10

	
	gibbs_iterations = 100
	hierarchy_levels = 3
	truncate_length = 10
	layer_constraints = {2: 10}


	dataset = TensorDataset(train_x, train_y)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	dbm = DBM(data_dimension, [500, 500, latent_dimension], mode="bernoulli", k=dbm_stable_param)

	for batch_data, batch_labels in data_loader:
		dbm.pre_train(batch_data)
		dbm.train(batch_data)
	
	batch_index = 1
	for batch_data, batch_labels in data_loader:
		latent_variables = dbm.generate_top_level_latent_variables(batch_data, latent_sample_size)
		hp = HierarchicalDirichletProcess(latent_dimension, hierarchy_levels, batch_size, truncate_length, layer_constraints)    
		hp.gibbs_update(batch_index, gibbs_iterations, latent_variables)
		hp.display_hierarchical_results(batch_labels)

		latent_distribution = hp.get_latent_distributions()

		reconstructed_data = dbm.generate_visible_variables(latent_distribution, latent_sample_size)

		print("Reconstruction Difference: ", torch.mean(torch.abs(reconstructed_data.to(torch.device("cpu")) - batch_data)))
		batch_index += 1