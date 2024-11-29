import torch
import os
import time
from typing import Any, Union, List, Tuple, Dict
from dbn import DBN
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist
from utils import visualize_rbm, visualize_data, project_points_to_simplex
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from load_dataset import MNIST


class DBM(DBN):
    """
    Deep Boltzmann Machine
    """
    def __init__(self, input_size: int, layers: list, batch_size: int, epochs: int = 10, savefile: str = None, mode: str = "bernoulli", multinomial_top: bool=False, multinomial_sample_size: int = 0, bias: bool = False, k: int = 5, gaussian_top: bool = False, top_sigma: torch.Tensor = None, sigma: torch.Tensor = None, disc_alpha: float = 1.0):
        super().__init__(input_size, layers, batch_size, epochs, savefile, mode, multinomial_top, multinomial_sample_size, bias, k, gaussian_top, top_sigma, sigma, disc_alpha)

        self.layer_mean_field_parameters = [{"mu":None} for _ in range(len(layers))]
        self.regression_progress = []
        self.progress = []
    
    def two_way_sample_h(self, layer_index: int, x_bottom: torch.Tensor, x_top: torch.Tensor = None) -> torch.Tensor:
        """
        Sample hidden units given visible units
        """
        W_bottom = self.layer_parameters[layer_index]["W"]
        bias = self.layer_parameters[layer_index]["hb"]
        if (layer_index == len(self.layers)-1):
            W_top = self.top_parameters["W"]
        else:
            W_top = self.layer_parameters[layer_index+1]["W"]
        if (x_top is not None):
            x_top = x_top.to(self.device)
        else:
            x_top = torch.zeros((1, W_top.size(0)), device=self.device)

        if (layer_index == 0):
            activation = torch.matmul(x_bottom/(self.sigma**2), W_bottom.t()) + torch.matmul(x_top, W_top) + bias
        elif (layer_index == len(self.layers)-1):
            activation = torch.matmul(x_bottom, W_bottom.t()) + torch.matmul(x_top/(self.top_sigma**2), W_top) + bias
        else:
            activation = torch.matmul(x_bottom, W_bottom.t()) + torch.matmul(x_top, W_top) + bias

        if (layer_index == len(self.layers)-1 and self.multinomial_top):
            p_h_given_v = torch.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device=self.device).scatter_(2, indices.unsqueeze(-1), 1)
            variables = torch.sum(one_hot, dim=1)
        else:
            p_h_given_v = torch.sigmoid(activation)
            variables = torch.bernoulli(p_h_given_v)
        return p_h_given_v, variables

    def generate_latent_sample_for_layer(self, index: int, dataset: torch.Tensor) -> torch.Tensor:
        """
        Generate input for layer
        """
        if (index == 0):
            return dataset
        else:
            x_gen = []
            for _ in range(self.k):
                x_dash = dataset.to(self.device)
                for i in range(index):
                    _, x_dash = self.two_way_sample_h(i, x_dash)
                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            return x_dash

    def gibbs_update_dataloader(self, dataloader: DataLoader, gibbs_iterations, discriminator: bool = False) -> DataLoader:
        """
        Gibbs update dataloader
        """
        # Update samples (subsample a subset of Markov Chains is excluded, it might need be added later for better performance)
        new_mcmc = [[] for _ in range(len(self.layers)+2)]
        if (discriminator):
            pass
        else:
            for variables in dataloader:
                pre_updated_variables = [var.to(self.device) for var in variables]
                for _ in range(gibbs_iterations):
                    new_variables = []
                    for index in range(len(self.layers)+2):
                        if (index == 0):
                            _, var = self.sample_v(index, pre_updated_variables[index+1])
                            new_variables.append(var)
                        elif (index == len(self.layers)+1):
                            _, var = self.sample_r(pre_updated_variables[index-1])
                            new_variables.append(var)
                        else:  
                            _, var = self.two_way_sample_h(index-1, pre_updated_variables[index-1], pre_updated_variables[index+1])
                            new_variables.append(var)
                    pre_updated_variables = new_variables
                for index, var in enumerate(pre_updated_variables):
                    new_mcmc[index].append(var)
            new_tensor_variables = []
            for variable in new_mcmc:
                new_tensor_variables.append(torch.cat(variable))
        return DataLoader(TensorDataset(*new_tensor_variables), batch_size=self.batch_size, shuffle=False)

    def gibbs_update(self, data: torch.Tensor, label: torch.Tensor, gibbs_iterations: int) -> List[torch.Tensor]:
        """
        Gibbs update for reconstruction
        """
        # Update samples (subsample a subset of Markov Chains is excluded, it might need be added later for better performance)
        variables = []
        for index in range(len(self.layers)+1):
            variables.append(self.generate_latent_sample_for_layer(index, data))
        variables.append(label)
        for _ in range(gibbs_iterations):
            new_variables = []
            for index in range(len(self.layers)+2):
                if (index == 0):
                    _, var = self.sample_v(index, variables[index+1], label)
                    new_variables.append(var)
                elif (index == len(self.layers)+1):
                    _, var = self.sample_r(variables[-1])
                    new_variables.append(var)
                else:  
                    _, var = self.two_way_sample_h(index-1, variables[index-1], variables[index+1])
                    new_variables.append(var)
            variables = new_variables
        return variables
    
    def calc_ELBO(self, data: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the Evidence Lower Bound (ELBO) of the data
        '''
        with torch.no_grad():
            elbo = torch.tensor(0., device=self.device)
            for index in range(len(self.layers)-1):
                elbo += torch.sum(torch.matmul(self.layer_mean_field_parameters[index+1]["mu"], self.layer_parameters[index+1]["W"])*self.layer_mean_field_parameters[index]["mu"])
                if (self.bias):
                    elbo += torch.sum(self.layer_mean_field_parameters[index+1]["mu"]*self.layer_parameters[index+1]["hb"])
            elbo += torch.sum(torch.matmul(self.layer_mean_field_parameters[0]["mu"], self.layer_parameters[0]["W"])*data)

            for index in range(len(self.layers)):
                clamped_mf = torch.clamp(self.layer_mean_field_parameters[index]["mu"], min=1e-3, max=1-1e-3)
                raw = -clamped_mf*torch.log(clamped_mf) - (1-clamped_mf)*torch.log(1-clamped_mf)                         
                entropy = torch.sum(raw)
                elbo += entropy
        return elbo.item()

    def visualize_ELBO(self, dataset_index: int, epoch: int, elbos: list):
        """
        Visualize the training process
        """
        directory = "../results/plots/DBM/ELBOs/dataset_{}/".format(dataset_index) 
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt_title = "Training ELBO for epoch {} of dataset {}".format(epoch, dataset_index)
        x = np.arange(1, len(elbos)+1)
        plt.figure()
        plt.plot(x, np.array(elbos))
        plt.xlabel("Iterations")
        plt.ylabel("ELBO")
        plt.title(plt_title)
        plt.savefig(directory+"epoch_{}.png".format(epoch))
        plt.close()

    def visualize_training_curve(self):
        """
        Visualize the training process
        """
        plot_title = "Training Loss"
        directory = "../results/plots/DBM/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        x = np.arange(1, len(self.progress)+1)
        plt.figure()
        plt.plot(x, np.array(self.progress), label="Total Loss")
        plt.plot(x, np.array(self.regression_progress), label="Regression Loss")
        plt.title(plot_title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(directory + plot_title.replace(" ", "_") + ".png")
        plt.close()

    def train(self, dataloader: DataLoader, gibbs_iterations: int=50, mf_maximum_steps: int=30, mf_threshold: float=0.1, convergence_consecutive_hits: int=3):
        """
        Train DBM
        """
        # Initialize mean field parameters
        variables = [[] for _ in range(len(self.layers)+2)]
        for data, label in dataloader:
            for index in range(len(self.layers)+2):
                if (index == 0):
                    variables[index].append(data)
                elif (index == len(self.layers)+1):
                    variables[index].append(label.to(torch.float32).unsqueeze(1))
                else:
                    variables[index].append(self.generate_latent_sample_for_layer(index, data))
        
        tensor_variables = []
        for variable in variables:
            tensor_variables.append(torch.cat(variable))
        mcmc_loader = DataLoader(TensorDataset(*tensor_variables), batch_size=self.batch_size, shuffle=False)

        disc_loader = DataLoader(TensorDataset(*tensor_variables), batch_size=self.batch_size, shuffle=False)

        # Mean field updates
        learning = trange(self.epochs, desc=str("Starting..."))
        for epoch in learning:
            with torch.no_grad():
                start_time = time.time()
                train_loss = torch.tensor([0.], device=self.device)
                regression_loss = torch.tensor([0.], device=self.device)
                counter = 0
                mcmc_loader = self.gibbs_update_dataloader(mcmc_loader, gibbs_iterations)
                disc_loader = self.gibbs_update_dataloader(disc_loader, gibbs_iterations, discriminator=False)
                alpha =1./(1000 + epoch)
                dataset_index = 0
                for dataset, mcmc_samples, disc_samples in zip(dataloader, mcmc_loader, disc_loader):
                    elbos = []
                    label = dataset[1].unsqueeze(1).to(torch.float32).to(self.device)
                    dataset = dataset[0].to(self.device)
                    mcmc_samples = [sample.to(self.device) for sample in mcmc_samples]
                    disc_samples = [sample.to(self.device) for sample in disc_samples]
                    for index, _ in enumerate(self.layers):
                        unnormalized_mf_param = torch.rand((self.batch_size, self.layers[index]), device = self.device)
                        self.layer_mean_field_parameters[index]["mu"] = unnormalized_mf_param/torch.sum(unnormalized_mf_param, dim=1).unsqueeze(1)

                    mf_step = 0
                    mf_convergence_count = [0]*len(self.layers)
                    mf_difference = {k: [] for k in range(len(self.layers))}
                    while (mf_step < mf_maximum_steps):
                        for index, _ in enumerate(self.layers):
                            old_mu = self.layer_mean_field_parameters[index]["mu"]
                            if (index == len(self.layers)-1):
                                activation = torch.matmul(self.layer_mean_field_parameters[index-1]["mu"], self.layer_parameters[index]["W"].t()) + torch.matmul(label/self.top_sigma, self.top_parameters["W"])

                            elif (index == 0):
                                activation = torch.matmul(dataset/self.sigma, self.layer_parameters[index]["W"].t())+ torch.matmul(self.layer_mean_field_parameters[index+1]["mu"], self.layer_parameters[index+1]["W"])

                            else:
                                activation = torch.matmul(self.layer_mean_field_parameters[index-1]["mu"], self.layer_parameters[index]["W"].t()) + torch.matmul(self.layer_mean_field_parameters[index+1]["mu"], self.layer_parameters[index+1]["W"])

                            self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)
                            if ((self.layer_mean_field_parameters[index]["mu"] < 0).any()):
                                print(activation)
                                raise ValueError("Negative Mean Field Parameters")

                            new_diff = torch.max(torch.abs(old_mu - self.layer_mean_field_parameters[index]["mu"])).item()
                            mf_difference[index].append(new_diff)
                            if (new_diff < mf_threshold):
                                mf_convergence_count[index] += 1
                            else:
                                mf_convergence_count[index] -=1
                        elbos.append(self.calc_ELBO(dataset))
                        mf_step += 1
                        if (all(x > convergence_consecutive_hits for x in mf_convergence_count)):
                            print("Mean Field Converged with {} iterations".format(mf_step))
                            break
                    dataset_index += 1
                    if (mf_step == mf_maximum_steps):
                        torch.set_printoptions(precision=2)
                        # print("For episode {} dataset {}, Mean Field did not converge with maximum layerwise difference {}".format(epoch, dataset_index, mf_difference))
                        print("For episode {} dataset {}, Mean Field did not converge".format(epoch, dataset_index))
                        directory = "../results/plots/DBM/MF_Differences/dataset_{}/".format(dataset_index)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        plt.title("Mean Field Difference for dataset {} at epoch {}".format(dataset_index, epoch))
                        plt.plot(mf_difference[0], label="Layer 1")
                        plt.plot(mf_difference[1], label="Layer 2")
                        plt.plot(mf_difference[2], label="Layer 3")
                        plt.xlabel("MF Iterations")
                        plt.legend()
                        #  plt.show()
                        plt.savefig(directory+"epoch_{}.png".format(epoch))
                        plt.close()

                    self.visualize_ELBO(dataset_index, epoch, elbos)

                    # Update model parameters
                    for index, _ in enumerate(self.layers):
                        if (index == 0):
                            generation_loss = torch.matmul(self.layer_mean_field_parameters[index]["mu"].t(), dataset/self.sigma)/self.batch_size - torch.matmul(mcmc_samples[index+1].t(), mcmc_samples[index]/self.sigma)/self.batch_size
                            discrimination_loss = torch.matmul(self.layer_mean_field_parameters[index]["mu"].t(), dataset/self.sigma)/self.batch_size - torch.matmul(disc_samples[index+1].t(), disc_samples[index]/self.sigma)/self.batch_size
                            self.layer_parameters[index]["W"] = self.layer_parameters[index]["W"] + alpha * (generation_loss + self.disc_alpha * discrimination_loss)
                            if (self.bias):
                                generation_loss = (torch.sum(self.layer_mean_field_parameters[index]["mu"] - mcmc_samples[index+1], dim=0)/self.batch_size)
                                discrimination_loss = (torch.sum(self.layer_mean_field_parameters[index]["mu"] - disc_samples[index+1], dim=0)/self.batch_size)
                                self.layer_parameters[index]["hb"] = self.layer_parameters[index]["hb"] + alpha * (generation_loss + self.disc_alpha * discrimination_loss)

                                generation_loss = torch.sum(dataset/self.sigma - mcmc_samples[index]/self.sigma, dim=0)/self.batch_size
                                discrimination_loss = torch.sum(dataset/self.sigma - disc_samples[index]/self.sigma, dim=0)/self.batch_size
                                self.layer_parameters[index]["vb"] = self.layer_parameters[index]["vb"] + alpha * (generation_loss + self.disc_alpha * discrimination_loss)
                        else:
                            generation_loss = torch.matmul(self.layer_mean_field_parameters[index]["mu"].t(), self.layer_mean_field_parameters[index-1]["mu"])/self.batch_size - torch.matmul(mcmc_samples[index+1].t(), mcmc_samples[index])/self.batch_size
                            discrimination_loss = torch.matmul(self.layer_mean_field_parameters[index]["mu"].t(), self.layer_mean_field_parameters[index-1]["mu"])/self.batch_size - torch.matmul(disc_samples[index+1].t(), disc_samples[index])/self.batch_size
                            self.layer_parameters[index]["W"] = self.layer_parameters[index]["W"] + alpha * (generation_loss + self.disc_alpha * discrimination_loss)
                            if (self.bias):
                                generation_loss = torch.sum(self.layer_mean_field_parameters[index]["mu"] - mcmc_samples[index+1], dim=0)/self.batch_size
                                discrimination_loss = torch.sum(self.layer_mean_field_parameters[index]["mu"] - disc_samples[index+1], dim=0)/self.batch_size
                                self.layer_parameters[index]["hb"] = self.layer_parameters[index]["hb"] + alpha * (generation_loss + self.disc_alpha * discrimination_loss)

                                self.layer_parameters[index]["vb"] = self.layer_parameters[index-1]["hb"]
                    if (self.gaussian_top):
                        generation_loss = torch.matmul(label.t()/self.top_sigma, self.layer_mean_field_parameters[-1]["mu"])/self.batch_size - torch.matmul(mcmc_samples[-1].t()/self.top_sigma, mcmc_samples[-2])/self.batch_size
                        discrimination_loss = torch.matmul(label.t()/self.top_sigma, self.layer_mean_field_parameters[-1]["mu"])/self.batch_size - torch.matmul(disc_samples[-1].t()/self.top_sigma, disc_samples[-2])/self.batch_size
                        self.top_parameters["W"] = self.top_parameters["W"] + alpha * (generation_loss + self.disc_alpha * discrimination_loss)
                        if (self.bias):
                            generation_loss = torch.sum(label/self.top_sigma - mcmc_samples[-1]/self.top_sigma, dim=0)/self.batch_size
                            discrimination_loss = torch.sum(label/self.top_sigma - disc_samples[-1]/self.top_sigma, dim=0)/self.batch_size
                            self.top_parameters["hb"] = self.top_parameters["hb"] + alpha * (generation_loss + self.disc_alpha * discrimination_loss)

                            self.top_parameters["vb"] = self.layer_parameters[-1]["hb"]
                    counter += 1

                self.progress.append(train_loss.item()/counter)
                self.regression_progress.append(regression_loss.item()/counter)
                details = {"epoch": epoch+1, "loss": round(train_loss.item()/counter, 4), "regression_loss": round(regression_loss.item()/counter, 4)}
                learning.set_description(str(details))
                learning.refresh()    
                
                end_time = time.time()
                print("Time taken for DBM epoch {} is {}".format(epoch, end_time-start_time))
                if (epoch%50 == 0):
                    savefile = "dbm_epoch_{}.pth".format(epoch)
                    self.save_model(savefile)
                    print("Model saved at epoch", epoch)
                if (epoch%10 == 0):
                    self.visualize_training_curve()
        learning.close()   

        if (self.savefile != None):
            model = self.initialize_nn_model()
            savefile = self.savefile.replace(".pth", "_nn.pth")
            torch.save(model, savefile)     
            self.save_model()   


if __name__ == "__main__":
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()
    print('MAE for all 0 selection:', torch.mean(train_x))
    batch_size = 1000	
    epochs = 5
    datasize = train_x.shape[0]
    data_dimension = train_x.shape[1]
    
    print("The whole dataset has {} data. The dimension of each data is {}. Batch size is {}.".format(datasize, data_dimension, batch_size))

    dataset = TensorDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for experiment in ["multinomial_label", "bernoulli_label", "multinomial", "bernoulli"]:
        directory = "../results/plots/DBM/"
        experi_type = experiment
        directory = directory + "UMAP_" + experi_type + "/"
        dbn_name = "dbn_" + experi_type + ".pth"
        filename = "dbm_" + experi_type + ".pth"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if (experiment == "bernoulli"):
            dbm = DBM(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = False, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = False, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
        elif (experiment == "bernoulli_label"):
            dbm = DBM(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = False, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = True, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
        elif (experiment == "multinomial"):
            dbm = DBM(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = False, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
        elif (experiment == "multinomial_label"):
            dbm = DBM(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = epochs, savefile=filename, mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = True, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
        else:
            raise ValueError("Invalid Experiment Type")
        dbm.load_model(dbn_name)
        dbm.train(data_loader)

        latent_loader = dbm.encode(data_loader)
        new_dir = directory + "final_latent_embedding.png"
        visualize_data(latent_loader, 3, new_dir)
        print("Finished {} Experiment".format(experiment))