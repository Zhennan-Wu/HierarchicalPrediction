import torch
import os
import time
from typing import Any, Union, List, Tuple, Dict
from dbn import DBN
from rbm import RBM
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
import pyro.distributions as dist

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from load_dataset import MNIST


class DBM:
    """
    Deep Boltzmann Machine
    """
    def __init__(self, input_size: int, layers: list, batch_size: int, epochs: int = 100, savefile: str = None, mode: str = "bernoulli", multinomial_top: bool=False, multinomial_sample_size: int = 0, bias: bool = False, k: int = 5, early_stopping_patient: int = 20, gaussian_top: bool = False, top_sigma: torch.Tensor = None, sigma: torch.Tensor = None, disc_alpha: float = 0.):
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        self.input_size = input_size
        self.layers = layers
        self.bias = bias
        self.batch_size = batch_size
        self.layer_parameters = [{"W":None, "hb":None, "vb": None} for _ in range(len(layers))]
        self.top_parameters = {"W":None, "hb":None, "vb": None}
        self.layer_mean_field_parameters = [{"mu":None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile
        self.epochs = epochs
        self.multinomial_top = multinomial_top
        self.gaussian_top = gaussian_top
        if (top_sigma == None):
            self.top_sigma = torch.ones((1,), dtype = torch.float32, device=self.device)
        else:
            self.top_sigma = top_sigma.to(torch.float32).to(self.device)
        if (sigma == None):
            self.sigma = torch.ones((input_size,), dtype = torch.float32, device=self.device)
        else:
            self.sigma = sigma.to(torch.float32).to(self.device)
        self.disc_alpha = disc_alpha
        self.multinomial_sample_size = multinomial_sample_size
        self.early_stopping_patient = early_stopping_patient
        self.stagnation = 0
        self.previous_loss_before_stagnation = 0
        self.training_loss = []
        self.progress = []

    def sample_v(self, layer_index: int, y: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given hidden units
        """
        W = self.layer_parameters[layer_index]["W"]
        vb = self.layer_parameters[layer_index]["vb"]
        activation = torch.matmul(y, W) + vb
        if (self.mode == "bernoulli"):
            p_v_given_h = torch.sigmoid(activation) 
            variable = torch.bernoulli(p_v_given_h)    
        elif (self.mode == "gaussian"):
            mean = activation * self.sigma
            gaussian_dist = torch.distributions.normal.Normal(mean, self.sigma)
            variable = gaussian_dist.sample()
            p_v_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            raise ValueError("Invalid mode")
        return p_v_given_h, variable
    
    def sample_h(self, layer_index: int, x_bottom: torch.Tensor, x_top: torch.Tensor = None) -> torch.Tensor:
        """
        Sample hidden units given visible units
        """
        W_bottom = self.layer_parameters[layer_index]["W"]
        b_bottom = self.layer_parameters[layer_index]["hb"]
        if (layer_index == len(self.layers)-1):
            W_top = self.top_parameters["W"]
            b_top = self.top_parameters["vb"]
        else:
            W_top = self.layer_parameters[layer_index+1]["W"]
            b_top = self.layer_parameters[layer_index+1]["vb"]
        if (x_top is not None):
            x_top = x_top.to(self.device)
        else:
            x_top = torch.zeros((1, W_top.size(0)), device=self.device)

        if (layer_index == 0):
            activation = torch.matmul(x_bottom/self.sigma, W_bottom.t()) + b_bottom + torch.matmul(x_top, W_top) + b_top
        elif (layer_index == len(self.layers)-1):
            activation = torch.matmul(x_bottom, W_bottom.t()) + torch.matmul(x_top/self.top_sigma, W_top) + b_bottom + b_top
        else:
            activation = torch.matmul(x_bottom, W_bottom.t()) + torch.matmul(x_top, W_top) + b_bottom + b_top

        if (layer_index == len(self.layers)-1 and self.multinomial_top):
            p_h_given_v = torch.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device=self.device).scatter_(2, indices.unsqueeze(-1), 1)
            variables = torch.sum(one_hot, dim=1)
        else:
            p_h_given_v = torch.sigmoid(activation)
            variables = torch.bernoulli(p_h_given_v)
        return p_h_given_v, variables
    
    def sample_r(self, x_bottom: torch.Tensor) -> torch.Tensor:
        """
        Sample reconstruction
        """
        if (self.gaussian_top):
            mean = (torch.mm(x_bottom, self.top_parameters["W"].t()) + self.top_parameters["hb"])*self.top_sigma
            gaussian_dist = torch.distributions.normal.Normal(mean, self.sigma)
            variable = gaussian_dist.sample()
            p_r_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            p_r_given_h = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
            variable = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
        return p_r_given_h, variable
        
    def generate_input_for_layer(self, index: int, dataset: torch.Tensor) -> torch.Tensor:
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
                    _, x_dash = self.sample_h(i, x_dash)
                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            return x_dash
        
    def pre_train(self, dataloader: DataLoader, savefile: str = None):
        """
        Train DBM
        """
        # duplicate input for k-step contrastive divergence
        input_layer = []
        input_labels = []
        for batch, label in dataloader:
            repeated_batch = []
            for _ in range(self.k):
                repeated_batch.append(batch)
            input_layer.append(input_batch)
            input_labels.append(label)
        hidden_labels = torch.cat(input_labels)

        for index, _ in enumerate(self.layers):
            if (index == 0):
                vn = self.input_size
            else:
                vn = self.layers[index-1]
            hn = self.layers[index]

            if (index == len(self.layers)-1):
                if (self.multinomial_top):
                    mode = "multinomial"
            else:
                mode = self.mode
            rbm = RBM(vn, hn, self.batch_size, epochs=self.epochs, savefile = "{}th layer_rbm.pth".format(index+1), bias = False, lr=0.0005, mode=mode, multinomial_sample_size=self.multinomial_sample_size, k=10, optimizer="adam", early_stopping_patient=10, gaussian_top=self.gaussian_top, top_sigma = self.top_sigma, sigma=self.sigma, disc_alpha=self.disc_alpha)

            hidden_batch = []
            for input_batch in input_layer:
                hidden_batch.append(torch.mean(torch.stack(input_batch), dim=0))
            hidden_data = torch.cat(hidden_batch)
            hidden_loader = DataLoader(TensorDataset(hidden_data, hidden_labels), batch_size=self.batch_size, shuffle=False)

            rbm.train(hidden_loader)
            self.layer_parameters[index]["W"] = rbm.weights
            self.layer_parameters[index]["hb"] = rbm.hidden_bias
            self.layer_parameters[index]["vb"] = rbm.visible_bias
            self.top_parameters["W"] = rbm.top_weights
            self.top_parameters["hb"] = rbm.top_bias
            self.top_parameters["vb"] = rbm.hidden_bias

            # generate next layer input
            new_input_layer = []
            for input_batch in input_layer:
                new_repeated_batch = []
                for x in input_batch:
                    _, var = self.sample_h(index, x)
                    new_repeated_batch.append(var)
                new_input_layer.append(new_repeated_batch)
            input_layer = new_input_layer
            print("Finished Training Layer", index, "to", index+1)

        if (savefile is not None):
            self.save_model(savefile)
    
    def load_model(self, savefile: str):
        """
        Load DBN or DBM model
        """
        model = torch.load(savefile, weights_only=False)
        layer_parameters = []
        for index in range(len(model["W"])):
            layer_parameters.append({"W":model["W"][index].to(self.device), "hb":model["hb"][index].to(self.device), "vb":model["vb"][index].to(self.device)})
        
        top_parameters = {"W":model["TW"][0].to(self.device), "hb":model["tb"][0].to(self.device), "vb":layer_parameters[-1]["hb"].to(self.device)}
        self.layer_parameters = layer_parameters
        self.top_parameters = top_parameters

    def load_nn_model(self, savefile: str):
        """
        Load nn model
        """
        dbn_model = torch.load(savefile, weights_only=False)
        for layer_no, layer in enumerate(dbn_model):
            # if (layer_no//2 == len(self.layer_parameters)-1):
            #     break
            if (layer_no%2 == 0):
                self.layer_parameters[layer_no//2]["W"] = layer.weight.to(self.device)
                if (self.bias):
                    self.layer_parameters[layer_no//2]["vb"] = layer.bias.to(self.device)
                print("Loaded Layer", layer_no//2)
        for index, layer in enumerate(self.layer_parameters):
            if (index < len(self.layer_parameters)-1):
                self.layer_parameters[index]["hb"] = self.layer_parameters[index+1]["vb"]

    def save_model(self, savefile: str = None):
        """
        Save model
        """
        if (savefile is None):
            savefile = self.savefile
        model = {"W": [], "vb": [], "hb": [], "TW": [], "tb": []}
        for layer in self.layer_parameters:
            model["W"].append(layer["W"])
            model["vb"].append(layer["vb"])
            model["hb"].append(layer["hb"])
        model["TW"].append(self.top_parameters["W"])
        model["tb"].append(self.top_parameters["hb"])
        torch.save(model, savefile)

    def initialize_nn_model(self):
        """
        Initialize model
        """
        print("The last layer will not be activated. The rest are activated using the Sigmoid function.")

        modules = []
        for index, layer in enumerate(self.layer_parameters):
            modules.append(torch.nn.Linear(layer["W"].shape[1], layer["W"].shape[0]))
            if (index < len(self.layer_parameters)-1):
                modules.append(torch.nn.Sigmoid())
        model = torch.nn.Sequential(*modules)
        model = model.to(self.device)

        for layer_no, layer in enumerate(model):
            if (layer_no//2 == len(self.layer_parameters)-1):
                break
            if (layer_no%2 == 0):
                model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no//2]["W"])
        return model
    
    def gibbs_update_dataloader(self, dataloader: DataLoader, gibbs_iterations, discriminator: bool = False) -> DataLoader:
        """
        Gibbs update dataloader
        """
        # Update samples (subsample a subset of Markov Chains is excluded, it might need be added later for better performance)
        new_mcmc = [[] for _ in range(len(self.layers)+2)]
        if (discriminator):
            for variables in dataloader:
                pre_updated_variables = variables
                for _ in range(gibbs_iterations):
                    new_variables = []
                    for index in range(len(self.layers)+2):
                        if (index == 0):
                            var = pre_updated_variables[index]
                            new_variables.append(var)
                        elif (index == len(self.layers)+1):
                            _, var = self.sample_r(pre_updated_variables[index-1])
                            new_variables.append(var)
                        else:  
                            _, var = self.sample_h(index-1, pre_updated_variables[index-1], pre_updated_variables[index+1])
                            new_variables.append(var)
                    pre_updated_variables = new_variables
                for index, var in enumerate(pre_updated_variables):
                    new_mcmc[index].append(var)
            new_tensor_variables = []
            for variable in new_mcmc:
                new_tensor_variables.append(torch.cat(variable))
        else:
            for variables in dataloader:
                pre_updated_variables = variables
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
                            _, var = self.sample_h(index-1, pre_updated_variables[index-1], pre_updated_variables[index+1])
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
            variables.append(self.generate_input_for_layer(index, data))
        variables.append(label)
        for _ in range(gibbs_iterations):
            new_variables = []
            for index in range(len(self.layers)+2):
                if (index == 0):
                    _, var = self.sample_v(index, variables[index+1])
                    new_variables.append(var)
                elif (index == len(self.layers)+1):
                    _, var = self.sample_r(variables[-1])
                    new_variables.append(var)
                else:  
                    _, var = self.sample_h(index-1, variables[index-1], variables[index+1])
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
        plt.plot(x, np.array(self.progress))
        plt.title(plot_title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(directory + plot_title.replace(" ", "_") + ".png")
        plt.close()

    def train(self, dataloader: DataLoader, gibbs_iterations: int=3, mf_maximum_steps: int=100, mf_threshold: float=0.1, convergence_consecutive_hits: int=3):
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
                    variables[index].append(self.generate_input_for_layer(index, data))
        
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
                counter = 0
                mcmc_loader = self.gibbs_update_dataloader(mcmc_loader, gibbs_iterations)
                disc_loader = self.gibbs_update_dataloader(disc_loader, gibbs_iterations, discriminator=True)
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
                        print("For episode {} dataset {}, Mean Field did not converge with layerwise difference {}".format(epoch, dataset_index, mf_difference))
                        directory = "../results/plots/DBM/MF_Differences/dataset_{}/".format(dataset_index)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        plt.title("Mean Field Difference for dataset {} at epoch {}".format(dataset_index, epoch))
                        plt.plot(mf_difference[0], label="Layer 1")
                        plt.plot(mf_difference[1], label="Layer 2")
                        plt.plot(mf_difference[2], label="Layer 3")
                        plt.xlabel("MF Iterations")
                        plt.legend()
                        plt.show()
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

                    reconstructed_data, _, reconstrcted_label = self.reconstructor(dataset, label, gibbs_iterations)
                    train_loss += torch.mean(torch.abs(dataset - reconstructed_data.to(self.device))) + torch.mean(torch.abs(label - reconstrcted_label.to(self.device)))
                    counter += 1

                self.progress.append(train_loss.item()/counter)
                details = {"epoch": epoch+1, "loss": round(train_loss.item()/counter, 4)}
                learning.set_description(str(details))
                learning.refresh()    

                if (train_loss.item()/counter > self.previous_loss_before_stagnation and epoch>self.early_stopping_patient+1):
                    self.stagnation += 1
                    if (self.stagnation == self.early_stopping_patient-1):
                        learning.close()
                        print("Not Improving the stopping training loop.")
                        break
                else:
                    self.previous_loss_before_stagnation = train_loss.item()/counter
                    self.stagnation = 0
                
                end_time = time.time()
                print("Time taken for DBM epoch {} is {}".format(epoch, end_time-start_time))
                if (epoch%50 == 0):
                    savefile = "dbm_epoch_{}.pth".format(epoch)
                    self.save_model(savefile)
                    print("Model saved at epoch", epoch)

        learning.close()   

        self.visualize_training_curve()

        if (self.savefile != None):
            model = self.initialize_nn_model()
            savefile = self.savefile.replace(".pth", "_nn.pth")
            torch.save(model, savefile)     
            self.save_model()   

    def reconstructor(self, x: torch.Tensor, label: torch.Tensor, repeat: int = 1, depth: int = -1) -> torch.Tensor:
        """
        Reconstruct input
        """
        if (depth == -1):
            depth = len(self.layers)
        
        for _ in range(repeat):
            x_gen = []
            r_gen = []
            for _ in range(self.k):
                x_dash = x.clone()
                for i in range(depth):
                    if (i == len(self.layers)-1):
                        _, r_dash = self.sample_h(i, x_dash)
                        _, r_dash = self.sample_r(r_dash)
                        r_gen.append(r_dash)
                        _, x_dash = self.sample_h(i, x_dash, label)
                    else:
                        _, x_dash = self.sample_h(i, x_dash)

                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            r_dash = torch.stack(r_gen)
            r_dash = torch.mean(r_dash, dim=0)

            y = x_dash

            y_gen = []
            for _ in range(self.k):
                y_dash = y.clone()
                for i in range(depth-1, -1, -1):
                    _, y_dash = self.sample_v(i, y_dash)
                y_gen.append(y_dash)
            y_dash = torch.stack(y_gen)
            y_dash = torch.mean(y_dash, dim=0)
            x = y_dash

        return y_dash, x_dash, r_dash

    def reconstruct(self, dataloader: DataLoader, repeat: int = 1, depth: int = -1) -> DataLoader:
        """
        Reconstruct input
        """
        visible_data = []
        latent_vars = []
        data_labels = []
        pseudo_labels = []
        for batch, label in dataloader:
            batch = batch.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            visible, latent, pseudo_labels = self.reconstructor(batch, label, repeat, depth)
            visible_data.append(visible)
            latent_vars.append(latent)
            data_labels.append(label)
            pseudo_labels.append(pseudo_labels)
        visible_data = torch.cat(visible_data, dim=0)
        latent_vars = torch.cat(latent_vars, dim=0)
        data_labels = torch.cat(data_labels, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        dataset = TensorDataset(visible_data, latent_vars, data_labels, pseudo_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
   
    def encoder(self, dataset: torch.Tensor, label: torch.Tensor, repeat: int) -> torch.Tensor:
        """
        Generate top level latent variables
        """
        dataset = dataset.to(self.device)
        if (self.multinomial_top):
            x_bottom = self.generate_input_for_layer(len(self.layers)-1, dataset)
            W_bottom = self.layer_parameters[-1]["W"].to(self.device)
            b_bottom = self.layer_parameters[-1]["hb"].to(self.device)
            activation = torch.matmul(x_bottom, W_bottom.t()) + b_bottom + (torch.matmul(label, self.top_parameters["W"].to(self.device)) + self.top_parameters["hb"].to(self.device))/self.top_sigma
            p_h_given_v = torch.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device=self.device).scatter_(2, indices.unsqueeze(-1), 1)
            return one_hot           
        else:
            x_gen = []
            for _ in range(repeat):
                x_dash = dataset
                for i in range(len(self.layers)):
                    if (i == len(self.layers)-1):
                        _, x_dash = self.sample_h(i, x_dash, label)
                    else:
                        _, x_dash = self.sample_h(i, x_dash)
                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen, dim=1)
            return x_dash

    def decoder(self, top_level_latent_variables_distributions: torch.Tensor, repeat: int):
        """
        Reconstruct observation
        """
        if (self.multinomial_top):
            x = dist.Bernoulli(probs=top_level_latent_variables_distributions).sample((self.multinomial_sample_size,))
            y_dash = torch.sum(x, dim=1)

            _, r_dash = self.sample_r(y_dash)

            y_gen = []
            for _ in range(self.k):
                for i in range(len(self.layers)-1, -1, -1):
                    _, y_dash = self.sample_v(y_dash, self.layer_parameters[i]["W"])
                y_gen.append(y_dash)
            y_dash = torch.stack(y_gen)
            y_dash = torch.mean(y_dash, dim=0)
        else:
            top_level_latent_variables_distributions = top_level_latent_variables_distributions.to(self.device)
            y_gen = []
            for _ in range(repeat):
                y_gen.append(torch.bernoulli(top_level_latent_variables_distributions))
            y_dash = torch.sum(torch.stack(y_gen, dim=1), dim=1)

            _, r_dash = self.sample_r(y_dash)

            y_gen = []
            for _ in range(self.k):
                for i in range(len(self.layers)-1, -1, -1):
                    _, y_dash = self.sample_v(y_dash, self.layer_parameters[i]["W"])
                y_gen.append(y_dash)
            y_dash = torch.stack(y_gen)
            y_dash = torch.mean(y_dash, dim=0)

        return y_dash, r_dash
    
    def encode(self, dataloader: DataLoader, repeat: int = 10) -> DataLoader:
        """
        Encode data
        """
        latent_vars = []
        labels = []
        for data, label in dataloader:
            data = data.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            latent_vars.append(self.encoder(data, label, repeat))
            labels.append(label)
        latent_vars = torch.cat(latent_vars, dim=0)
        labels = torch.cat(labels, dim=0)
        latent_dataset = TensorDataset(latent_vars, labels)

        return DataLoader(latent_dataset, batch_size=self.batch_size, shuffle=False)
    
    def decode(self, dataloader: DataLoader, repeat: int = 10) -> DataLoader:
        """
        Decode data
        """
        visible_data = []
        pseudo_labels = []
        labels = []
        for data, label in dataloader:
            pseudo_data, pseudo_label = self.decoder(data, repeat)
            visible_data.append(pseudo_data)
            pseudo_labels.append(pseudo_label)
            labels.append(label)
        visible_data = torch.cat(visible_data, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        labels = torch.cat(labels, dim=0)
        if (self.gaussian_top):
            visible_dataset = TensorDataset(visible_data, pseudo_labels, labels)
        else:
            visible_dataset = TensorDataset(visible_data, labels)

        return DataLoader(visible_dataset, batch_size=self.batch_size, shuffle=False)
    

if __name__ == "__main__":
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()
    print('MAE for all 0 selection:', torch.mean(train_x))

    batch_size = 1000	
    datasize = train_x.shape[0]
    data_dimension = train_x.shape[1]
    print("The whole dataset has {} data. The dimension of each data is {}. Batch size is {}.".format(datasize, data_dimension, batch_size))

    # train_x = train_x[:batch_size*3]
    # train_y = train_y[:batch_size*3]    

    dataset = TensorDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dbm = DBM(data_dimension, layers=[1000, 500, 100], batch_size=batch_size, epochs = 400, savefile="dbm.pth", mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10, bias = False, k = 5, early_stopping_patient = 20, gaussian_top = False, top_sigma = 0.5*torch.ones((1,), dtype=torch.float32), sigma = None, disc_alpha = 0.5)
    dbm.load_model("dbn.pth")
    dbm.train(data_loader)
