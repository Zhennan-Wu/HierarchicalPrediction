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
    def __init__(self, input_size: int, layers: list, batch_size: int, epochs: int = 100, savefile: str = None, mode: str = "bernoulli", multinomial_top: bool=False, multinomial_sample_size: int = 0, bias: bool = False, k: int = 5, early_stopping_patient: int = 20):
        self.input_size = input_size
        self.layers = layers
        self.bias = bias
        self.batch_size = batch_size
        if (self.bias):
            self.layer_parameters = [{"W":None, "hb":None} for _ in range(len(layers))]
        else:
            self.layer_parameters = [{"W":None} for _ in range(len(layers))]
        self.layer_mean_field_parameters = [{"mu":None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile
        self.epochs = epochs
        self.multinomial_top = multinomial_top
        self.multinomial_sample_size = multinomial_sample_size
        self.early_stopping_patient = early_stopping_patient
        self.stagnation = 0
        self.previous_loss_before_stagnation = 0
        self.training_loss = []
        self.progress = []

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def sample_v(self, layer_index: int, y: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given hidden units
        """
        y = y.to(self.device)
        W = self.layer_parameters[layer_index]["W"]
        if (self.bias):
            hb = self.layer_parameters[layer_index]["hb"]
            activation = torch.matmul(y, W) + hb
        else:
            activation = torch.matmul(y, W)
        p_v_given_h = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        elif (self.mode == "gaussian"):
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
    def sample_h(self, layer_index: int, x_bottom: torch.Tensor, x_top: torch.Tensor = None) -> torch.Tensor:
        """
        Sample hidden units given visible units
        """
        x_bottom = x_bottom.to(self.device)
        W_bottom = self.layer_parameters[layer_index]["W"]
        if (x_top is not None):
            W_top = self.layer_parameters[layer_index+1]["W"]
            x_top = x_top.to(self.device)
        if (self.bias):
            b_bottom = self.layer_parameters[layer_index]["hb"]
            if (x_top is not None):
                b_top = self.layer_parameters[layer_index+1]["hb"]
                activation = torch.matmul(x_bottom, W_bottom.t()) + torch.matmul(x_top, W_top) + b_bottom + b_top
            else:
                activation = torch.matmul(x_bottom, W_bottom.t()) + b_bottom
        else:
            if (x_top is not None):
                activation = torch.matmul(x_bottom, W_bottom.t()) + torch.matmul(x_top, W_top)
            else:
                activation = torch.matmul(x_bottom, W_bottom.t())

        if (layer_index == len(self.layers)-1 and self.multinomial_top):
            p_h_given_v = torch.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device=self.device).scatter_(2, indices.unsqueeze(-1), 1)
            variables = torch.sum(one_hot, dim=1)
            return p_h_given_v, variables

        p_h_given_v = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        elif (self.mode == "gaussian"):
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
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
            rbm = RBM(vn, hn, self.batch_size, epochs=self.epochs, savefile = "{}th layer_rbm.pth".format(index+1), bias = False, lr=0.0005, mode=mode, multinomial_sample_size=self.multinomial_sample_size, k=10, optimizer="adam", early_stopping_patient=10)

            hidden_batch = []
            for input_batch in input_layer:
                hidden_batch.append(torch.mean(torch.stack(input_batch), dim=0))
            hidden_data = torch.cat(hidden_batch)
            hidden_loader = DataLoader(TensorDataset(hidden_data, hidden_labels), batch_size=self.batch_size, shuffle=False)

            rbm.train(hidden_loader)
            self.layer_parameters[index]["W"] = rbm.weights

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
            torch.save(self.layer_parameters, savefile)
    
    def load_dbn(self, savefile: str):
        """
        Load DBN
        """
        dbn_model = torch.load(savefile, weights_only=False)
        for layer_no, layer in enumerate(dbn_model):
            # if (layer_no//2 == len(self.layer_parameters)-1):
            #     break
            if (layer_no%2 == 0):
                self.layer_parameters[layer_no//2]["W"] = layer.weight.to(self.device)
                if (self.bias):
                    self.layer_parameters[layer_no//2]["hb"] = layer.bias.to(self.device)
                print("Loaded Layer", layer_no//2)

    def load_dbm(self, savefile: str):
        """
        Load DBN
        """
        dbn_model = torch.load(savefile, weights_only=False)
        for layer_no, layer in enumerate(dbn_model):
            # if (layer_no//2 == len(self.layer_parameters)-1):
            #     break
            if (layer_no%2 == 0):
                self.layer_parameters[layer_no//2]["W"] = layer.weight.to(self.device)
                if (self.bias):
                    self.layer_parameters[layer_no//2]["hb"] = layer.bias.to(self.device)
                print("Loaded Layer", layer_no//2)
    
    def gibbs_update_dataloader(self, dataloader: DataLoader, gibbs_iterations) -> DataLoader:
        """
        """
        # Update samples (subsample a subset of Markov Chains is excluded, it might need be added later for better performance)
        new_mcmc = [[] for _ in range(len(self.layers)+1)]
        for variables in dataloader:
            pre_updated_variables = variables
            for _ in range(gibbs_iterations):
                new_variables = []
                for index in range(len(self.layers)+1):
                    if (index == 0):
                        _, var = self.sample_v(index, pre_updated_variables[index+1])
                        new_variables.append(var)
                    elif (index == len(self.layers)):
                        _, var = self.sample_h(index-1, pre_updated_variables[index-1])
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

    def gibbs_update(self, data: torch.Tensor, gibbs_iterations: int) -> List[torch.Tensor]:
        """
        """
        # Update samples (subsample a subset of Markov Chains is excluded, it might need be added later for better performance)
        variables = []
        for index in range(len(self.layers)+1):
            variables.append(self.generate_input_for_layer(index, data))

        for _ in range(gibbs_iterations):
            new_variables = []
            for index in range(len(self.layers)+1):
                if (index == 0):
                    _, var = self.sample_v(index, variables[index+1])
                    new_variables.append(var)
                elif (index == len(self.layers)):
                    _, var = self.sample_h(index-1, variables[index-1])
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

    def visualize_mean_field(self, dataset_index: int, epoch: int):
        """
        Visualize the training process
        """
        directory = "../results/plots/DBM/MeanField/dataset_{}/".format(dataset_index) 
        if not os.path.exists(directory):
            os.makedirs(directory)
        for index, layer in enumerate(self.layer_mean_field_parameters):
            plt_title = "Mean Field for layer {} of epoch {} of dataset {}".format(index, epoch, dataset_index)
            plt.figure()
            plt.imshow(layer["mu"].cpu().detach().numpy())
            plt.colorbar()
            plt.title(plt_title)
            plt.savefig(directory+"epoch_{}_layer_{}.png".format(epoch, index))
            plt.close()

    def train(self, dataloader: DataLoader, gibbs_iterations: int=200, mf_maximum_steps: int=100, mf_threshold: float=0.01, convergence_consecutive_hits: int=3):
        """
        Train DBM
        """
        # Initialize mean field parameters
        variables = [[] for _ in range(len(self.layers)+1)]
        for data, _ in dataloader:
            for index in range(len(self.layers)+1):
                if (index == 0):
                    variables[index].append(data)
                else:
                    variables[index].append(self.generate_input_for_layer(index, data))
        
        tensor_variables = []
        for variable in variables:
            tensor_variables.append(torch.cat(variable))
        mcmc_loader = DataLoader(TensorDataset(*tensor_variables), batch_size=self.batch_size, shuffle=False)

        # Mean field updates
        alpha = 0.01
        step_size = alpha/(self.epochs+1)
        learning = trange(self.epochs, desc=str("Starting..."))
        for epoch in learning:
            with torch.no_grad():
                start_time = time.time()
                train_loss = torch.tensor([0.], device=self.device)
                counter = 0
                mcmc_loader = self.gibbs_update_dataloader(mcmc_loader, gibbs_iterations)
                alpha -= step_size
                dataset_index = 0
                for dataset, mcmc_samples in zip(dataloader, mcmc_loader):
                    elbos = []
                    dataset = dataset[0].to(self.device)
                    mcmc_samples = [sample.to(self.device) for sample in mcmc_samples]
                    for index, _ in enumerate(self.layers):
                        unnormalized_mf_param = torch.rand((self.batch_size, self.layers[index]), device = self.device)
                        self.layer_mean_field_parameters[index]["mu"] = unnormalized_mf_param/torch.sum(unnormalized_mf_param, dim=1).unsqueeze(1)

                    mf_step = 0
                    mf_convergence_count = [0]*len(self.layers)
                    mf_difference = []
                    while (mf_step < mf_maximum_steps):
                        mf_difference = []
                        for index, _ in enumerate(self.layers):
                            old_mu = self.layer_mean_field_parameters[index]["mu"]
                            if (index == len(self.layers)-1):
                                activation = torch.matmul(self.layer_mean_field_parameters[index-1]["mu"], self.layer_parameters[index]["W"].t())
                            elif (index == 0):
                                activation = torch.matmul(dataset, self.layer_parameters[index]["W"].t()) + torch.matmul(self.layer_mean_field_parameters[index+1]["mu"], self.layer_parameters[index+1]["W"])
                            else:
                                activation = torch.matmul(self.layer_mean_field_parameters[index-1]["mu"], self.layer_parameters[index]["W"].t()) + torch.matmul(self.layer_mean_field_parameters[index+1]["mu"], self.layer_parameters[index+1]["W"])

                            self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)
                            if ((self.layer_mean_field_parameters[index]["mu"] < 0).any()):
                                print(activation)
                                raise ValueError("Negative Mean Field Parameters")

                            new_diff = torch.max(torch.abs(old_mu - self.layer_mean_field_parameters[index]["mu"])).item()
                            mf_difference.append(new_diff)
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
                    self.visualize_ELBO(dataset_index, epoch, elbos)

                    # Update model parameters
                    for index, _ in enumerate(self.layers):
                        if (index == 0):
                            self.layer_parameters[index]["W"] = self.layer_parameters[index]["W"] + alpha * (torch.matmul(self.layer_mean_field_parameters[index]["mu"].t(), dataset)/self.batch_size - torch.matmul(mcmc_samples[index+1].t(), mcmc_samples[index])/self.batch_size)
                            if (self.bias):
                                self.layer_parameters[index]["hb"] = self.layer_parameters[index]["hb"] + alpha * torch.sum(dataset - mcmc_samples[index+1], dim=0)/self.batch_size
                        else:
                            self.layer_parameters[index]["W"] = self.layer_parameters[index]["W"] + alpha * (torch.matmul(self.layer_mean_field_parameters[index]["mu"].t(), self.layer_mean_field_parameters[index-1]["mu"])/self.batch_size - torch.matmul(mcmc_samples[index+1].t(), mcmc_samples[index])/self.batch_size)
                            if (self.bias):
                                self.layer_parameters[index]["hb"] = self.layer_parameters[index]["hb"] + alpha * torch.sum(self.layer_mean_field_parameters[index]["mu"] - mcmc_samples[index+1], dim=0)/self.batch_size


                    train_loss += torch.mean(torch.abs(dataset - mcmc_samples[0]))
                    counter += 1

                self.progress.append(train_loss.item()/counter)
                details = {"epoch": epoch+1, "loss": round(train_loss.item()/counter, 4)}
                learning.set_description(str(details))
                learning.refresh()    

                # if (train_loss.item()/counter > self.previous_loss_before_stagnation and epoch>self.early_stopping_patient+1):
                #     self.stagnation += 1
                #     if (self.stagnation == self.early_stopping_patient-1):
                #         learning.close()
                #         print("Not Improving the stopping training loop.")
                #         break
                # else:
                #     self.previous_loss_before_stagnation = train_loss.item()/counter
                #     self.stagnation = 0
                
                end_time = time.time()
                print("Time taken for DBM epoch {} is {}".format(epoch, end_time-start_time))
                if (epoch%50 == 0):
                    savefile = "dbm_epoch_{}.pth".format(epoch)
                    torch.save(self.layer_parameters, savefile)
                    print("Model saved at epoch", epoch)

        learning.close()   

        self.visualize_training_curve()

        if (self.savefile != None):
            model = self.initialize_model()
            torch.save(model, self.savefile)        

    def reconstructor(self, x: torch.Tensor, depth: int = -1) -> torch.Tensor:
        """
        Reconstruct input
        """
        if (depth == -1):
            depth = len(self.layers)
        
        x = x.to(self.device)
        x_gen = []
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(depth):
                _, x_dash = self.sample_h(i, x_dash)
            x_gen.append(x_dash)
        x_dash = torch.stack(x_gen)
        x_dash = torch.mean(x_dash, dim=0)

        y = x_dash

        y_gen = []
        for _ in range(self.k):
            y_dash = y.clone()
            for i in range(depth-1, -1, -1):
                _, y_dash = self.sample_v(i, y_dash)
            y_gen.append(y_dash)
        y_dash = torch.stack(y_gen)
        y_dash = torch.mean(y_dash, dim=0)

        return y_dash, x_dash

    def reconstruct(self, dataloader: DataLoader, depth: int = -1) -> DataLoader:
        """
        Reconstruct input
        """
        visible_data = []
        latent_vars = []
        data_labels = []
        for batch, label in dataloader:
            visible, latent = self.reconstructor(batch, depth)
            visible_data.append(visible)
            latent_vars.append(latent)
            data_labels.append(label)
        visible_data = torch.cat(visible_data, dim=0)
        latent_vars = torch.cat(latent_vars, dim=0)
        data_labels = torch.cat(data_labels, dim=0)
        dataset = TensorDataset(visible_data, latent_vars, data_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def initialize_model(self):
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
    
    def encoder(self, dataset: torch.Tensor, repeat: int) -> torch.Tensor:
        """
        Generate top level latent variables
        """
        dataset = dataset.to(self.device)
        if (self.multinomial_top):
            x_bottom = self.generate_input_for_layer(len(self.layers)-1, dataset)
            W_bottom = self.layer_parameters[-1]["W"].to(self.device)
            if (self.bias):
                b_bottom = self.layer_parameters[-1]["hb"].to(self.device)
                activation = torch.matmul(x_bottom, W_bottom.t()) + b_bottom
            else:
                activation = torch.matmul(x_bottom, W_bottom.t())

            p_h_given_v = torch.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device=self.device).scatter_(2, indices.unsqueeze(-1), 1)
            return one_hot           
        else:
            x_gen = []
            for _ in range(repeat):
                x_dash = dataset
                for i in range(len(self.layers)):
                    _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]["W"])
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

            y_gen = []
            for _ in range(self.k):
                for i in range(len(self.layers)-1, -1, -1):
                    _, y_dash = self.sample_v(y_dash, self.layer_parameters[i]["W"])
                y_gen.append(y_dash)
            y_dash = torch.stack(y_gen)
            y_dash = torch.mean(y_dash, dim=0)

        return y_dash
    
    def encode(self, dataloader: DataLoader, repeat: int = 10) -> DataLoader:
        """
        Encode data
        """
        latent_vars = []
        labels = []
        for data, label in dataloader:
            latent_vars.append(self.encoder(data, repeat))
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
        labels = []
        for data, label in dataloader:
            visible_data.append(self.decoder(data, repeat))
            labels.append(label)
        visible_data = torch.cat(visible_data, dim=0)
        labels = torch.cat(labels, dim=0)
        visible_dataset = TensorDataset(visible_data, labels)

        return DataLoader(visible_dataset, batch_size=self.batch_size, shuffle=False)
    
    def initialize_model(self):
        """
        Initialize model
        """
        # print("The last layer will not be activated. The rest are activated using the Sigmoid function.")

        modules = []
        for index, layer in enumerate(self.layer_parameters):
            modules.append(torch.nn.Linear(layer["W"].shape[1], layer["W"].shape[0]))
            if (index < len(self.layer_parameters)-1):
                modules.append(torch.nn.Sigmoid())
        model = torch.nn.Sequential(*modules)
        model = model.to(self.device)

        for layer_no, layer in enumerate(model):
            # if (layer_no//2 == len(self.layer_parameters)-1):
            #     break
            if (layer_no%2 == 0):
                model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no//2]["W"])
                if (self.bias):
                    model[layer_no].bias = torch.nn.Parameter(self.layer_parameters[layer_no//2]["hb"])
        return model
    

if __name__ == "__main__":
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()
    print('MAE for all 0 selection:', torch.mean(train_x))

    batch_size = 1000	

    # train_x = train_x[:batch_size*3, :]
    # train_y = train_y[:batch_size*3]    

    datasize = train_x.shape[0]
    data_dimension = train_x.shape[1]
    print(datasize, data_dimension, batch_size)

    dataset = TensorDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dbm = DBM(data_dimension, [1000, 500, 100], batch_size, epochs = 400, savefile="dbm.pth", mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10)
    dbm.load_dbn("dbn.pth")
    dbm.train(data_loader)
