import torch
import time
from typing import Any, Union, List, Tuple, Dict
from rbm import RBM
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np

from load_dataset import MNIST


class DBN:
    """
    Deep Boltzmann Machine
    """
    def __init__(self, input_size: int, layers: list, batch_size: int, epoch: int = 10, savefile: str = None, mode: str = "bernoulli", multinomial_top: bool=False, multinomial_sample_size: int=0, bias: bool = False, k: int = 5, gaussian_top = False, top_sigma: float = 1, sigma: float = 1, disc_alpha: float = 0.):
        self.input_size = input_size
        self.layers = layers
        self.bias = bias
        self.batch_size = batch_size
        self.layer_parameters = [{"W":None, "hb":None, "vb":None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.gaussian_top = gaussian_top
        self.sigma = sigma*torch.ones((input_size,), dtype=torch.float32)
        self.top_sigma = top_sigma*torch.ones((1,), dtype=torch.float32)
        self.savefile = savefile
        self.epoch = epoch
        self.multinomial_top = multinomial_top
        self.multinomial_sample_size = multinomial_sample_size
        self.depthwise_training_loss = []
        self.top_parameters = {"W":None, "hb":None, "vb":None}
        self.disc_alpha = disc_alpha

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def sample_v(self, layer_index: int, y: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given hidden units
        """
        W = self.layer_parameters[layer_index]["W"].to(self.device)
        hb = self.layer_parameters[layer_index]["hb"].to(self.device)
        activation = torch.matmul(y, W) + hb

        if (self.mode == "bernoulli"):
            p_v_given_h = torch.sigmoid(activation)
            variable = torch.bernoulli(p_v_given_h)
        elif (self.mode == "gaussian"):
            mean = activation*self.sigma
            gaussian_dist = torch.distributions.normal.Normal(mean, self.sigma)
            variable = gaussian_dist.sample()
            p_v_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            raise ValueError("Invalid mode")
        return p_v_given_h, variable
    
    def sample_h(self, layer_index: int, x_bottom: torch.Tensor, label: torch.Tensor, top_down_sample: bool=False) -> torch.Tensor:
        """
        Sample hidden units given visible units
        """
        x_bottom = x_bottom.to(self.device)
        W_bottom = self.layer_parameters[layer_index]["W"].to(self.device)
        b_bottom = self.layer_parameters[layer_index]["vb"].to(self.device)
        if (layer_index == 0):
            activation = (torch.matmul(x_bottom, W_bottom.t()) + b_bottom)/self.sigma
        else:    
            activation =torch.matmul(x_bottom, W_bottom.t()) + b_bottom 

        if (layer_index == len(self.layers)-1 and self.multinomial_top):
            if (top_down_sample):
                activation = activation + (torch.matmul(label, self.top_parameters["W"].to(self.device)) + self.top_parameters["hb"].to(self.device))/self.top_sigma
            p_h_given_v = torch.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device=self.device).scatter_(2, indices.unsqueeze(-1), 1)
            variable = torch.sum(one_hot, dim=1)
        else:
            p_h_given_v = torch.sigmoid(activation)
            variable = torch.bernoulli(p_h_given_v)
        return p_h_given_v, variable
    
    def sample_r(self, x_bottom: torch.Tensor) -> torch.Tensor:
        """
        Sample reconstruction
        """
        if (self.gaussian_top):
            mean = (torch.mm(x_bottom, self.top_parameters["W"].t()) + self.top_parameters["vb"])*self.sigma
            gaussian_dist = torch.distributions.normal.Normal(mean, self.sigma)
            variable = gaussian_dist.sample()
            p_r_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            p_r_given_h = torch.tensor([1.], device=self.device)
            variable = x_bottom
        return p_r_given_h, variable
        
    def generate_input_for_layer(self, index: int, dataloader: DataLoader) -> DataLoader:
        """
        Generate input for layer
        """
        input_layer = []
        input_labels = []
        if (index == 0):
            return dataloader
        else:
            for batch, label in dataloader:
                x_gen = []
                for _ in range(self.k):
                    x_dash = batch.to(self.device)
                    for i in range(index):
                        _, x_dash = self.sample_h(i, x_dash, label)
                    x_gen.append(x_dash.detach())
                x_dash = torch.stack(x_gen)
                x_dash = torch.mean(x_dash, dim=0)
                input_layer.append(x_dash)
                input_labels.append(label)
            input_data = torch.cat(input_layer, dim=0)
            input_labels = torch.cat(input_labels, dim=0)
            dataset = TensorDataset(input_data, input_labels)
            hidden_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            return hidden_loader
        
    def train(self, dataloader: DataLoader):
        """
        Train DBN
        """
        for index, _ in enumerate(self.layers):
            start_time = time.time()
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
            rbm = RBM(vn, hn, self.batch_size, epochs=self.epoch, savefile="{}th layer_rbm.pth".format(index+1), bias = False, lr=0.0005, mode = mode, multinomial_sample_size=self.multinomial_sample_size, k=10, optimizer="adam", early_stopping_patient=10, gaussian_top=self.gaussian_top, top_sigma=self.top_sigma, sigma=self.sigma, disc_alpha=self.disc_alpha)

            hidden_loader = self.generate_input_for_layer(index, dataloader)

            rbm.train(hidden_loader)
            self.layer_parameters[index]["W"] = rbm.weights
            self.layer_parameters[index]["hb"] = rbm.hidden_bias
            self.layer_parameters[index]["vb"] = rbm.visible_bias
            self.top_parameters["W"] = rbm.top_weights
            self.top_parameters["hb"] = rbm.top_bias
            self.top_parameters["vb"] = rbm.hidden_bias

            print("Finished Training Layer", index, "to", index+1)
            training_loss = self.calc_training_loss(dataloader, index+1)
            print("Training Loss of DBN with {} layers:".format(index+1), training_loss)
            self.depthwise_training_loss.append(training_loss)
            end_time = time.time()
            print("Time taken for training DBN layer", index, "to", index+1, "is", end_time-start_time, "seconds")

        if (self.savefile is not None):
            model = self.initialize_nn_model()
            nn_savefile = self.savefile.replace(".pth", "_nn.pth")
            torch.save(model, nn_savefile)
            self.save_model()
        
        self.visualize_training_curve()

    def visualize_training_curve(self):
        """
        Visualize training curve
        """
        directory = "../results/plots/DBN/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt_title = "Training Loss for increasing depth of DBN"
        x = np.arange(1, len(self.depthwise_training_loss)+1)
        plt.plot(x, np.array(self.depthwise_training_loss))
        plt.xlabel("Depth")
        plt.ylabel("Training Loss")
        plt.title(plt_title)
        plt.savefig(directory + plt_title.replace(" ", "_") + ".png")
        plt.close()
        
    def calc_training_loss(self, dataloader: DataLoader, depth: int):
        '''
        '''
        train_loss = torch.tensor([0.], device=self.device)
        for batch_data, _ in dataloader:
            v_original = batch_data.to(self.device)
            v_reconstruct, _ = self.reconstructor(v_original, depth)
            train_loss += torch.mean(torch.abs(v_original - v_reconstruct))
        return train_loss.item()

    def reconstructor(self, x: torch.Tensor, y: torch.Tensor, depth: int = -1) -> torch.Tensor:
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
                if (depth == len(self.layers)-1 and self.gaussian_top):
                    top_down_sample = True
                    _, x_dash = self.sample_h(i, x_dash, y, top_down_sample)
                else:
                    _, x_dash = self.sample_h(i, x_dash, y)
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
            visible, latent = self.reconstructor(batch, label, depth)
            visible_data.append(visible)
            latent_vars.append(latent)
            data_labels.append(label)
        visible_data = torch.cat(visible_data, dim=0)
        latent_vars = torch.cat(latent_vars, dim=0)
        data_labels = torch.cat(data_labels, dim=0)
        dataset = TensorDataset(visible_data, latent_vars, data_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
    def load_model(self, savefile: str):
        """
        Load DBN or DBM model
        """
        model = torch.load(savefile)
        layer_parameters = []
        for index in range(len(model)):
            layer_parameters.append({"W":model["W"][index], "hb":model["hb"][index], "vb":model["vb"][index]})
        
        top_parameters = {"W":model["TW"][0], "hb":model["tb"][0], "vb":layer_parameters[-1]["hb"]}
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

    def load_rbm(self, layer: int, savefile: str):
        """
        Load RBM
        """
        model = torch.load(savefile)
        self.layer_parameters[layer]["W"] = model["W"].to(self.device)
        self.layer_parameters[layer]["vb"] = model["vb"].to(self.device)
        self.layer_parameters[layer]["hb"] = model["hb"].to(self.device)
        self.top_parameters["W"] = model["TW"].to(self.device)
        self.top_parameters["vb"] = model["hb"].to(self.device)
        self.top_parameters["hb"] = model["tb"].to(self.device)


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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    dbn = DBN(data_dimension, [1000, 500, 100], batch_size, epoch = 400, savefile="dbn.pth", mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10)
    dbn.train(data_loader)