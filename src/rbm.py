import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import trange
from typing import Any, Union, List, Tuple, Dict
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from load_dataset import MNIST


class RBM:
    """
    Restricted Boltzmann Machine
    """
    def __init__(self, num_visible: int, num_hidden: int, batch_size: int = 32,  epochs: int = 5, savefile: str = None, bias: bool = False, lr: float = 0.001, mode: str = "bernoulli", multinomial_sample_size: int = 0, k: int = 3, optimizer: str = "adam", early_stopping_patient: int = 5):
        self.mode = mode
        self.multinomial_sample_size = multinomial_sample_size
        self.bias = bias
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k
        self.optimizer = optimizer
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-7
        self.m = [0, 0, 0]
        self.v = [0, 0, 0]
        self.m_batches = {0:[], 1:[], 2:[]}
        self.v_batches = {0:[], 1:[], 2:[]}
        self.savefile = savefile
        self.early_stopping_patient = early_stopping_patient
        self.stagnation = 0
        self.previous_loss_before_stagnation = 0
        self.progress = []

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize weights
        std = 4*np.sqrt(6./(self.num_visible + self.num_hidden))  
        self.weights = torch.normal(mean=0, std=std, size=(self.num_hidden, self.num_visible), device=self.device)
        if (self.bias):
            self.hidden_bias = torch.zeros(self.num_hidden, dtype = torch.float32, device=self.device)

    def sample_h(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units
        """
        if (self.bias):
            activation = torch.mm(x, self.weights.t()) + self.hidden_bias
        else:
            activation = torch.mm(x, self.weights.t())
        if (self.mode == "bernoulli"):
            p_h_given_v = torch.sigmoid(activation)
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        elif (self.mode == "gaussian"):
            p_h_given_v = torch.sigmoid(activation)
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape, device=self.device))
        elif (self.mode == "multinomial"):
            activation = torch.mm(x, self.weights.t())
            p_h_given_v = torch.nn.functional.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device = self.device).scatter_(2, indices.unsqueeze(-1), 1)
            variable = torch.sum(one_hot, dim=1)
            return p_h_given_v, variable
        else:
            raise ValueError("Invalid mode")
    
    def sample_v(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units
        """
        if (self.bias):
            activation = torch.mm(y, self.weights)
        else:
            activation = torch.mm(y, self.weights)
        p_v_given_h = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        elif (self.mode == "gaussian"):
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape, device=self.device))
        elif (self.mode == "multinomial"):
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        else:
            raise ValueError("Invalid mode")
    
    def adam(self, g: torch.Tensor , epoch: int, index: int) -> torch.Tensor:
        """
        Adam optimizer
        """
        self.m[index] = self.beta_1*self.m[index] + (1-self.beta_1)*g
        self.v[index] = self.beta_2*self.v[index] + (1-self.beta_2)*torch.pow(g, 2)
        m_hat = self.m[index]/(1-np.power(self.beta_1, epoch)) + (1 - self.beta_1)*g/(1-np.power(self.beta_1, epoch))
        v_hat = self.v[index]/(1-np.power(self.beta_2, epoch))
        return m_hat/(torch.sqrt(v_hat) + self.epsilon)
    
    def update(self, v0: torch.Tensor, vk: torch.Tensor, ph0: torch.Tensor, phk: torch.Tensor, epoch: int):
        """
        Update weights and biases
        """
        dW = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        if (self.bias):
            dV = torch.sum(v0 - vk, dim=0)
            dH = torch.sum(ph0 - phk, dim=0)

        if (self.optimizer == "adam"):
            dW = self.adam(dW, epoch, 0)
            if (self.bias):
                dV = self.adam(dV, epoch, 1)
                dH = self.adam(dH, epoch, 2)

        self.weights += self.lr*dW
        if (self.bias):
            self.hidden_bias += self.lr*dH

    def train(self, dataloader: DataLoader):
        """
        Train RBM
        """
        learning = trange(self.epochs, desc=str("Starting..."))
        for epoch in learning:
            start_time = time.time()
            train_loss = torch.tensor([0.], device=self.device)
            counter = 0
            for batch_data, _ in dataloader:
                vk = batch_data.to(self.device)
                v0 = batch_data.to(self.device)
                ph0, _ = self.sample_h(v0)

                for _ in range(self.k):
                    _, hk = self.sample_h(vk)
                    _, vk = self.sample_v(hk)
                phk, _ = self.sample_h(vk)
                self.update(v0, vk, ph0, phk, epoch+1)
                train_loss += torch.mean(torch.abs(v0 - vk))
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
            print("Time taken for RBM epoch {} is {:.2f} sec".format(epoch+1, end_time-start_time))
        learning.close()
        if (self.savefile != None):
            if (self.bias):
                model = {"W": self.weights, "hb": self.hidden_bias}
            else:
                model = {"W": self.weights}
            torch.save(model, self.savefile)
        self.visualize_training_curve()

    def load_rbm(self, savefile: str):
        """
        Load RBM
        """
        model = torch.load(savefile)
        self.weights = model["W"].to(self.device)
        if (self.bias):
            self.hidden_bias = model["hb"].to(self.device)
    
    def visualize_training_curve(self):
        """
        Visualize training curve
        """
        plot_title = "Training Curve of {}".format(self.savefile)
        directory = "../results/plots/RBM/"
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
        
if __name__ == "__main__":
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()
    print('MAE for all 0 selection:', torch.mean(train_x))

    batch_size = 1000	
    print(train_x.shape)
    # train_x = train_x[:batch_size*3, :]
    # train_y = train_y[:batch_size*3]    

    # datasize = train_x.shape[0]
    # data_dimension = train_x.shape[1]
    # print(datasize, data_dimension, batch_size)

    # dataset = TensorDataset(train_x, train_y)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # rbm = RBM(data_dimension, 500, batch_size=batch_size, epochs=10, savefile="rbm.pth")
    # rbm.train(data_loader)
    # rbm.visualize_training_curve()