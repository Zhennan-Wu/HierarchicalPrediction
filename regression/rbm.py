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
    def __init__(self, num_visible: int, num_hidden: int, batch_size: int = 32, epochs: int = 5, savefile: str = None, bias: bool = False, lr: float = 0.001, mode: str = "bernoulli", multinomial_sample_size: int = 0, k: int = 3, optimizer: str = "adam", early_stopping_patient: int = 5, gaussian_top: bool = False, top_sigma: torch.Tensor = None, sigma: torch.Tensor = None, disc_alpha: float = 0.):
        """
        Initialize RBM
        """
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
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
        self.m = [0, 0, 0, 0, 0]
        self.v = [0, 0, 0, 0, 0]
        self.m_batches = {0:[], 1:[], 2:[], 3:[], 4:[]}
        self.v_batches = {0:[], 1:[], 2:[], 3:[], 4:[]}
        self.savefile = savefile
        self.early_stopping_patient = early_stopping_patient
        self.stagnation = 0
        self.previous_loss_before_stagnation = 0
        self.progress = []
        self.gaussian_top = gaussian_top
        if  (top_sigma == None):
            self.top_sigma = torch.ones((1,), dtype = torch.float32, device=self.device)
        else:
            self.top_sigma = top_sigma.to(torch.float32).to(self.device)
        if (sigma == None):
            self.sigma = torch.ones((num_visible,), dtype = torch.float32, device=self.device)
        else:
            self.sigma = sigma.to(torch.float32).to(self.device)
        self.disc_alpha = disc_alpha

        # Initialize weights (handle different mode and setting here by initialization)
        std = 4*np.sqrt(6./(self.num_visible + self.num_hidden))  
        self.weights = torch.normal(mean=0, std=std, size=(self.num_hidden, self.num_visible), device=self.device)
        if (self.gaussian_top):
            self.top_weights = torch.normal(mean=0, std=std, size=(1, self.num_hidden), device=self.device)
        else:
            self.top_weights = torch.zeros((1, self.num_hidden), dtype = torch.float32, device=self.device)
        if (self.bias):
            self.hidden_bias = torch.randn(self.num_hidden, dtype = torch.float32, device=self.device)
            self.visible_bias = torch.randn(self.num_visible, dtype = torch.float32, device=self.device)
            if (self.gaussian_top):
                self.top_bias = torch.randn(1, dtype = torch.float32, device=self.device)
            else:
                self.top_bias = torch.zeros(1, dtype = torch.float32, device=self.device)
        else:
            self.hidden_bias = torch.zeros(self.num_hidden, dtype = torch.float32, device=self.device)
            self.top_bias = torch.zeros(1, dtype = torch.float32, device=self.device)
            self.visible_bias = torch.zeros(self.num_visible, dtype = torch.float32, device=self.device)

    def sample_h(self, x: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units
        """
        activation = torch.mm(x/self.sigma, self.weights.t()) + self.hidden_bias + torch.mm(r/self.top_sigma, self.top_weights) + self.hidden_bias
        if (self.mode == "multinomial"):
            p_h_given_v = torch.nn.functional.softmax(activation, dim=1)
            indices = torch.multinomial(p_h_given_v, self.multinomial_sample_size, replacement=True)
            one_hot = torch.zeros(p_h_given_v.size(0), self.multinomial_sample_size, p_h_given_v.size(1), device = self.device).scatter_(2, indices.unsqueeze(-1), 1)
            variables = torch.sum(one_hot, dim=1)
        else:
            p_h_given_v = torch.sigmoid(activation)
            variables = torch.bernoulli(p_h_given_v)
        return p_h_given_v, variables
    
    def sample_v(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units
        """
        activation = torch.mm(y, self.weights) + self.visible_bias
        if (self.mode == "gaussian"):
            mean = activation*self.sigma
            gaussian_dist = torch.distributions.normal.Normal(mean, self.sigma)
            variable = gaussian_dist.sample()
            p_v_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            p_v_given_h = torch.sigmoid(activation)
            variable = torch.bernoulli(p_v_given_h)
        return p_v_given_h, variable
    
    def sample_r(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units
        """
        if (self.gaussian_top):
            mean = (torch.mm(x, self.top_weights.t()) + self.top_bias)*self.top_sigma
            gaussian_dist = torch.distributions.normal.Normal(mean, self.top_sigma)
            variable = gaussian_dist.sample()
            p_r_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            p_r_given_h = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
            variable = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
        return p_r_given_h, variable
        
    def adam(self, g: torch.Tensor , epoch: int, index: int) -> torch.Tensor:
        """
        Adam optimizer
        """
        self.m[index] = self.beta_1*self.m[index] + (1-self.beta_1)*g
        self.v[index] = self.beta_2*self.v[index] + (1-self.beta_2)*torch.pow(g, 2)
        m_hat = self.m[index]/(1-np.power(self.beta_1, epoch)) + (1 - self.beta_1)*g/(1-np.power(self.beta_1, epoch))
        v_hat = self.v[index]/(1-np.power(self.beta_2, epoch))
        return m_hat/(torch.sqrt(v_hat) + self.epsilon)
    
    def update(self, v0: torch.Tensor, vk: torch.Tensor, ph0: torch.Tensor, phk: torch.Tensor, r0: torch.Tensor, rk: torch.Tensor, epoch: int, discriminator = False):
        """
        Update weights and biases
        """
        dW = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        dW_top = (torch.mm(ph0.t(), r0) - torch.mm(phk.t(), rk)).t()
        dV = torch.sum(v0 - vk, dim=0)
        dH = torch.sum(ph0 - phk, dim=0)
        dR = torch.sum(r0 - rk, dim=0)

        if (self.optimizer == "adam"):
            dW = self.adam(dW, epoch, 0)
            dW_top = self.adam(dW_top, epoch, 1)
            dV = self.adam(dV, epoch, 2)
            dH = self.adam(dH, epoch, 3)
            dR = self.adam(dR, epoch, 4)

        if (discriminator):
            self.weights += self.lr*dW*self.disc_alpha
        else:
            self.weights += self.lr*dW

        if (self.gaussian_top):
            if (discriminator):
                self.top_weights += self.lr*dW_top*self.disc_alpha
            else:
                self.top_weights += self.lr*dW_top
                
        if (self.bias):
            if (discriminator):
                self.hidden_bias += self.lr*dH*self.disc_alpha
                self.visible_bias += self.lr*dV*self.disc_alpha
                if (self.gaussian_top):
                    self.top_bias += self.lr*dR*self.disc_alpha
            else:
                self.hidden_bias += self.lr*dH
                self.visible_bias += self.lr*dV
                if (self.gaussian_top):
                    self.top_bias += self.lr*dR

    def train(self, dataloader: DataLoader):
        """
        Train RBM
        """
        learning = trange(self.epochs, desc=str("Starting..."))
        for epoch in learning:
            start_time = time.time()
            train_loss = torch.tensor([0.], device=self.device)
            counter = 0
            for batch_data, label in dataloader:
                # Discriminator part
                disc_vk = batch_data.to(self.device)
                disc_v0 = batch_data.to(self.device)
                disc_rk = label.unsqueeze(1).to(torch.float).to(self.device)
                disc_r0 = label.unsqueeze(1).to(torch.float).to(self.device)
                
                disc_ph0, _ = self.sample_h(disc_v0, disc_r0)

                for _ in range(self.k):
                    _, disc_hk = self.sample_h(disc_vk, disc_rk)
                    _, disc_rk = self.sample_r(disc_hk)
                disc_phk, _ = self.sample_h(disc_vk, disc_rk)

                # Generation part
                vk = batch_data.to(self.device)
                v0 = batch_data.to(self.device)
                rk = label.unsqueeze(1).to(torch.float).to(self.device)
                r0 = label.unsqueeze(1).to(torch.float).to(self.device)
                
                ph0, _ = self.sample_h(v0, r0)

                for _ in range(self.k):
                    _, hk = self.sample_h(vk, rk)
                    _, vk = self.sample_v(hk)
                    _, rk = self.sample_r(hk)
                phk, _ = self.sample_h(vk, rk)
                self.update(v0, vk, ph0, phk, r0, rk, epoch+1)
                self.update(disc_v0, disc_vk, disc_ph0, disc_phk, disc_r0, disc_rk, epoch+1, discriminator=True)

                train_loss += torch.mean(torch.abs(v0 - vk)) + torch.mean(torch.abs(r0 - rk)) 
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
            model = {"W": self.weights, "TW": self.top_weights, "hb": self.hidden_bias, "vb": self.visible_bias, "tb": self.top_bias}
            torch.save(model, self.savefile)
        self.visualize_training_curve()

    def load_rbm(self, savefile: str):
        """
        Load RBM
        """
        model = torch.load(savefile, weights_only=False)
        self.weights = model["W"].to(self.device)
        self.top_weights = model["TW"].to(self.device)
        self.hidden_bias = model["hb"].to(self.device)
        self.visible_bias = model["vb"].to(self.device)
        self.top_bias = model["tb"].to(self.device)
    
    def visualize_training_curve(self):
        """
        Visualize training curve
        """
        plot_title = "Training Curve of {}".format(self.savefile.replace(".pth", ""))
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
    datasize = train_x.shape[0]
    data_dimension = train_x.shape[1]
    print("The whole dataset has {} data. The dimension of each data is {}. Batch size is {}.".format(datasize, data_dimension, batch_size))

    dataset = TensorDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    rbm = RBM(data_dimension, num_hidden=500, batch_size=batch_size, epochs=10, savefile="rbm.pth", bias = False, lr = 0.001, mode = "bernoulli", multinomial_sample_size = 10, k = 3, optimizer = "adam", early_stopping_patient = 5, gaussian_top = True, top_sigma = 3.*torch.ones((1,)), sigma = None, disc_alpha = 0.5)
    rbm.train(data_loader)
    rbm.visualize_training_curve()