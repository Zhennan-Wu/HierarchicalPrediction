import numpy as np
import torch
from tqdm import trange


class RBM_no_bias:
    """
    Restricted Boltzmann Machine
    """
    def __init__(self, num_visible, num_hidden, lr=0.001, epochs=5, mode="bernoulli", batch_size=32, k=3, optimizer="adam", gpu=False, savefile=None, early_stopping_patient=5):
        self.mode = mode
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

        if (torch.cuda.is_available() and gpu):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize weights
        std = 4*np.sqrt(6./(self.num_visible + self.num_hidden))  
        self.weights = torch.normal(mean=0, std=std, size=(self.num_hidden, self.num_visible), device=self.device)

    def sample_h(self, x):
        """
        Sample hidden units given visible units
        """
        activation = torch.mm(x, self.weights.t())
        p_h_given_v = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        elif (self.mode == "gaussian"):
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
    def sample_v(self, y):
        """
        Sample visible units given hidden units
        """
        activation = torch.mm(y, self.weights)
        p_v_given_h = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        elif (self.mode == "gaussian"):
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
    def adam(self, g, epoch, index):
        """
        Adam optimizer
        """
        self.m[index] = self.beta_1*self.m[index] + (1-self.beta_1)*g
        self.v[index] = self.beta_2*self.v[index] + (1-self.beta_2)*torch.pow(g, 2)
        m_hat = self.m[index]/(1-np.power(self.beta_1, epoch)) + (1 - self.beta_1)*g/(1-np.power(self.beta_1, epoch))
        v_hat = self.v[index]/(1-np.power(self.beta_2, epoch))
        return m_hat/(torch.sqrt(v_hat) + self.epsilon)
    
    def update(self, v0, vk, ph0, phk, epoch):
        """
        Update weights and biases
        """
        dW = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()

        if (self.optimizer == "adam"):
            dW = self.adam(dW, epoch, 0)

        self.weights += self.lr*dW

    def train(self, dataset):
        """
        Train RBM
        """
        dataset = torch.tensor(dataset, dtype=torch.float32).to(self.device)
        learning = trange(self.epochs, desc=str("Starting..."))
        for epoch in learning:
            train_loss = 0
            counter = 0
            for batch_start_index in range(0, dataset.shape[0]-self.batch_size, self.batch_size):
                vk = dataset[batch_start_index:batch_start_index+self.batch_size]
                v0 = vk
                ph0, _ = self.sample_h(v0)

                for k in range(self.k):
                    _, hk = self.sample_h(vk)
                    _, vk = self.sample_v(hk)
                    vk[v0 < 0] = v0[v0 < 0]
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
        learning.close()
        if (self.savefile != None):
            model = {"W": self.weights, "vb": self.visible_bias, "hb": self.hidden_bias}
            torch.save(model, self.savefile)

    def load_rbm(self, savefile):
        """
        Load RBM
        """
        model = torch.load(savefile)
        self.weights = model["W"].to(self.device)