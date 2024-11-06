import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os


class Binarize(object):
    def __call__(self, tensor):
        # Binarize based on a 0.5 threshold (after normalization)
        return (tensor > 0.5).float()
    

class RBM:
    def __init__(self, visible_dim, hidden_dim, learning_rate, batch_size, n_iter, verbose, random_state):
        # Number of visible and hidden units
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def sample_from_prob(self, prob):
        # Sampling binary states from probabilities
        return torch.bernoulli(prob)

    def v_to_h(self, v):
        # Propagate visible layer to hidden layer
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        h_sample = self.sample_from_prob(h_prob)
        return h_prob, h_sample

    def h_to_v(self, h):
        # Propagate hidden layer to visible layer
        v_prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        v_sample = self.sample_from_prob(v_prob)
        return v_prob, v_sample
    
    def persistent_contrastive_divergence(self, v, iter):
        # Gibbs Sampling for Persistent Contrastive Divergence
        h_pos, _ = self.v_to_h(v)
        _, v_neg  = self.h_to_v(self.h_samples_)
        h_neg, _ = self.v_to_h(v_neg)
        
        # Positive and negative phase
        pos_phase = torch.matmul(h_pos.t(), v)  # Should match [hidden_dim, visible_dim]
        neg_phase = torch.matmul(h_neg.t(), v_neg)  # Should match [hidden_dim, visible_dim]

        # Update weights and biases
        lr = self.learning_rate/(self.batch_size)
        self.W += lr*(pos_phase - neg_phase) / v.size(0)
        self.v_bias += lr*torch.sum(v - v_neg, dim=0)
        self.h_bias += lr*torch.sum(h_pos - h_neg, dim=0)

        # Update the persistent chain
        self.h_samples_ = self.sample_from_prob(h_neg)

    def _free_energy(self, v):
        # Energy function of the RBM
        vbias_term = torch.matmul(v, self.v_bias.reshape(-1, 1))
        hidden_term = torch.sum(torch.log(1 + torch.exp(torch.matmul(v, self.W.t()) + self.h_bias)), dim=1)
        return -vbias_term - hidden_term

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : Tensor of shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).
        """
        # Ensure X is a PyTorch tensor
        X = torch.as_tensor(X).float()

        # Randomly corrupt one feature in each sample in X
        n_samples, n_features = X.shape
        ind = (torch.arange(n_samples), torch.randint(0, n_features, (n_samples,)))

        # Create a copy of X and corrupt one feature in each sample
        X_corrupted = X.clone()
        X_corrupted[ind] = 1 - X_corrupted[ind]

        # Calculate free energy for the original and corrupted inputs
        fe = self._free_energy(X)
        fe_corrupted = self._free_energy(X_corrupted)

        # Compute the pseudo-likelihood using the logistic function of the difference
        pseudo_likelihood = -n_features * torch.logaddexp(torch.tensor(0.0), -(fe_corrupted - fe))

        return pseudo_likelihood

    def encoder(self, dataset: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Generate top level latent variables
        """
        p_h_given_v, _ = self.v_to_h(dataset)
        return p_h_given_v

    def encode(self, dataloader: DataLoader) -> DataLoader:
        """
        Encode data
        """
        latent_vars = []
        labels = []
        for data, label in dataloader:
            data = data.view(-1, self.visible_dim)
            label = label.unsqueeze(1).to(torch.float32)
            latent_vars.append(self.encoder(data, label))
            labels.append(label)
        latent_vars = torch.cat(latent_vars, dim=0)
        labels = torch.cat(labels, dim=0)
        latent_dataset = TensorDataset(latent_vars, labels)

        return DataLoader(latent_dataset, batch_size=dataloader.batch_size, shuffle=False)
    
    def fit(self, data_loader: DataLoader, k=1):
        """
        Fit the RBM model
        """
        self.W = torch.randn(self.hidden_dim, self.visible_dim, dtype=torch.float32)
        self.v_bias = torch.zeros(self.visible_dim, dtype=torch.float32)
        self.h_bias = torch.zeros(self.hidden_dim, dtype=torch.float32)
        self.h_samples_ = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float32)
        
        for epoch in range(self.n_iter):
            epoch_loss = 0
            for batch in data_loader:
                v, _ = batch  # Ignore labels
                v = v.view(-1, self.visible_dim)  # Flatten the input images to [batch_size, visible_dim]
                # Update parameters
                self.persistent_contrastive_divergence(v, epoch)

            print(f"Epoch {epoch+1}/{self.n_iter} finished")


# Example usage
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize()
])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist, batch_size=64, shuffle=True, drop_last=True)

# Initialize and train the RBM
visible_dim = 28 * 28  # MNIST images are 28x28
hidden_dim = 128       # You can adjust this value
learning_rate = 0.1
batch_size = 64
n_iter = 10
verbose = 1
random_state = 42
rbm = RBM(visible_dim, hidden_dim, learning_rate, batch_size, n_iter, verbose, random_state)

rbm.fit(data_loader)

latent_loader = rbm.encode(data_loader)



latent, _ = latent_loader.dataset.tensors
all_data = []
all_labels = []

# Iterate through the DataLoader
for data, labels in data_loader:
    all_data.append(data.view(-1, rbm.visible_dim))
    all_labels.append(labels)

# Concatenate all data and labels into single tensors
full_data_tensor = torch.cat(all_data, dim=0)  # Shape: [num_samples, channels, height, width]
full_label_tensor = torch.cat(all_labels, dim=0)  # Shape: [num_samples]

latent_data = latent.detach().cpu().numpy()
true_label = full_label_tensor.cpu().numpy().flatten()
original_data = full_data_tensor.cpu().numpy()

directory = "../results/plots/RBM_new/UMAP_new/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
from umap import UMAP


digits = latent_data
umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

embedding = umap.fit_transform(digits)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute       

new_dir = directory
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
plt.scatter(embedding[:, 0], embedding[:, 1], c=true_label, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset with final latent embedding Ground Truth', fontsize=24)
plt.savefig(new_dir+"final_latent_embedding.png")
plt.show()
plt.close()

digits = original_data
embedding = umap.fit_transform(digits)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute

plt.scatter(embedding[:, 0], embedding[:, 1], c=true_label, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset with original data Ground Truth', fontsize=24)
plt.savefig(new_dir+"original_data.png")
plt.show()
plt.close()
