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
    

class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super(RBM, self).__init__()
        
        # Number of visible and hidden units
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.1)  # weights
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))  # visible layer bias
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))   # hidden layer bias

    def sample_from_prob(self, prob):
        # Sampling binary states from probabilities
        return torch.bernoulli(prob)

    def v_to_h(self, v):
        # Propagate visible layer to hidden layer
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        h_sample = self.sample_from_prob(h_prob)
        return h_prob, h_sample

    def h_to_v(self, h):
        # Propagate hidden layer to visible layer
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        v_sample = self.sample_from_prob(v_prob)
        return v_prob, v_sample

    def forward(self, v):
        # Single step forward pass from visible to hidden and back
        h_prob, h_sample = self.v_to_h(v)
        v_prob, v_sample = self.h_to_v(h_sample)
        return v_prob, v_sample

    def contrastive_divergence(self, v, k=1):
        # Gibbs Sampling for Contrastive Divergence
        v_k = v
        for _ in range(k):
            h_prob, h_sample = self.v_to_h(v_k)
            v_prob, v_k = self.h_to_v(h_sample)
        
        # Positive and negative phase
        h_prob_0, _ = self.v_to_h(v)  # Hidden probabilities from initial visible layer
        pos_phase = torch.matmul(h_prob_0.t(), v)  # Should match [hidden_dim, visible_dim]
        neg_phase = torch.matmul(h_prob.t(), v_k)  # Should match [hidden_dim, visible_dim]

        # Update weights and biases
        self.W.grad = -(pos_phase - neg_phase) / v.size(0)
        self.v_bias.grad = -torch.mean(v - v_k, dim=0)
        self.h_bias.grad = -torch.mean(h_prob_0 - h_prob, dim=0)


    def energy(self, v):
        # Energy function of the RBM
        vbias_term = torch.matmul(v, self.v_bias)
        hidden_term = torch.sum(torch.log(1 + torch.exp(F.linear(v, self.W, self.h_bias))), dim=1)
        return -vbias_term - hidden_term

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
    
def train_rbm(rbm, data_loader, epochs=10, batch_size=64, k=1, learning_rate=0.01):
    optimizer = optim.SGD(rbm.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data_loader:
            v, _ = batch  # Ignore labels
            v = v.view(-1, rbm.visible_dim)  # Flatten the input images to [batch_size, visible_dim]

            # Zero gradients
            optimizer.zero_grad()

            # Contrastive Divergence
            rbm.contrastive_divergence(v, k=k)
            
            # Update parameters
            optimizer.step()

            # Calculate loss (Free Energy difference)
            batch_loss = torch.mean(rbm.energy(v))
            epoch_loss += batch_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data_loader)}")


# Example usage
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize()
])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist, batch_size=64, shuffle=True)

# Initialize and train the RBM
visible_dim = 28 * 28  # MNIST images are 28x28
hidden_dim = 128       # You can adjust this value

rbm = RBM(visible_dim, hidden_dim)
train_rbm(rbm, data_loader, epochs=10, batch_size=64, k=1, learning_rate=0.01)

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
import umap


digits = latent_data
reducer = umap.UMAP(random_state=42)
reducer.fit(digits)

embedding = reducer.transform(digits)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
embedding.shape        

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
reducer = umap.UMAP(random_state=42)
reducer.fit(digits)

embedding = reducer.transform(digits)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
embedding.shape        

plt.scatter(embedding[:, 0], embedding[:, 1], c=true_label, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset with original data Ground Truth', fontsize=24)
plt.savefig(new_dir+"original_data.png")
plt.show()
plt.close()
