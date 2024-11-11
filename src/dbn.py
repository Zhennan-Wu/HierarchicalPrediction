import torch
import time
from typing import Any, Union, List, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
from rbm import RBM

from load_dataset import MNIST


class DBN:
    """
    Deep Belief Network
    """
    def __init__(self, input_size: int, layers: list, batch_size: int, epochs: int = 10, savefile: str = None, mode: str = "bernoulli", multinomial_top: bool=False, multinomial_sample_size: int=0, bias: bool = False, k: int = 5, gaussian_top = False, top_sigma: torch.Tensor = None, sigma: torch.Tensor = None, disc_alpha: float = 1.):
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        self.input_size = input_size
        self.layers = layers
        self.bias = bias
        self.batch_size = batch_size
        self.layer_parameters = [{"W":None, "hb":None, "vb":None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.gaussian_top = gaussian_top
        if (top_sigma is None):
            self.top_sigma = torch.ones((1,), dtype=torch.float32, device=self.device)/10.0
        else:
            self.top_sigma = top_sigma.to(torch.float32).to(self.device)
        if (sigma is None):
            self.sigma = torch.ones((input_size,), dtype=torch.float32, device=self.device)/10.0
        else:
            self.sigma = sigma.to(torch.float32).to(self.device)
        self.savefile = savefile
        self.epochs = epochs
        self.multinomial_top = multinomial_top
        self.multinomial_sample_size = multinomial_sample_size
        self.depthwise_training_loss = []
        self.top_parameters = {"W":None, "hb":None, "vb":None}
        self.disc_alpha = disc_alpha

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
            gaussian_dist = torch.distributions.normal.Normal(activation, self.sigma)
            variable = gaussian_dist.sample()
            p_v_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            raise ValueError("Invalid mode")
        return p_v_given_h, variable
    
    def sample_h(self, layer_index: int, x_bottom: torch.Tensor, label: torch.Tensor, top_down_sample: bool=False) -> torch.Tensor:
        """
        Sample hidden units given visible units
        """
        W_bottom = self.layer_parameters[layer_index]["W"]
        b_bottom = self.layer_parameters[layer_index]["hb"]
        if (layer_index == 0):
            activation = torch.matmul(x_bottom/(self.sigma**2), W_bottom.t()) + b_bottom
        else:    
            activation =torch.matmul(x_bottom, W_bottom.t()) + b_bottom 

        if (layer_index == len(self.layers)-1 and self.multinomial_top):
            if (top_down_sample):
                activation = activation + torch.matmul(label/self.top_sigma**2, self.top_parameters["W"])
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
            mean = torch.mm(x_bottom/(self.top_sigma**2), self.top_parameters["W"].t()) + self.top_parameters["hb"]
            gaussian_dist = torch.distributions.normal.Normal(mean, self.top_sigma)
            variable = gaussian_dist.sample()
            p_r_given_h = torch.exp(gaussian_dist.log_prob(variable))
        else:
            p_r_given_h = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
            variable = torch.ones((self.batch_size, 1), dtype=torch.float32, device=self.device)
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
                x_dash = self.generate_input_dataset_for_layer(index, batch, label)
                input_layer.append(x_dash)
                input_labels.append(label)
            input_data = torch.cat(input_layer, dim=0)
            input_labels = torch.cat(input_labels, dim=0)
            dataset = TensorDataset(input_data, input_labels)
            hidden_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            return hidden_loader

    def generate_input_dataset_for_layer(self, index: int, dataset: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
                    p_x, x_dash = self.sample_h(i, x_dash, labels)
                x_gen.append(p_x)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            x_binary = x_dash > 0.5
            return x_binary.to(torch.float32).to(self.device)
    
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
            rbm = RBM(n_components=hn, learning_rate=0.1, batch_size=self.batch_size, n_iter=self.epochs, verbose=0, add_bias=self.bias, target_in_model=self.gaussian_top, hybrid=False, input_dist=self.mode, latent_dist=mode, target_dist='gaussian')

            hidden_loader = self.generate_input_for_layer(index, dataloader)

            rbm.fit_dataloader(hidden_loader, vn, 1, self.multinomial_sample_size, self.disc_alpha)
            self.layer_parameters[index]["W"] = torch.tensor(rbm.components_, device=self.device)
            self.layer_parameters[index]["hb"] = torch.tensor(rbm.intercept_hidden_, device=self.device)
            self.layer_parameters[index]["vb"] = torch.tensor(rbm.intercept_visible_, device=self.device)
            self.top_parameters["W"] = torch.tensor(rbm.target_components_, device=self.device)
            self.top_parameters["hb"] = torch.tensor(rbm.intercept_target_, device=self.device)
            self.top_parameters["vb"] = torch.tensor(rbm.intercept_hidden_, device=self.device)

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
        for batch_data, label in dataloader:
            v_original = batch_data.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            v_reconstruct, _ = self.reconstructor(v_original, label, depth)
            train_loss += torch.mean(torch.abs(v_original - v_reconstruct))
        return train_loss.item()

    def reconstructor(self, x: torch.Tensor, y: torch.Tensor, depth: int = -1) -> torch.Tensor:
        """
        Reconstruct input
        """
        if (depth == -1):
            depth = len(self.layers)
        x_gen = []
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(depth):
                if (i == len(self.layers)-1 and self.gaussian_top):
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
            batch = batch.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            visible, latent = self.reconstructor(batch, label, depth)
            visible_data.append(visible)
            latent_vars.append(latent)
            data_labels.append(label)
        visible_data = torch.cat(visible_data, dim=0)
        latent_vars = torch.cat(latent_vars, dim=0)
        data_labels = torch.cat(data_labels, dim=0)
        dataset = TensorDataset(visible_data, latent_vars, data_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def encoder(self, dataset: torch.Tensor, label: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Generate top level latent variables
        """
        dataset = dataset.to(self.device)
        x_bottom = self.generate_input_dataset_for_layer(depth-1, dataset, label)
        W_bottom = self.layer_parameters[depth-1]["W"].to(self.device)
        b_bottom = self.layer_parameters[depth-1]["hb"].to(self.device)
        activation = torch.matmul(x_bottom, W_bottom.t()) + b_bottom 
        if (depth == len(self.layers)):
            activation = activation + (torch.matmul(label, self.top_parameters["W"].to(self.device)) + self.top_parameters["hb"].to(self.device))/self.top_sigma
        if (self.multinomial_top):
            p_h_given_v = torch.softmax(activation, dim=1)      
        else:
            p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v

    def encode(self, dataloader: DataLoader,  depth: int = -1) -> DataLoader:
        """
        Encode data
        """
        if (depth == -1):
            depth = len(self.layers)
        latent_vars = []
        labels = []
        for data, label in dataloader:
            data = data.to(self.device)
            label = label.unsqueeze(1).to(torch.float32).to(self.device)
            latent_vars.append(self.encoder(data, label, depth))
            labels.append(label)
        latent_vars = torch.cat(latent_vars, dim=0)
        labels = torch.cat(labels, dim=0)
        latent_dataset = TensorDataset(latent_vars, labels)

        return DataLoader(latent_dataset, batch_size=self.batch_size, shuffle=False)
    
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


def project_points_to_simplex(points):
    """
    Projects each point in a multidimensional cube [0, 1]^n onto the n-dimensional simplex.

    Parameters:
        points (np.ndarray): A 2D numpy array where each row is a point in the hypercube [0, 1]^n.

    Returns:
        np.ndarray: A 2D numpy array with each row projected onto the n-dimensional simplex.
    """
    # Number of points and dimension
    num_points, dim = points.shape
    
    # Array to store the projected points
    projected_points = np.zeros_like(points)
    
    for i in range(num_points):
        point = points[i]
        
        # Step 1: Sort the point in descending order
        u = np.sort(point)[::-1]
        
        # Step 2: Find the largest k such that the projection condition holds
        cumulative_sum = np.cumsum(u)
        rho = np.where(u > (cumulative_sum - 1) / (np.arange(dim) + 1))[0][-1]
        
        # Step 3: Compute theta
        theta = (cumulative_sum[rho] - 1) / (rho + 1)
        
        # Step 4: Project point onto the simplex
        projected_points[i] = np.maximum(point - theta, 0)
    
    return projected_points


if __name__ == "__main__":
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()
    print('MAE for all 0 selection:', torch.mean(train_x))
    batch_size = 1000	
    datasize = train_x.shape[0]
    data_dimension = train_x.shape[1]
    
    print("The whole dataset has {} data. The dimension of each data is {}. Batch size is {}.".format(datasize, data_dimension, batch_size))

    # train_x = train_x[:3*batch_size]
    # train_y = train_y[:3*batch_size]
    
    dataset = TensorDataset(train_x, train_y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for project_status in [True, False]:
        for experiment in ["bernoulli", "bernoulli_label", "multinomial", "multinomial_label"]:
            directory = "../results/plots/DBN/"
            experi_type = experiment
            if (project_status == True):
                directory = directory + "UMAP_proj_" + experi_type + "/"
                if (experiment == "multinomial" or experiment == "multinomial_label"):
                    continue
            else:
                directory = directory + "UMAP_" + experi_type + "/"
            filename = "dbn_" + experi_type + ".pth"
            if not os.path.exists(directory):
                os.makedirs(directory)

            if (experiment == "bernoulli"):
                dbn = DBN(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = 10, savefile=filename, mode = "bernoulli", multinomial_top = False, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = False, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
            elif (experiment == "bernoulli_label"):
                dbn = DBN(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = 10, savefile=filename, mode = "bernoulli", multinomial_top = False, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = True, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
            elif (experiment == "multinomial"):
                dbn = DBN(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = 10, savefile=filename, mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = False, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
            elif (experiment == "multinomial_label"):
                dbn = DBN(data_dimension, layers=[500, 300, 100], batch_size=batch_size, epochs = 10, savefile=filename, mode = "bernoulli", multinomial_top = True, multinomial_sample_size = 10, bias = False, k = 50, gaussian_top = True, top_sigma = 0.1*torch.ones((1,)), sigma = None, disc_alpha = 1.)
            else:
                raise ValueError("Invalid Experiment Type")
            dbn.train(data_loader)

            # dbn.load_model(filename)
            from sklearn.cluster import KMeans
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            # model test
            image_index = 0
            reconstructed_loader = dbn.reconstruct(data_loader)
            # directory = "../results/plots/DBN/Reconstructed/"

            # for data, latent, true_label in reconstructed_loader:
            #     for image, value in zip(data, true_label):
            #         plt.imshow(image.cpu().numpy().reshape(28, 28), cmap='gray')
            #         new_directory = directory+"true_label_{}/".format(value.item())
            #         if not os.path.exists(new_directory):
            #             os.makedirs(new_directory)
            #         plt.savefig(new_directory + "true_label_{}.png".format(image_index))
            #         image_index += 1
            latent_loader = dbn.encode(data_loader)
            first_layer = dbn.encode(data_loader, depth=1)
            second_layer = dbn.encode(data_loader, depth=2)


            first_level, _ = first_layer.dataset.tensors
            second_level, _ = second_layer.dataset.tensors
            final_level, _ = latent_loader.dataset.tensors
            original, labels = data_loader.dataset.tensors
            if (experiment == "multinomial" or experiment == "multinomial_label"):
                data = final_level.cpu().numpy()
            elif (experiment == "bernoulli" or experiment == "bernoulli_label"):
                if (project_status == True):
                    data = project_points_to_simplex(final_level.cpu().numpy())
                else:
                    data = final_level.cpu().numpy()
            else:
                raise ValueError("Invalid Experiment Type")
            first_level_data = first_level.cpu().numpy()
            second_level_data = second_level.cpu().numpy()
            true_label = labels.cpu().numpy().flatten()
            original_data = original.cpu().numpy()

            import numpy as np
            from sklearn.datasets import load_digits
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
            import umap


            digits = data
            reducer = umap.UMAP()
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
            plt.close()

            digits = second_level_data
            reducer = umap.UMAP()
            reducer.fit(digits)

            embedding = reducer.transform(digits)
            # Verify that the result of calling transform is
            # idenitical to accessing the embedding_ attribute
            assert(np.all(embedding == reducer.embedding_))
            embedding.shape        

            plt.scatter(embedding[:, 0], embedding[:, 1], c=true_label, cmap='Spectral', s=5)
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
            plt.title('UMAP projection of the Digits dataset with second layer latent embedding Ground Truth', fontsize=24)
            plt.savefig(new_dir+"second_latent_embedding.png")
            plt.close()

            digits = first_level_data
            reducer = umap.UMAP()
            reducer.fit(digits)

            embedding = reducer.transform(digits)
            # Verify that the result of calling transform is
            # idenitical to accessing the embedding_ attribute
            assert(np.all(embedding == reducer.embedding_))
            embedding.shape        

            plt.scatter(embedding[:, 0], embedding[:, 1], c=true_label, cmap='Spectral', s=5)
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
            plt.title('UMAP projection of the Digits dataset with first layer latent embedding Ground Truth', fontsize=24)
            plt.savefig(new_dir+"first_latent_embedding.png")
            plt.close()

            digits = original_data
            reducer = umap.UMAP()
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
            plt.close()
