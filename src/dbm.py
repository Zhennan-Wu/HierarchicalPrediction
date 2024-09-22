import torch
from typing import Any, Union, List, Tuple, Dict
from rbm_no_bias import RBM_no_bias


class DBM:
    """
    Deep Boltzmann Machine
    """
    def __init__(self, input_size, layers, mode="bernoulli", k=5, savefile=None):
        self.input_size = input_size
        self.layers = layers
        self.layer_parameters = [{"W":None} for _ in range(len(layers))]
        self.layer_mean_field_parameters = [{"mu":None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile

        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def sample_v(self, y, W):
        """
        Sample visible units given hidden units
        """
        activation = torch.matmul(y.to(self.device), W.to(self.device))   
        p_v_given_h = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            # return p_v_given_h, torch.bernoulli(p_v_given_h)
            return torch.bernoulli(p_v_given_h)
        elif (self.mode == "gaussian"):
            # return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape, device=self.device))
            return torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
    def sample_h(self, x_bottom, W_bottom, x_top = torch.zeros(1), W_top = torch.zeros(1, 1)):
        """
        Sample hidden units given visible units
        """
        activation = torch.matmul(x_bottom.to(self.device), W_bottom.t().to(self.device)) + torch.matmul(x_top.to(self.device), W_top.to(self.device))
        p_h_given_v = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            # return p_h_given_v, torch.bernoulli(p_h_given_v)
            return torch.bernoulli(p_h_given_v)
        elif (self.mode == "gaussian"):
            # return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape, device=self.device))
            return torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
    def generate_input_for_layer(self, index, x):
        """
        Generate input for layer
        """
        if (index == 0):
            return x
        else:
            x_gen = []
            for _ in range(self.k):
                x_dash = x.clone()
                for i in range(index):
                    x_dash = self.sample_h(x_dash, self.layer_parameters[i]["W"])
                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            return x_dash.to(self.device)
    
    def pre_train(self, dataset):
        """
        Train DBM
        """
        for index, _ in enumerate(self.layers):
            if (index == 0):
                vn = self.input_size
            else:
                vn = self.layers[index-1]
            hn = self.layers[index]

            rbm = RBM_no_bias(vn, hn, epochs=100, mode="bernoulli", lr=0.0005, k=10, batch_size=128, optimizer="adam", early_stopping_patient=10)
            if (rbm.device != self.device):
                self.device = rbm.device
            x_dash = self.generate_input_for_layer(index, dataset)
            rbm.train(x_dash)
            self.layer_parameters[index]["W"] = rbm.weights
            print("Finished Training Layer", index, "to", index+1)
        if (self.savefile is not None):
            torch.save(self.layer_parameters, self.savefile)
    
    def train(self, dataset, iterations=100, mf_maximum_steps=10):
        """
        Train DBM
        """
        # Initialize mean field parameters
        batch_size = int(dataset.size()[0])
        for index, _ in enumerate(self.layers):
            unnormalized_mf_param = torch.rand(self.layer_parameters[index]["W"].shape[0])
            self.layer_mean_field_parameters[index]["mu"] = unnormalized_mf_param/torch.sum(unnormalized_mf_param)
        variables = []
        for index in range(len(self.layers)+1):
            if (index == 0):
                variables.append(dataset)
            else:
                variables.append(self.generate_input_for_layer(index, dataset))

        # Mean field updates
        for _ in range(iterations):
            mf_step = 0
            mu_diff = torch.tensor(-1)
            while (mf_step < mf_maximum_steps):
                for index, _ in enumerate(self.layers):
                    if (index == len(self.layers)-1):
                        old_mu = self.layer_mean_field_parameters[index]["mu"]

                        activation = torch.matmul(self.layer_mean_field_parameters[index-1]["mu"].to(self.device), self.layer_parameters[index]["W"].t().to(self.device))
                        self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)

                        mu_diff = torch.max(torch.sum(torch.abs(old_mu.to(self.device) - self.layer_mean_field_parameters[index]["mu"].to(self.device))), mu_diff)
                    elif (index == 0):
                        old_mu = self.layer_mean_field_parameters[index]["mu"]
                        activation = torch.matmul(dataset.to(self.device), self.layer_parameters[index]["W"].t().to(self.device)) + torch.matmul(self.layer_mean_field_parameters[index+1]["mu"].to(self.device), self.layer_parameters[index+1]["W"].to(self.device))
                        self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)

                        mu_diff = torch.max(torch.sum(torch.abs(old_mu.to(self.device) - self.layer_mean_field_parameters[index]["mu"].to(self.device))), mu_diff)
                    else:
                        old_mu = self.layer_mean_field_parameters[index]["mu"]

                        activation = torch.matmul(self.layer_mean_field_parameters[index-1]["mu"].to(self.device), self.layer_parameters[index]["W"].t().to(self.device)) + torch.matmul(self.layer_mean_field_parameters[index+1]["mu"].to(self.device), self.layer_parameters[index+1]["W"].to(self.device))
                        self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)
                        
                        mu_diff = torch.max(torch.sum(torch.abs(old_mu.to(self.device) - self.layer_mean_field_parameters[index]["mu"].to(self.device))), mu_diff)
                mf_step += 1
                if (mu_diff < 0.001):
                    print("Mean Field Converged")
                    break
            if (mf_step == mf_maximum_steps):
                print("Mean Field did not converge with maximum difference {}".format(mu_diff))
            
            # Update samples (subsample a subset of Markov Chains is excluded, it might need be added later for better performance)
            new_variables = []
            for index in range(len(self.layers)+1):
                if (index == 0):
                    new_variables.append(self.sample_v(variables[index+1], self.layer_parameters[index]["W"]))
                elif (index == len(self.layers)):
                    new_variables.append(self.sample_h(variables[index-1], self.layer_parameters[index-1]["W"]))
                else:  
                    new_variables.append(self.sample_h(variables[index-1], self.layer_parameters[index-1]["W"], variables[index+1], self.layer_parameters[index]["W"]))
            
            # Update model parameters
            alpha = 0.01
            for index, _ in enumerate(self.layers):
                if (index == 0):
                    self.layer_parameters[index]["W"] += alpha * (torch.matmul(self.layer_mean_field_parameters[index]["mu"].t().to(self.device), variables[index].to(self.device))/batch_size - torch.matmul(new_variables[index+1].t().to(self.device), new_variables[index].to(self.device))/batch_size)
                else:
                    self.layer_parameters[index]["W"] += alpha * (torch.matmul(self.layer_mean_field_parameters[index]["mu"].t().to(self.device), self.layer_mean_field_parameters[index-1]["mu"].to(self.device))/batch_size - torch.matmul(new_variables[index+1].t().to(self.device), new_variables[index].to(self.device))/batch_size)
            
            variables = new_variables
            alpha -= 0.001

    def fine_tune(self, x, top_level_latent_variables):
        pass

    def reconstructor(self, x):
        x_gen = []
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(len(self.layer_parameters)):
                x_dash = self.sample_h(x_dash, self.layer_parameters[i]["W"])
            x_gen.append(x_dash)
        x_dash = torch.stack(x_gen)
        x_dash = torch.mean(x_dash, dim=0)

        y = x_dash

        y_gen = []
        for _ in range(self.k):
            y_dash = y.clone()
            for i in range(len(self.layer_parameters)-1, -1, -1):
                y_dash = self.sample_v(y_dash, self.layer_parameters[i]["W"])
            y_gen.append(y_dash)
        y_dash = torch.stack(y_gen)
        y_dash = torch.mean(y_dash, dim=0)

        return y_dash, x_dash

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

        for layer_no, layer in enumerate(model):
            if (layer_no//2 == len(self.layer_parameters)-1):
                break
            if (layer_no%2 == 0):
                model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no//2]["W"].to(self.device))
        return model
    
    def generate_top_level_latent_variables(self, dataset, repeat):
        """
        Generate top level latent variables
        """
        # version 1
        x_gen = []
        for _ in range(self.k):
            x_dash = dataset.clone()
            for i in range(len(self.layers)-1):
                x_dash = self.sample_h(x_dash, self.layer_parameters[i]["W"])
            x_gen.append(x_dash)
        x_gen = torch.stack(x_gen)
        x_dash = torch.mean(x_gen, dim=0)
        
        x_gen = []
        for _ in range(repeat):
            x_gen.append(self.sample_h(x_dash, self.layer_parameters[-1]["W"]))
        x_dash = torch.stack(x_gen, dim=1)
        return x_dash

    def generate_visible_variables(self, top_level_latent_variables_distributions, repeat):
        """
        Reconstruct observation
        """
        y_gen = []
        for _ in range(repeat):
            y_gen.append(torch.bernoulli(top_level_latent_variables_distributions))
        y_dash = torch.sum(torch.stack(y_gen, dim=1), dim=1)
        for i in range(len(self.layers)-1, -1, -1):
            y_dash = self.sample_v(y_dash, self.layer_parameters[i]["W"])
        return y_dash
    
if __name__ == "__main__":
    # Test DBM
    dbm = DBM(784, [500, 500, 2000], mode="bernoulli", k=5)
    dataset = torch.rand(128, 784)
    dbm.pre_train(dataset)
    dbm.train(dataset, 100)
    model = dbm.initialize_model()
    print(model)
    model = model.to(dbm.device)
    reconstructed = model(dataset.to(dbm.device))
    print(reconstructed)