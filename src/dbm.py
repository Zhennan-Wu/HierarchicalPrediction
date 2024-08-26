import torch
from rbm_no_bias import RBM_no_bias


class DBM:
    """
    Deep Boltzmann Machine
    """
    def __init__(self, input_size, layers, mode="bernoulli", gpu=False, k=5, savefile=None):
        self.input_size = input_size
        self.layers = layers
        self.layer_parameters = [{"W":None} for _ in range(len(layers))]
        self.layer_mean_field_parameters = [{"mu":None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile
    
    def sample_v(self, y, W):
        """
        Sample visible units given hidden units
        """
        activation = torch.mm(y, W)   
        p_v_given_h = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        elif (self.mode == "gaussian"):
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
    def sample_h(self, x_bottom, W_bottom, x_top = torch.zeros(1), W_top = torch.zeros(1, 1)):
        """
        Sample hidden units given visible units
        """
        activation = torch.mm(x_bottom, W_bottom.t()) + torch.mm(x_top, W_top.t())
        p_h_given_v = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        elif (self.mode == "gaussian"):
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape, device=self.device))
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
                    _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]["W"])
                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            return x_dash
    
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

            rbm = RBM_no_bias(vn, hn, epochs=100, mode="bernoulli", lr=0.0005, k=10, batch_size=128, gpu=True, optimizer="adam", early_stopping_patient=10)
            x_dash = self.generate_input_for_layer(index, dataset)
            rbm.train(x_dash)
            self.layer_parameters[index]["W"] = rbm.weights
            print("Finished Training Layer", index, "to", index+1)
        if (self.savefile is not None):
            torch.save(self.layer_parameters, self.savefile)
    
    def train(self, dataset, iterations, mf_maximum_steps=10):
        """
        Train DBM
        """
        # Initialize mean field parameters
        batch_size, data_dimension = dataset.size()
        for index, _ in enumerate(self.layers):
            unnormalized_mf_param = torch.rand(data_dimension)
            self.layer_mean_field_parameters[index]["mu"] = unnormalized_mf_param/torch.sum(unnormalized_mf_param)
        variables = []
        for index, _ in enumerate(self.layers):
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
                    if (index == self.layers-1):
                        old_mu = self.layer_mean_field_parameters[index]["mu"]

                        activation = torch.mm(self.layer_mean_field_parameters[index-1]["mu"], self.layer_parameters[index]["W"])
                        self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)

                        mu_diff = torch.max(torch.sum(torch.abs(old_mu - self.layer_mean_field_parameters[index]["mu"])), mu_diff)
                    elif (index == 0):
                        old_mu = self.layer_mean_field_parameters[index]["mu"]

                        activation = torch.mm(dataset, self.layer_parameters[index]["W"]) + torch.mm(self.layer_mean_field_parameters[index+1]["mu"], self.layer_parameters[index+1]["W"].t())
                        self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)

                        mu_diff = torch.max(torch.sum(torch.abs(old_mu - self.layer_mean_field_parameters[index]["mu"])), mu_diff)
                    else:
                        old_mu = self.layer_mean_field_parameters[index]["mu"]

                        activation = torch.mm(self.layer_mean_field_parameters[index-1]["mu"], self.layer_parameters[index]["W"]) + torch.mm(self.layer_mean_field_parameters[index+1]["mu"], self.layer_parameters[index+1]["W"].t())
                        self.layer_mean_field_parameters[index]["mu"] = torch.sigmoid(activation)
                        
                        mu_diff = torch.max(torch.sum(torch.abs(old_mu - self.layer_mean_field_parameters[index]["mu"])), mu_diff)
                mf_step += 1
                if (mu_diff < 0.001):
                    print("Mean Field Converged")
                    break
            if (mf_step == mf_maximum_steps):
                print("Mean Field did not converge with maximum difference {}".format(mu_diff))
            
            # Update samples (subsample a subset of Markov Chains is excluded, it might need be added later for better performance)
            new_variables = []
            for index, _ in enumerate(self.layers):
                if (index == 0):
                    new_variables.append(self.sample_v(variables[index+1], self.layer_parameters[index]["W"]))
                elif (index == self.layers-1):
                    new_variables.append(self.sample_h(variables[index-1], self.layer_parameters[index]["W"]))
                else:  
                    new_variables.append(self.sample_h(variables[index-1], self.layer_parameters[index]["W"], variables[index+1], self.layer_parameters[index+1]["W"]))
            
            # Update model parameters
            alpha = 0.01
            for index, _ in enumerate(self.layers):
                if (index == 0):
                    self.layer_parameters[index]["W"] += alpha * (torch.mm(variables[index].t(), self.layer_mean_field_parameters[index]["mu"])/batch_size - torch.mm(new_variables[index].t(), new_variables[index+1])/batch_size)
                else:
                    self.layer_parameters[index]["W"] += alpha * (torch.mm(self.layer_mean_field_parameters[index-1]["mu"].t(), self.layer_mean_field_parameters[index]["mu"])/batch_size - torch.mm(new_variables[index].t(), new_variables[index+1])/batch_size)
            
            variables = new_variables
            alpha -= 0.001

    def fine_tune(self, x, top_level_latent_variables):
        pass

    def reconstructor(self, x):
        x_gen = []
        for _ in range(self.k):
            x_dash = x.clone()
            for i in range(len(self.layer_parameters)):
                _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]["W"], self.layer_parameters[i]["hb"])
            x_gen.append(x_dash)
        x_dash = torch.stack(x_gen)
        x_dash = torch.mean(x_dash, dim=0)

        y = x_dash

        y_gen = []
        for _ in range(self.k):
            y_dash = y.clone()
            for i in range(len(self.layer_parameters)-1, -1, -1):
                _, y_dash = self.sample_v(y_dash, self.layer_parameters[i]["W"], self.layer_parameters[i]["vb"])
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
                model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no//2]["W"])
                model[layer_no].bias = torch.nn.Parameter(self.layer_parameters[layer_no//2]["hb"])
        return model
    