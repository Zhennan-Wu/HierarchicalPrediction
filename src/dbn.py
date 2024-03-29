import torch
from rbm import RBM


class DBN:
    """
    Deep Belief Network
    """
    def __init__(self, input_size, layers, mode="bernoulli", gpu=False, k=5, savefile=None):
        self.input_size = input_size
        self.layers = layers
        self.layer_parameters = [{"W":None, "hb":None, "vb":None} for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.savefile = savefile
    
    def sample_v(self, y, W, vb):
        """
        Sample visible units given hidden units
        """
        wy = torch.mm(y, W)
        activation = wy + vb
        p_v_given_h = torch.sigmoid(activation)
        if (self.mode == "bernoulli"):
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        elif (self.mode == "gaussian"):
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape, device=self.device))
        else:
            raise ValueError("Invalid mode")
    
    def sample_h(self, x, W, hb):
        """
        Sample hidden units given visible units
        """
        wx = torch.mm(x, W.t())
        activation = wx + hb
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
                    _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]["W"], self.layer_parameters[i]["hb"])
                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
            return x_dash
    
    def train(self, x):
        """
        Train DBN
        """
        for index, _ in enumerate(self.layers):
            if (index == 0):
                vn = self.input_size
            else:
                vn = self.layers[index-1]
            hn = self.layers[index]

            rbm = RBM(vn, hn, epochs=100, mode="bernoulli", lr=0.0005, k=10, batch_size=128, gpu=True, optimizer="adam", early_stopping_patient=10)
            x_dash = self.generate_input_for_layer(index, x)
            rbm.train(x_dash)
            self.layer_parameters[index]["W"] = rbm.weights
            self.layer_parameters[index]["hb"] = rbm.hidden_bias
            self.layer_parameters[index]["vb"] = rbm.visible_bias
            print("Finished Training Layer", index, "to", index+1)
        if (self.savefile is not None):
            torch.save(self.layer_parameters, self.savefile)

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
    