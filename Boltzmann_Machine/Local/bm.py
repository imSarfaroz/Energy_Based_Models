import torch # main library
import torch.nn as NN # contains neural network building blocks
import torch.nn.functional as F # to gain access to neccessary functions
import torch.utils.data # provides tools for data loading.

class BM(nn.Module): # Here i am defining BM class from nn.Module, which the base class for all neural network modules in PyTorch
    def __init__(self, n_visible, n_hidden):
        super(BM, self).__init__()
        self.n_visible = n_visible  
        self.n_hidden = n_hidden 

       # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.1)  # inter-layer weights
        self.Wvv = nn.Parameter(torch.randn(n_visible, n_visible) * 0.1)  # Visible-visible weights
        self.Whh = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.1)  # Hidden-hidden weights
        self.b_visible = nn.Parameter(torch.zeros(n_visible))  # visible biases
        self.b_hidden = nn.Parameter(torch.zeros(n_hidden))  # hidden biases

    def sample_hidden(self, visible):
        # Computing the activations of the hidden units given the visible units.

        activation = torch.matmul(visible, self.W) + self.b_hidden  
        p_hidden = torch.sigmoid(activation)  
        sampled_hidden = torch.bernoulli(p_hidden) 

        return sampled_hidden

    def sample_visible(self, hidden):
        # Computing the activations of the visible units given the hidden units.

        activation = torch.matmul(hidden, self.W.t()) + self.b_visible 
        p_visible = torch.sigmoid(activation) 
        sampled_visible = torch.bernoulli(p_visible) 
        return sampled_visible

    def energy(self, visible, hidden):
        # Computing the energy of the current configuration of visible and hidden units.
        
        batch_size = visible.shape[0]
        # Computing the interaction terms
        energy = -torch.sum(torch.matmul(visible, self.W) * hidden, dim=1)
        energy -= 0.5 * torch.sum(torch.matmul(visible, self.Wvv) * visible, dim=1)
        energy -= 0.5 * torch.sum(torch.matmul(hidden, self.Whh) * hidden, dim=1)
        
        # Computing the bias terms
        energy -= torch.sum(visible * self.b_visible, dim=1)
        energy -= torch.sum(hidden * self.b_hidden, dim=1)
        return energy.mean()  # Returning the average energy over the batch

    def forward(self, visible):
        # Performing a forward pass to compute the activations of the hidden units.
        hidden = self.sample_hidden(visible)
        visible_gibbs = self.sample_visible(hidden)
        return visible, visible_gibbs