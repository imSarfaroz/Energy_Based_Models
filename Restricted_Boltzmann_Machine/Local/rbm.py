import torch # main library
import torch.nn as NN # contains neural network building blocks
import torch.nn.functional as F # to gain access to neccessary functions
import torch.utils.data # provides tools for data loading.

class RBM(NN.Module): # Here i am defining RBM class from nn.Module, which the base class for all neural network modules in PyTorch

    def __init__(self, num_visible=784, num_hidden=128, k=1):
        # The constructor for the RBM class. Initializing my class specified number of visible
        # units num_visible = 784 (taking into considiration MNIST dataset consisting of 28x28 pixels)
        # hidden units num_hidden = 200 (ensuring its fair complexity)
        # Gibbs sampling steps (k) = 1

        super(RBM, self).__init__() 
        self.visible = NN.Parameter(torch.randn(1, num_visible)) 
        self.hidden = NN.Parameter(torch.randn(1, num_hidden))  
        self.W= NN.Parameter(torch.randn(num_hidden, num_visible)) 
        self.k = k

    def visible_to_hidden(self, visible):
        # Defining a method to sample hidden units given visible units using the conditional probability

        p = torch.sigmoid(F.linear(visible, self.W, self.hidden))
        # Computing the probability p that each hidden unit is activated, given the visible units v. This is done using a sigmoid activation applied to the linear transformation of v by weights W and biases h.

        return p.bernoulli() # Sampling from the Bernoulli distribution determined by p to generate binary states for the hidden units.

    def hidden_to_visible(self, hidden):
        # Defining a method to sample visible units given hidden units using the conditional probability

        p = torch.sigmoid(F.linear(hidden, self.W.t(), self.visible))
        # Computing the probability  p p that each visible unit is activated, given the hidden units h. This is achieved using a sigmoid function applied to the linear transformation of h by the transpose of weights W and biases v.

        return p.bernoulli() # Sampling from the Bernoulli distribution as determined by p to generate binary states for the visible units.

    def free_energy(self, visible):
        #Free energy function. A method to compute the free energy of a visible state v, 
        # which is essential for computing gradients during training. Formula was taken from the third-party articles&books

        v_term = torch.matmul(visible, self.visible.t())
        w_x_h = F.linear(visible, self.W, self.hidden)
        h_term = torch.sum(F.softplus(w_x_h), dim=1)
        return torch.mean(-h_term - v_term)

    def forward(self, visible):
       # method which computes the visible states after k steps of Gibbs sampling.

        hidden = self.visible_to_hidden(visible)
        for _ in range(self.k):
            v_gibb = self.hidden_to_visible(hidden)
            hidden = self.visible_to_hidden(v_gibb)
        return visible, v_gibb
