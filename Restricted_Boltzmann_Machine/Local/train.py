import torch.optim as optim # for gaining access to optimizers
import numpy as np # numerical operations


# Defining my train function which takes the model (RBM), trainer_loader (data loader for training data)
# num_epochs (number of epochs to train for, default = 1) and and lr (learning rate, default = 0.01)
def train(model, train_loader, n_epochs=1, learning_rate=0.01):

    # optimizer
    train_op = optim.Adam(model.parameters(), learning_rate)

    model.train()

    # A loop specified number of epochs, updating weights using gradient descent
    # to minimize the difference in free energy between original and reconstructed images, and prints the average loss for each epoch
    
    for epoch in range(n_epochs):
        loss_ = []
        for _, (data, target) in enumerate(train_loader):
            v, v_gibbs = model(data.view(-1, 784))
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        print('Epoch #%d\t ||| Loss=%.2f' % (epoch, np.mean(loss_)))

    return model