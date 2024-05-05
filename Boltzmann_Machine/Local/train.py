import torch
import torch.optim as optim
import numpy as np # numerical operations

def calculate_mse(original, reconstructed):
    return ((original - reconstructed) ** 2).mean()

def train(bm, train_loader, optimizer, epochs, device):
    # Training BM on MNIST datset

    bm.to(device)  # Move the model to the device (CPU/GPU)

    for epoch in range(epochs):
        total_mse = 0
        for batch, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            data = torch.bernoulli(data)

            # Performing CD training
            hidden = bm.sample_hidden(data)
            visible_recon = bm.sample_visible(hidden)

            # Calculating reconstruction error
            mse = calculate_mse(data, visible_recon)
            total_mse += mse.item()

            hidden_recon = bm.sample_hidden(visible_recon)
            positive_grad = bm.energy(data, hidden)
            negative_grad = bm.energy(visible_recon, hidden_recon)

            # Computing the gradients and updating the model parameters
            loss = positive_grad - negative_grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average MSE over all batches
        average_mse = total_mse / len(train_loader)
        print(f"Epoch: {epoch+1}/{epochs}, Avg MSE: {average_mse:.2f}")