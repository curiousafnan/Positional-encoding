import torch
import math
from matplotlib import pyplot as plt

def PositionEncoding(max_length:int,dim:int)->torch.tensor:
    pos = torch.zeros(max_length,dim)
    max_pos = torch.arange(max_length).unsqueeze(1)
    fun = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    angle = max_pos * fun
    pos[:,0::2] = torch.sin(angle)
    pos[:,1::2] = torch.cos(angle)
    return pos

def heatmap(data):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='coolwarm')
    cbar = ax.figure.colorbar(im, ax=ax,)
    cbar.ax.set_ylabel("Scale Label", rotation=-90, va="bottom")
    ax.set_title("Heatmap of different frequencies")
    fig.tight_layout()
    plt.show()

def plot_frequencies(data, dims=[0, 4, 20, 60]):
    
    fig, ax = plt.subplots()
    im = ax.imshow(data,cmap='coolwarm')
    cbar = ax.figure.colorbar(im,ax=ax,)
    cbar.ax.set_ylabel("Scale Label", rotation=-90, va="bottom")
    ax.set_title("Heatmap of different frequencies")
    fig.tight_layout()
    plt.show()

def plot_frequencies(data, dims=[0, 4, 20, 60]):
    # Convert tensor to CPU for plotting
    data = data.detach().cpu()

    positions = torch.arange(data.shape[0])

    plt.figure(figsize=(14, 5))

    for d in dims:
        plt.plot(positions.numpy(), data[:, d].numpy(), label=f"Dim {d}")

    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.title("Positional Encoding Frequencies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

data = PositionEncoding(64,128)

heatmap(data)
plot_frequencies(data)

# print(PositionEncoding(5,8))

    