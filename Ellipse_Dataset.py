import torch
from torch.utils.data import Dataset, DataLoader
import math
from matplotlib import pyplot as plt


def generate_training_data(N, a, b, noise_scale=0.1, plot_data=True):
    # prepare the training dataset
    c = math.sqrt(a*a - b*b)
    xHohmann = torch.linspace(-(a+c), (a-c), N)
    yHohmann = b * torch.sqrt(1.0-(xHohmann+c)**2/a**2)
    yNoise = yHohmann + (torch.rand(yHohmann.shape) - 0.5) * noise_scale

    xEarth = torch.linspace(-(a-c), (a-c), N)
    yEarth = torch.sqrt((a-c)**2-(xEarth)**2)
    xEarth = torch.cat((xEarth, torch.flip(xEarth,[0])))
    yEarth = torch.cat((yEarth, torch.flip(-yEarth,[0])))

    xMars = torch.linspace(-(a+c), (a+c), N)
    yMars = torch.sqrt((a+c)**2-(xMars)**2)
    xMars = torch.cat((xMars, torch.flip(xMars,[0])))
    yMars = torch.cat((yMars, torch.flip(-yMars,[0])))

    if plot_data:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(xHohmann, yNoise, color="green", s=0.5)
        plt.plot(xEarth, yEarth, color="blue", linewidth=1.0, linestyle="--")
        plt.plot(xMars, yMars, color="red", linewidth=1.0, linestyle="--")
        plt.plot(0.0, 0.0, 'yo', label='Sun', markersize=8)
        plt.text(0.05, -0.1, 'Sun', color='black')
        plt.plot(a-c, 0.0, 'bo', label='Earth',  markersize=4)
        plt.text(a-c+0.05, -0.1, 'Earth', color='blue')
        plt.plot(-(a+c), 0.0, 'ro', label='Mars',  markersize=3)
        plt.text(-(a+c)+0.05, -0.1, 'Mars', color='red')
        plt.text(1.0, -1.75, 'in unit of AU', color='black')
        plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=1.0)
        plt.axvline(x=0.0, color='gray', linestyle='--', linewidth=1.0)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        ax.set_aspect('equal', adjustable='box')

        # ellipse in polar coordinates
        e = c / a
        Thetas = torch.linspace(0, math.pi, N)
        R = a * (1 - e**2) / (1 + e * torch.cos(Thetas))
        xPolar = R * torch.cos(Thetas)
        yPolar = R * torch.sin(Thetas)
        # plt.plot(xPolar, yPolar, color="black", linewidth=1.0, linestyle="-")

        N0 = 2000
        theta = 0.0
        Xs = [1.0]
        Ys = [0.0]

        for i in range(N0):
            r = a * (1 - e ** 2) / (1 + e * math.cos(theta))
            d_theta = math.pi * b / a / N0 / ((1-e*e)/(1+e*math.cos(theta)))**2
            theta += d_theta
            newX = r * math.cos(theta)
            newY = r * math.sin(theta)
            Xs.append(newX)
            Ys.append(newY)

        # plt.plot(Xs, Ys, color="blue", linewidth=1.0, linestyle="-")
        plt.show(block=False)

    return (xHohmann, yNoise)


class EllipseDataset(Dataset):
    """Ellipse dataset."""
    def __init__(self, N, A, B, noise_scale=0.1, transform=None):
        """
        Args:
            N: number of samples of the points on the ellipse.
            A: the semimajor axis of the ellipse
            B: the semiminor axis of the ellipse
            noise_scale: scaling factor for the noise to add.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.Xs, self.Ys = generate_training_data(N, A, B, noise_scale, plot_data=True)
        self.transform = transform

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pntX, pntY = self.Xs[idx], self.Ys[idx]
        sample = {'x': pntX, 'y': pntY}

        if self.transform:
            sample = self.transform(sample)

        return sample
