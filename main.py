import torch
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.nn import functional as F


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(2, 64)
        self.l2 = nn.Linear(64, 2)

    def forward(self, x):
        h = self.l1(x)
        h = F.relu(h)
        h = self.l2(h)
        return h


class ReplayBuffer:

    def __init__(self, init_freq):
        self.init_freq = init_freq
        self._buffer = []

    def append(self, data):
        for d in data:
            self._buffer.append(d)

    def sample(self, shape):
        n = shape[0]
        if len(self._buffer) < n or self.init_freq > np.random.random(1):
            return np.random.uniform(-1, 1, shape).astype(np.float32)

        perm = np.random.permutation(len(self._buffer))
        res = [self._buffer[perm[i]] for i in range(n)]
        return np.array(res, dtype=np.float32)


class MCMCSampler:

    def __init__(
        self,
        model,
        num_sgld_step: int,
    ):
        self.model = model
        self.num_sgld_step = num_sgld_step

    def sample(self, X_sample):
        model = self.model
        device = next(model.parameters()).device
        X_sample.requires_grad_(True)
        for t in range(self.num_sgld_step):
            y_sample = model.forward(X_sample)
            energy = torch.logsumexp(y_sample, dim=1)
            grad, = torch.autograd.grad(energy.mean(), X_sample)
            noise = torch.from_numpy(
                np.random.randn(*X_sample.size()).astype(np.float32))
            noise = noise.to(device)
            X_sample = X_sample + grad + 0.1 * noise
        return X_sample.detach()


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    X = np.concatenate([
        np.random.randn(1000, 2) + 3,
        np.random.randn(1000, 2) - 3,
    ]).astype(np.float32)
    y = np.concatenate([
        np.zeros((1000,)),
        np.ones((1000,)),
    ]).astype(np.int64)
    n = len(X)
    device = torch.device('cuda')

    model = Model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    replay_buffer = ReplayBuffer(0.05)
    num_sgld_step = 30
    mcmc_sampler = MCMCSampler(model, num_sgld_step)
    batchsize = 128
    xent = nn.CrossEntropyLoss()
    for epoch in range(100):
        index = np.random.permutation(n)
        losses = []
        for start in range(0, n, batchsize):
            end = min(n, start+batchsize)
            X_batch = torch.from_numpy(X[index[start:end]]).to(device)
            y_batch = torch.from_numpy(y[index[start:end]]).to(device)

            optimizer.zero_grad()

            y_pred = model.forward(X_batch)
            xent_loss = xent(y_pred, y_batch)

            X_sample = torch.from_numpy(replay_buffer.sample(X_batch.size()))
            X_sample = X_sample.to(device)
            X_sample = mcmc_sampler.sample(X_sample)
            replay_buffer.append(X_sample.cpu().numpy())

            y_sample = model.forward(X_sample)

            gen_loss = (torch.logsumexp(
                y_pred, dim=1) - torch.logsumexp(y_sample, dim=1)).mean()
            loss = xent_loss - gen_loss
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()

        print(epoch, np.mean(losses))

        X_sample = torch.from_numpy(
            np.random.uniform(-1, 1, (100, 2)).astype(np.float32))
        X_sample = X_sample.to(device)
        X_sample = mcmc_sampler.sample(X_sample)
        X_sample = X_sample.cpu().numpy()
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        sns.kdeplot(X_sample[:, 0], X_sample[:, 1], ax=ax)
        ax.scatter(X_sample[:, 0], X_sample[:, 1], s=1)
        ax.set_title('Joint Distribution')
        path = Path('results')/f'{epoch:03d}.png'
        path.parent.mkdir(exist_ok=True)
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    main()
