import numpy as np
from torch.utils import data
from torch import nn
import torch
import tqdm
from function import init_weights

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

def SubsetSampler(subset):
    # i = 0
    n = len(subset)
    i = n - 1
    order = np.random.permutation(subset)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def SequentialSampler(n):
    i = 0
    order = np.arange(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            i = 0

class SequentialSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(SequentialSampler(self.num_samples))

    def __len__(self):
        return self.num_samples

class LatentClustering(nn.Module):
    def __init__(self):
        super(LatentClustering, self).__init__()
        self.latent_compress = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 8),
            nn.LeakyReLU(),
        )
        self.project_in = nn.Sequential(
            nn.Linear(32, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 8),
            nn.LeakyReLU()
        )
        self.project_out = nn.Sequential(
            nn.Linear(8, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
        )
        self.latent_decompress = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 2,2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4,4
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8,8
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16,16
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, 3)
        )
        self.loss = nn.MSELoss()

    def project_down(self, x):
        x = self.latent_compress(x).flatten(1)
        x = self.project_in(x)
        return x

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.latent_compress(x).flatten(1)
        x = self.project_in(x)
        out = self.project_out(x).reshape(b, 32, 1, 1)
        out = self.latent_decompress(out)
        loss = self.loss(out, x)
        return loss


class SimilarityRankedSampler(data.sampler.Sampler):
    def __init__(self, data_source, style_batch, tmp_dataset, tmp_dataset2,encoder,r=20000):
        self.num_samples = len(data_source)
        style_feats = encoder(style_batch)['r4_1']
        device = torch.device('cuda')
        latent_model = LatentClustering().to(device)
        init_weights(latent_model)
        optimizer = torch.optim.Adam(latent_model.parameters(), lr=1e-4)
        loss = latent_model(style_feats)
        loss.backward()
        optimizer.step()
        for i in range(r):
            x = next(tmp_dataset).to(device)
            x = encoder(x)
            loss = latent_model(x)
            loss.backward()
            optimizer.step()
            if i%10 == 0 and i !=0:
                print(loss.item())
        style_latent = latent_model.project_down(style_feats)
        self.similarity=[]
        print('measuring similarity')
        for i in tqdm.tqdm(tmp_dataset2):
            x = encoder(i.to(device))
            c_latent = latent_model.project_down(x)
            self.similarity.append(torch.abs(style_latent-c_latent).cpu().numpy())
        self.similarity = np.concatenate(self.similarity,axis=0)
        top_similar = []
        for i in range(self.similarity.shape[1]):
            top_similar.append(self.similarity[:,i].argsort()[-10:])
        top_similar = np.hstack(top_similar)
        self.i=0
        self.current_subset = top_similar

    def __iter__(self):
        self.i += 1
        if i == 1000:
            expand_subset(1000)
        if i == 2500:
            expand_subset(2000)
        if i == 7500:
            expand_subset(5000)
        if i == 10000:
            expand_subset(10000)
        if i == 20000:
            expand_subset(20000)
        if i <= 30000:
            return iter(SubsetSampler(self.current_subset))
        else:
            return iter(InfiniteSampler(self.num_samples))

    def expand_subset(self, n):
        top = n//8
        top_similar = []
        for i in range(self.similarity.shape[1]):
            top_similar.append(self.similarity[:,i].argsort()[-top:])
        self.current_subset = np.hstack(top_similar)

    def __len__(self):
        return self.current_subset
