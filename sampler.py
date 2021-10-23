import numpy as np
from torch.utils import data
from torch import nn
import torch
import tqdm
from function import init_weights, calc_mean_std

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
            nn.Conv2d(32, 32, 5,stride=3),
            nn.LeakyReLU(),
        )
        self.project_in = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.latent_decompress = nn.Sequential(
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
        out = x.clone()
        size = out.size()
        content_mean, content_std = calc_mean_std(out)

        out = (out - content_mean.expand(
            size)) / content_std.expand(size)
        out = self.latent_compress(out).flatten(1)
        out = self.project_in(out).reshape(b, 32, 2, 2)
        out = self.latent_decompress(out)
        loss = self.loss(out, x)
        return loss


class SimilarityRankedSampler(data.sampler.Sampler):
    def __init__(self, data_source, style_batch, tmp_dataset, tmp_dataset2,encoder,r=10000):
        self.num_samples = len(data_source)
        style_feats = encoder(style_batch)['r4_1']
        device = torch.device('cuda')
        latent_model = LatentClustering().to(device)
        init_weights(latent_model)
        optimizer = torch.optim.Adam(latent_model.parameters(), lr=1e-4)
        loss = latent_model(style_feats)
        loss.backward()
        optimizer.step()
        for i in tqdm.tqdm(range(r)):
            x = next(tmp_dataset).to(device)
            x = encoder(x)
            loss = latent_model(x['r4_1'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0 and i !=0:
                print(loss.item())
        style_latent = latent_model.project_down(style_feats)
        self.similarity=[]
        print('measuring similarity')
        cosine_sim = nn.CosineSimilarity()
        for i in tqdm.tqdm(range(self.num_samples//8)):
            x = next(tmp_dataset2).to(device)
            x = encoder(x)
            c_latent = latent_model.project_down(x['r4_1'])
            self.similarity.append(cosine_sim(c_latent,style_latent).detach().cpu().numpy())
        self.similarity = np.concatenate(self.similarity,axis=0)
        top_similar = self.similarity.argsort()[-120:][::-1]
        top_similar = np.hstack(top_similar)
        self.i=0
        self.current_subset = top_similar

    def __iter__(self):
        self.i += 1
        if self.i == 1000:
            self.expand_subset(500)
        if self.i == 2500:
            self.expand_subset(750)
        if self.i == 7500:
            self.expand_subset(2000)
        if self.i == 10000:
            self.expand_subset(3500)
        if self.i == 20000:
            self.expand_subset(5000)
        if self.i <= 30000:
            return iter(SubsetSampler(self.current_subset))
        else:
            return iter(InfiniteSampler(self.num_samples))

    def expand_subset(self, n):
        top = n//8
        top_similar = []
        for i in range(self.similarity.shape[1]):
            top_similar.append(self.similarity[:,i].argsort()[-top:][::-1])
        self.current_subset = np.hstack(top_similar)

    def __len__(self):
        return self.current_subset
