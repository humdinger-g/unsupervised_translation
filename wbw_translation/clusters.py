from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm.auto import tqdm
from wbw_translation.anisotropy import calculate_anisotropy_torch, intrinsic_dimension

from typing import Optional, Tuple


class Clusterizer:
    def __init__(self, 
                 embeds: torch.Tensor, 
                 num_clusters: int, 
                 sort: Optional[str], 
                 rs: int = 42, 
                 translations: Optional[np.ndarray] = None):
        
        self.rs = rs
        self.sort = sort
        self.num_clusters = num_clusters
        self.embeds = embeds
        self.labels = KMeans(n_clusters=num_clusters, random_state=self.rs, algorithm='elkan').fit(embeds).labels_
        self.clusters = [embeds[np.where(self.labels == i)] for i in range(num_clusters)]
        if self.sort == 'count':
            self.clusters = sorted(self.clusters, key=len)
            self.values = np.array([len(x) for x in self.clusters])
        elif self.sort == 'norm':
            self.clusters = sorted(self.clusters, key=lambda x: torch.norm(x[:100]))
            self.values = np.array([torch.norm(x) for x in self.clusters])
        elif self.sort == 'dist':
            distances = []
            for i in range(self.num_clusters):
                nn_cur = NearestNeighbors(n_neighbors=10).fit(self.clusters[i])
                distances += [nn_cur.kneighbors(self.clusters[i])[0].flatten().mean()]
            idx = np.argsort(distances)
            self.clusters = [self.clusters[j] for j in idx]
            self.values = np.array((sorted(distances)))
        elif self.sort == 'len':
            cluster2len = {}
            for i in range(self.num_clusters):
                idx = np.where(self.labels == i)
                words = translations[idx]
                lengths = [len(x) for x in words]
                cluster2len[i] = np.mean(lengths[:100])
            idx = np.argsort(list(cluster2len.values()))
            self.clusters = [self.clusters[j] for j in idx]
            self.values = np.array(sorted(list(cluster2len.values())))
        elif self.sort == 'full':
            counts = np.array([len(x) for x in self.clusters])
            norms = np.array([torch.norm(x) for x in self.clusters])
            dists = []
            for i in range(self.num_clusters):
                nn_cur = NearestNeighbors(n_neighbors=10).fit(self.clusters[i])
                dists += [nn_cur.kneighbors(self.clusters[i])[0].flatten().mean()]
            cluster2len = {}
            for i in range(self.num_clusters):
                idx = np.where(self.labels == i)
                words = translations[idx]
                lengths = [len(x) for x in words]
                cluster2len[i] = np.mean(lengths[:100])
            lens = np.array(list(cluster2len.values()))
            dists = np.array(dists)
            scores = counts * norms * dists * lens
            idx = np.argsort(scores)
            self.clusters = [self.clusters[j] for j in idx]
            self.values = np.array(sorted(scores))
        elif sort == 'anisotropy':
            # anisotropies = 
            # self.clusters = sorted(self.clusters, key=lambda x: calculate_anisotropy_torch(x))
            # self.values = np.array([calculate_anisotropy_torch(x) for x in self.clusters])
            anisotropies = [calculate_anisotropy_torch(x) for x in self.clusters]
            self.clusters = [self.clusters[x] for x in np.argsort(anisotropies)]
            self.values = np.array(sorted(anisotropies))

        elif sort == 'dimension':
            dimensions = [intrinsic_dimension(x) for x in self.clusters]
            self.clusters = [self.clusters[x] for x in np.argsort(dimensions)]
            self.values = np.array(sorted(dimensions))



        else:
            self.values = None

    def __getitem__(self, idx):
        return self.clusters[idx]
    
    def get_centers(self, median=False):
        if not median:
            centers = torch.vstack([
                self.clusters[i].mean(0) for i in range(self.num_clusters)
            ])
        else:
            centers = torch.vstack([
                self.clusters[i].median(0)[1] for i in range(self.num_clusters)
            ])

        return centers



    def plot(self, clusters=None):
        if not clusters:
            clusters = np.arange(self.num_clusters)
        for i, x in enumerate(self.clusters):
            if i not in clusters:
                continue
            proj = PCA(n_components=2, random_state=self.rs).fit_transform(x)
            plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
        plt.show()


def translate_cluster(cluster, embeds, translations, sample_size, top=False):
    n = len(cluster)
    words = []
    if top:
        idx = np.arange(sample_size)
    else:
        idx = np.random.choice(np.arange(n), size=sample_size, replace=False)
    for i in range(sample_size):
        word = torch.topk(torch.cosine_similarity(cluster[idx[i]], embeds), 1)[1].item()
        words += [translations[word]]
    return words


def alignment_score(num_clusters, x_clusters, y_clusters):
    distance, diversity = 0, 0
    for i in range(num_clusters):
        nns_x = NearestNeighbors(n_neighbors=1).fit(x_clusters[i])
        nns_y = NearestNeighbors(n_neighbors=1).fit(y_clusters[i])
        neighbors_to_x = nns_y.kneighbors(x_clusters[i])
        neighbors_to_y = nns_x.kneighbors(y_clusters[i])
        distance += (neighbors_to_x[0].mean() + neighbors_to_y[0].mean()) / 2
        diversity += len(set(neighbors_to_x[1].flatten())) / len(neighbors_to_x[1].flatten())
        diversity += len(set(neighbors_to_y[1].flatten())) / len(neighbors_to_y[1].flatten())

    score = diversity / distance
    return score / num_clusters


def find_best_clusterization(x_embeds: torch.Tensor,
                             y_embeds: torch.Tensor, 
                             num_clusters: int, 
                             sort: Optional[str], 
                             num_runs: int = 25
                    ) -> Tuple[Clusterizer, Clusterizer]:
    results = {}
    for rs in tqdm(range(1, num_runs+1)):
        x_clusters = Clusterizer(x_embeds, num_clusters, sort, rs=rs)
        y_clusters = Clusterizer(y_embeds, num_clusters, sort, rs=rs)
        results[rs] = np.linalg.norm(x_clusters.values - y_clusters.values)
    
    best_rs = np.argmin(list(results.values())) + 1

    x_clusters = Clusterizer(x_embeds, num_clusters, sort, rs=best_rs)
    y_clusters = Clusterizer(y_embeds, num_clusters, sort, rs=best_rs)
    return x_clusters, y_clusters

    