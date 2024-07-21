import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors
from wbw_translation.clusters import Clusterizer
from tqdm.auto import tqdm
import wandb
from IPython.display import clear_output
from wbw_translation.models import LinearModel
from visualizations import plot_embeddings

from typing import Optional, List, Dict, Any



class ICP:
    def __init__(self,
                 x_embeds: torch.Tensor, 
                 y_embeds: torch.Tensor,
                 cluster: bool = False, 
                 x_clusters: Optional[Clusterizer] = None,
                 y_clusters: Optional[Clusterizer] = None,
                 device: str = 'cpu',
                 save_path: Optional[str] = None):
        """
        ICP initialization with source and target embeddings and possible specification of clusters

        Args:
            x_embeds (torch.Tensor): source embeddigns
            y_embeds (torch.Tensor): target embeddigns
            cluster (bool, optional): if True, nearest neighbors 
                will be found only in the corresponding clusters. Defaults to False.
            x_clusters (Clusterizer, optional): source clusters. Defaults to None.
            y_clusters (Clusterizer, optional): target clusters. Defaults to None.
            device (str, optional): device on which the models will be trained. 
                If gpu, neighbors search will be performed via torch.dist. 
                If cpu, search is via sklearn.neighbors. Defaults to 'cpu'.
        """
        
        self.x_embeds = x_embeds
        self.y_embeds = y_embeds
        self.cluster = cluster
        self.device = device
        if self.cluster:
            assert (x_clusters is not None) and (y_clusters is not None), \
            "Provide source and target clusters"
            self.x_clusters = x_clusters
            self.y_clusters = y_clusters

        self.save_path = save_path

    def train_epoch(self,
                    model: nn.Module,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    batch_size: int = 128,
                    cut: Optional[int] = None,
                    epoch: int = 1

            ) -> float:
        """
        Single train epoch

        Args:
            model (nn.Module): word-by-word translation model
            criterion (nn.Module): loss function.
            optimizer (torch.optim.Optimizer): optimization algorithm
            scheduler (torch.optim.lr_scheduler.LRScheduler): learning rate scheduler
            batch_size (int, optional): size of a signle batch of embeddings. 
                Defaults to 128.
            cut (Optional[int], optional): if specified, only cut number of first
                embeddings will be used in this epoch. Defaults to None.
            epoch (int, optional): number of the current epoch. 
                Is used for possible dynamic batch size. Defaults to 1.

        Returns:
            float: average distance between mapped embeddings in both directions
        """
        
        if cut:
            x_embeds, y_embeds = self.x_embeds[:cut], self.y_embeds[:cut]
        else:
            x_embeds, y_embeds = self.x_embeds, self.y_embeds

        x_mapped, y_mapped = model(x_embeds, y_embeds)
        if self.device == 'cpu':
            # nns_x = NearestNeighbors(n_neighbors=1).fit(y_mapped.detach())
            # nns_y = NearestNeighbors(n_neighbors=1).fit(x_mapped.detach())
            # indices_x = nns_x.kneighbors(x_embeds)[1].flatten()
            # indices_y = nns_y.kneighbors(y_embeds)[1].flatten()
            dist_x_to_y = torch.cdist(x_embeds.double(), y_mapped.double())
            dist_y_to_x = torch.cdist(y_embeds.double(), x_mapped.double())
            indices_x = dist_x_to_y.topk(1, largest=False, dim=1)[1].flatten()
            indices_y = dist_y_to_x.topk(1, largest=False, dim=1)[1].flatten()
        else:
            dist_x_to_y = torch.cdist(x_embeds.double(), y_mapped.double())
            dist_y_to_x = torch.cdist(y_embeds.double(), x_mapped.double())
            indices_x = dist_x_to_y.topk(1, largest=False, dim=1)[1].flatten()
            indices_y = dist_y_to_x.topk(1, largest=False, dim=1)[1].flatten()

        # y's that are closest ot x's & x's that are closest to y's
        yfx = y_embeds[indices_x]
        xfy = x_embeds[indices_y]

        # reconstruction loss
        rec_x = ((x_embeds - y_mapped[indices_x])**2).sum(1).mean()
        rec_y = ((y_embeds - x_mapped[indices_y])**2).sum(1).mean()
        reconstruct_loss = rec_x + rec_y

        # divide embeds into random batches
        n_steps = len(x_embeds) // batch_size
        idx_x = np.random.permutation(len(x_embeds))
        idx_y = np.random.permutation(len(x_embeds))

        total_loss = 0
        for i in range(n_steps):

            optimizer.zero_grad()
            idx_cur = i*batch_size + np.arange(batch_size)
            x_batch = x_embeds[idx_x[idx_cur]]
            y_batch = y_embeds[idx_y[idx_cur]]
            x_pred, y_pred = model(x_batch, y_batch)
            x_target, y_target = yfx[idx_x[idx_cur]], xfy[idx_y[idx_cur]]
            x_cycle, y_cycle = model.cycle(x_batch, y_batch)
            loss_map = criterion(x_pred, x_target) + criterion(y_pred, y_target)

            loss_cycle = criterion(x_batch, x_cycle) + criterion(y_batch, y_cycle)
            loss = loss_map + 0.05 * loss_cycle
            loss.backward()
            optimizer.step()
            total_loss += loss_map.item()

        if scheduler is not None:
            scheduler.step()

        return reconstruct_loss.detach().item()

    def train_iteration(self,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler.LRScheduler, 
                        num_epochs: int, 
                        batch_size: int,
                        plot: bool = False,
                        log: bool = False,
                        early_stopping: bool = False,
                        random_state: int = 42
                    ) -> List[float]:
        """
        Full training

        Args:
            model (nn.Module): word-by-word translation model.
            optimizer (torch.optim.Optimizer): optimization algorithm.
            criterion (nn.Module): loss function.
            scheduler (torch.optim.lr_scheduler.LRScheduler): learning rate scheduler.
            batch_size (int, optional): size of a signle batch of embeddings. 
            num_epochs (int): number of training epochs
            plot (bool, optional): if True, loss will be plotted after each epoch. Defaults to False.
            log (bool, optional): if True, loss will be logged to wandb after each epoch. Defaults to False.
            early_stopping (bool, optional): if True, training will be stopped if loss hasn't changed
                in the last 100 epochs. Defaults to False.
            random_state (int, optional): specified seed for training. Defaults to 42.

        Returns:
            List[float]: list of losses at each epoch
        """
        np.random.seed(random_state)

        model = model.to(self.device)
        x_embeds, y_embeds = self.x_embeds.to(self.device), self.y_embeds.to(self.device)
        criterion = nn.MSELoss()
        losses = [0.]
        pbar = tqdm(range(num_epochs), leave=False)
        time.sleep(3)
        for epoch in pbar:
            pbar.set_description(desc=f'{losses[-1]:.2f}')
            loss = self.train_epoch(model,
                                    criterion,
                                    optimizer, 
                                    scheduler, 
                                    batch_size,
                                    epoch=epoch)
            losses += [loss]

            if plot:
                assert x_embeds.shape[1] == 2 and y_embeds.shape[1] == 2, \
                 "Visualization is available only for 2d embeddings"
                clear_output(wait=True)
                x_pred, _ = model(x_embeds, y_embeds)

                fig, ax = plt.subplots(2, 1, figsize=(8,5), gridspec_kw={'height_ratios': [5, 2]})
                ax[0].scatter(x_pred.detach()[:,0], x_pred.detach()[:,1], color='blue',
                         s=5, alpha=0.5, label='source embeds')
                ax[0].scatter(y_embeds.detach()[:,0], y_embeds.detach()[:,1], color='red',
                         s=5, alpha=0.5, label='target embeds')
                ax[0].legend()
                ax[0].set_xlim(y_embeds.detach()[:,0].min() * .9, 
                           y_embeds.detach()[:,0].max() * 1.1)
                ax[0].set_ylim(y_embeds.detach()[:,1].min() * .9, 
                           y_embeds.detach()[:,1].max() * 1.1)

                ax[1].plot(np.arange(epoch+1), losses[1:], color='grey', label='loss')
                ax[1].legend()
                plt.show()

                if epoch == 0:
                    time.sleep(3)


            if log:
                wandb.log({'epoch': epoch + 1, 'loss': loss})

            # successful run won't be stuck in local minimum
            if early_stopping and epoch > 101 and losses[-100] / losses[-1] < 1:
                return losses[1:]

        if self.save_path:
            os.mkdirs(self.save_path, exists_ok=True)
            torch.save(model.state_dict(), self.save_path)

        return losses[1:]


    def run_sweep(self, 
                  sweep_config: Dict[str, Any], 
                  count: int = 1000
            ) -> None:
        """
        Run a W&B sweep.

        Args:
            sweep_config (Dict[str, Any]): Configuration for the W&B sweep.
            count (int, optional): Number of sweep iterations to run. Defaults to 1.
        """
        sweep_id = wandb.sweep(sweep_config, project='wbw-translation')
        wandb.agent(sweep_id, function=self.sweep_train_iteration, count=count)


    def sweep_train_iteration(self) -> None:
        """
        Wrapper for train_iteration to use with W&B sweep.
        """
        model = LinearModel(shape=300)
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
        scheduler = None

        self.train_iteration(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=wandb.config.num_epochs,
            batch_size=wandb.config.batch_size,
            verbose=True,
            random_state=wandb.config.random_state
        )
