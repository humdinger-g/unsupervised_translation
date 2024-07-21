import torch
from sklearn.decomposition import PCA
from wbw_translation.load_data import load_data
from utils import norm_sort, translate_sample, get_procrustes_mapping, get_same_words
from wbw_translation.models import LinearModel
from wbw_translation.icp import ICP
from tqdm.auto import tqdm

import wandb
import argparse
import yaml

def train_icp(config=None):
    with wandb.init(config=config):
        x_embeds, x_words, y_embeds, y_words = load_data('fasttext_data', wandb.config.source_lang, wandb.config.target_lang)
        x_embeds = x_embeds[:wandb.config.cut]
        y_embeds = y_embeds[:wandb.config.cut]
        x_embeds -= x_embeds.mean(0)
        y_embeds -= y_embeds.mean(0)
        x_norms = norm_sort(x_embeds)[:5000]
        y_norms = norm_sort(y_embeds)[:5000]


        icp = ICP(x_norms, y_norms, device=wandb.config.device)

        if wandb.config.procrustes_init == 'norm':
            Wxy, Wyx = get_procrustes_mapping(x_norms[:1000], y_norms[:1000])
            model = LinearModel(300, init=(Wxy.T, Wyx.T))
        elif wandb.config.procrustes_init == 'same':
            X, Y = get_same_words(x_embeds, y_embeds, x_words[:wandb.config.cut], y_words[:wandb.config.cut])
            Wxy, Wyx = get_procrustes_mapping(X, Y)
            model = LinearModel(300, init=(Wxy.T, Wyx.T))
        else:
            model = LinearModel(300)

        #model = LinearModel(300)
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
        
        # Train the model
        losses = icp.train_iteration(
            model,
            optimizer,
            scheduler=None if wandb.config.scheduler is None else \
                torch.optim.lr_scheduler.StepLR(optimizer, step_size=80,
                                                gamma=0.5),
            num_epochs=wandb.config.num_epochs,
            batch_size=wandb.config.batch_size,
            plot=False,
            log=True,
            random_state=wandb.config.random_state
        )
        
        wandb.log({
            'loss': losses[-1],
            'translation': translate_sample(model, wandb.config.source_lang, wandb.config.target_lang, seed=123)
        })

def main():
    parser = argparse.ArgumentParser(description='Run ICP with specified parameters.')
    parser.add_argument('source_lang', type=str, help='Source language')
    parser.add_argument('target_lang', type=str, help='Target language')
    parser.add_argument('--config', type=str, default='configs/sweep_config.yaml', help='Path to sweep configuration YAML file')

    args = parser.parse_args()

    with open(args.config) as file:
        sweep_config = yaml.safe_load(file)

    sweep_config['parameters']['source_lang'] = {'value': args.source_lang}
    sweep_config['parameters']['target_lang'] = {'value': args.target_lang}
    sweep_config['parameters']['device'] = {'value': 'cpu'}

    sweep_id = wandb.sweep(sweep_config, project=f"translation from {args.source_lang} to {args.target_lang}")
    print("wandb.config after init:", wandb.config)
    wandb.agent(sweep_id, function=train_icp)

if __name__ == "__main__":
    main()
