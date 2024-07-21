import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from wbw_translation.load_data import load_data
from utils import norm

from typing import Tuple

sns.set_theme()


def get_2d_embeds(x_embeds: torch.Tensor,
                  y_embeds: torch.Tensor,
        seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCA dimensionality reduction to 2d"""

    x2d = PCA(n_components=2, random_state=seed).fit_transform(x_embeds)
    y2d = PCA(n_components=2, random_state=seed).fit_transform(y_embeds)

    return x2d, y2d


# def plot_embeddings(x_embeds, y_embeds, x_words, y_words,
#                     source_lang='source', target_lang='target'):
#     x_scatter = go.Scatter(
#         x=x_embeds[:, 0],
#         y=x_embeds[:, 1],
#         mode='markers',
#         marker=dict(color='blue'),
#         text=x_words,
#         opacity=0.3,
#         name=f'{source_lang} embeds'
#     )

#     y_scatter = go.Scatter(
#         x=y_embeds[:, 0],
#         y=y_embeds[:, 1],
#         mode='markers',
#         marker=dict(color='red'),
#         text=y_words,
#         opacity=0.3,
#         name=f'{target_lang} embeds'
#     )

#     fig = go.Figure(data=[x_scatter, y_scatter])

#     fig.update_layout(
#         title="Embeddings",
#         xaxis_title="",
#         yaxis_title="",
#         hovermode='closest'
#     )

#     fig.show()




def plot_embeddings(x_embeds, y_embeds, x_words, y_words,
                    source_lang='source', target_lang='target', show_labels=False):
    x_scatter = go.Scatter(
        x=x_embeds[:, 0],
        y=x_embeds[:, 1],
        mode='markers+text' if show_labels else 'markers',
        marker=dict(color='blue'),
        text=x_words,
        textposition='top center' if show_labels else 'top center',
        opacity=0.3,
        name=f'{source_lang} embeds'
    )

    y_scatter = go.Scatter(
        x=y_embeds[:, 0],
        y=y_embeds[:, 1],
        mode='markers+text' if show_labels else 'markers',
        marker=dict(color='red'),
        text=y_words,
        textposition='top center' if show_labels else 'top center',
        opacity=0.3,
        name=f'{target_lang} embeds'
    )

    fig = go.Figure(data=[x_scatter, y_scatter])

    fig.update_layout(
        title="Embeddings",
        xaxis_title="",
        yaxis_title="",
        hovermode='closest'
    )

    fig.show()





def plot_k_embeddings(source_lang: str,
                      target_lang: str,
                      do_norm: bool = False,
                      k: int = 100) -> None:
    x_embeds, x_words, y_embeds, y_words = load_data(
        'fasttext_data', source_lang, target_lang
    )

    if do_norm:
        x_idx = np.argsort(torch.norm(norm(x_embeds[:1618]), dim=1))
        y_idx = np.argsort(torch.norm(norm(y_embeds[:1618]), dim=1))
        x_embeds = x_embeds[x_idx]
        y_embeds = y_embeds[y_idx]
        x_words = x_words[x_idx]
        y_words = y_words[y_idx]

    x2d, y2d = get_2d_embeds(x_embeds, y_embeds)
    plot_embeddings(x_embeds[:k], y_embeds[:k], 
                    x_words[:k], y_words[:k],
                    source_lang=source_lang,
                    target_lang=target_lang)