import string
import torch
import numpy as np
import pandas as pd
import json
import os
from torch import nn
import nltk
import spacy

from wbw_translation.models import (ProcrusterMapping, 
                    LinearModel, 
                    InferenceModel,
                    SplitModels, 
                    CombineModels)

from wbw_translation.load_data import load_data

from typing import Tuple, Dict, List, Optional

nlp_ru = spacy.load("ru_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

def set_determinism(rs):
    np.random.seed(rs)
    torch.manual_seed(rs)


def get_same_words(x_embeds: torch.Tensor,
                   y_embeds: torch.Tensor,
                   x_words: np.ndarray,
                   y_words: np.ndarray,
                   only_numeric: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function that returns a tuple (X, Y) with embeddings of
    the same numbers and punctuation



    Args:
        x_embeds (torch.Tensor): source embeddings
        y_embeds (torch.Tensor): target embeddings
        x_words (np.ndarray): source words
        y_words (np.ndarray): target words
        only_numeric (bool, optional): if True, ignores punctuation. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: source and target embeddings of same words
    """

    # create dictionaries {word: idx}
    x_num = {}
    for i in range(len(x_words)):
        if x_words[i].isnumeric() or\
              ((x_words[i] in string.punctuation) and not only_numeric):
            x_num[x_words[i]] = i

    y_num = {}
    for i in range(len(y_words)):
        if y_words[i].isnumeric() or\
              ((y_words[i] in string.punctuation) and not only_numeric):
            y_num[y_words[i]] = i

    X, Y = [], []
    for k, v in x_num.items():
        if k in y_num:
            X += [x_embeds[v]]
            Y += [y_embeds[y_num[k]]]
    X = torch.vstack(X)
    Y = torch.vstack(Y)

    return X, Y


def get_procrustes_mapping(X: torch.Tensor, 
                           Y: torch.Tensor,
                           full: bool = True
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the orthogonal matrices Wxy amd Wyx that minimize
    Frobenius norms of X @ Wxy - Y and X - Y @ Wyx

    Args:
        X (torch.Tensor): source embeddings
        Y (torch.Tensor): target embeddings
        full (bool): if true, calculates approximate SVD

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: weight matrices Wxy and Wyx
    """

    procrust_xy = ProcrusterMapping(X, Y)
    procrust_yx = ProcrusterMapping(Y, X)

    return procrust_xy.W, procrust_yx.W


def load_lang2iso() -> None:
    """Helper function to download ISO 639 codes"""

    url = "https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes"
    dfs = pd.read_html(url)
    df = dfs[0]
    language_dict = dict(zip(df['ISO Language Names'], 
                             df['Set 1']))

    if not os.path.exists('files'):
        os.makedirs('files')

    with open('files/lang2iso.json', 'w') as f:
        json.dump(language_dict, f, ensure_ascii=False, indent=4)


def lang2iso(language: str) -> str:
    """Helper function to convert language name to its ISO 639 code"""

    assert language[0].isupper(), 'Language name must start with an uppercase letter'

    if not os.path.exists('files/lang2iso.json'):
        load_lang2iso()
    
    with open('files/lang2iso.json', 'r') as f:
        mapping = json.load(f)

    return mapping[language]


def norm(embeds: torch.Tensor) -> torch.Tensor:
    """Modifies embeddings to zero mean and std = 1 along 0th dimension"""

    return (embeds - embeds.mean(0)) / embeds.std(0)


def norm_sort(embeds):
    """Sorts embeddings in ascending order of norm"""

    return embeds[np.argsort(torch.norm(norm(embeds), dim=1))]


def find_nearest_neighbors(x: torch.Tensor,
                           embeds: torch.Tensor,
                           k: int = 1,
                           return_embeddings: bool = False,
                           method: str = 'cos'
                    ) -> torch.Tensor:
    """
    Finds embedding that is nearest to x among embeds

    Args:
        x (torch.Tensor): embedding
        embeds (torch.Tensor): embeddings among which the search is done
        k (int): number of nearest neighbors
        return_embeddings (bool): if true, returns embeddings; else -- indices
        method (str, optional): method of similarity. Defaults to 'cos'.
            if 'cos', simple cosine similarity,
            TODO: if 'csls', cosine similiarity corrected on the average distance 
                to other nearest neighbors. Provides higher accuracy for
                popular words (i.e. words which have a lot of neighbors)

    Returns:
        torch.Tensor: nearest embedding
    """
    
    assert method != 'csls', "CSLS metric is not implemented yet"
    cosines = torch.cosine_similarity(x, embeds)
    idx = torch.topk(cosines, k=k)[1]
    if not return_embeddings:
        return idx
    return embeds[idx]


def translate_sample(model: nn.Module,
                     source_lang: str,
                     target_lang: str,
                     seed=42) -> Dict[str, str]:
    
    np.random.seed(seed)
    res = {}
    x_embeds, x_words, y_embeds, y_words = load_data('fasttext_data',
                                                     source_lang,
                                                     target_lang)
    sample_idx = np.random.choice(500, 10)
    sample = x_embeds[sample_idx]
    y_pred, _ = model(sample, sample)
    for i, y in enumerate(y_pred):
        idx = find_nearest_neighbors(y, y_embeds, k=5)
        res[x_words[sample_idx[i]]] = ', '.join(y_words[idx].tolist())
    return res


def save_inference_models():
    """takes all LinearModels' weights (two-sided) splits them and converts to inference models
    and saves to inferece_weights/"""
    
    current_pairs = list(map(lambda x: x.split('-')[-2:], os.listdir('inference_weights')))

    for file in os.listdir('linear_weights'):
        pair = [file.split('-')[-2], file.split('-')[-1]]
        weights = torch.load('linear_weights/' + file, map_location=torch.device('cpu'))
        model1 = InferenceModel(weights['Wxy.weight'])
        model2 = InferenceModel(weights['Wyx.weight'])

        if pair not in current_pairs:
            torch.save(model1.state_dict(), f'inference_weights/model-{pair[0]}-{pair[1]}')
            print(f'saved model-{pair[0]}-{pair[1]}')
        if pair[::-1] not in current_pairs:
            torch.save(model2.state_dict(), f'inference_weights/model-{pair[1]}-{pair[0]}')
            print(f'saved model-{pair[0]}-{pair[1]}')



def create_all_possible_models(path: str,
                               save: bool = False
                        ) -> Optional[Dict[str, InferenceModel]]:
    """
    Performs transitive closure obtaining all new possible inference models:
    If there are model that translate from X to Y and from Y to Z, then
    model for translation from X to Z can be created simply by multiplying weight matrices.

    Args:
        path (str): path to the folder with current InferenceModel's weights
        save (bool, optional): if True, new models will be saved, otherwise
            dictionary (language_pair, InferenceModel) will be returned. Defaults to False.

    Returns:
        Optional[Dict[str, InferenceModel]]: dictionary (language_pair, InferenceModel)
    """

    new_models = {}
    for model1 in os.listdir(path):
        for model2 in os.listdir(path):
            lang1, lang2 = model1.split('-')[-2:]
            lang3, lang4 = model2.split('-')[-2:]

            if (lang2 == lang3) and (lang1 != lang4) and \
                not os.path.exists(os.path.join(path, f'model-{lang1}-{lang4}')):

                Wxy = torch.load(os.path.join(path, model1))['W.weight']
                Wyz = torch.load(os.path.join(path, model2))['W.weight']
                W = (Wxy.T @ Wyz.T)
                model = InferenceModel(W.T)
                if save:
                    torch.save(model.state_dict(), os.path.join(path, f'model-{lang1}-{lang4}'))
                new_models[(lang1, lang4)] = model

    return new_models



def tensor(arr: np.ndarray) -> torch.Tensor:
    """Converts numpy array to 32-bit torch Tensor"""
    return torch.tensor(arr, dtype=torch.float)


def lemmatize(text, lang):
    text.replace("\n", "")
    nlp = nlp_ru if lang == 'ru' else nlp_en
    doc = nlp(text)
    degraded_tokens = []

    for token in doc:
        if token.pos_ in {"DET", "ADP"}:
            continue
        elif token.pos_ == "VERB":
            degraded_tokens.append(token.lemma_)
        elif token.pos_ == "NOUN" and token.tag_ == "NNS":
            degraded_tokens.append(token.lemma_)
        elif token.pos_ == "PRON":
            continue
        else:
            degraded_tokens.append(token.text.lower())

    return ' '.join(degraded_tokens)