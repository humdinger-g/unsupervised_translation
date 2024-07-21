import os
import numpy as np
import torch
from tqdm.auto import tqdm
import random
import spacy
import nltk
from nltk.corpus import wordnet


from typing import Dict, List

nlp_ru = spacy.load("ru_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")


def get_synonym_ru(word: str, 
                   embeds: torch.Tensor, 
                   words: np.ndarray[str], 
                   w2idx: Dict[str, int],
                   k: int = 5) -> List[str]:
    word = word.lower()
    if not word in w2idx:
        return '<NULL>'
    embed = embeds[w2idx[word]]
    synonyms_idx = torch.topk(torch.cosine_similarity(embeds, embed), k+1)[1]
    #idx = np.random.choice(synonyms_idx[1:], size=5)
    idx = synonyms_idx[1:]

    return words[idx].tolist()


def create_synonym_map(lang: str,
                       num_synonyms: int = 5,
                ) -> Dict[str, List[str]]:
    
    embeds = torch.load(f"fasttext_data/{lang}_embeds.pt")[:20000]
    words = np.load(f"fasttext_data/{lang}_words.npy")[:20000]
    w2idx = {w:i for i, w in enumerate(words)}

    synonyms = {}
    for word in tqdm(words):
        syn = get_synonym_ru(word, embeds, words, w2idx, k=num_synonyms)
        synonyms[word] = syn

    return synonyms


def get_synonyms(word):
    """Get synonyms for an English word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)