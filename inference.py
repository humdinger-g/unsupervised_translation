import os
import re
import string
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from wbw_translation.models import InferenceModel
from wbw_translation.load_data import load_data

import warnings
warnings.filterwarnings('ignore')



stopwords = {
    'en': ['am','is','are','to'],
    'ru': ['же','бы','y','об']
}

embeddings = {}
words = {}
models = {}


def word_by_word_translation(text: str,
                             source_lang: str,
                             target_lang: str,
                             add_prefix: bool = False
                    ) -> str:
    """
    runs InferenceModel for a given text string

    Args:
        text (str): input text in source language
        source_lang (str): source language code
        target_lang (str): target language code
        add_prefix (bool, optional): needed for full translation pipeline: if True,
            will add prefix token to the translation which is needed for 
            text-correction model. Defaults to False.

    Returns:
        str: input text translated into target language
    """
    assert os.path.exists(f'inference_weights/model-{source_lang}-{target_lang}'), \
        f"There is no inference weights for {source_lang}-{target_lang} translation"

    if add_prefix:
        assert target_lang == 'en' or target_lang == 'ru', \
            "Prefixes are available only for en and ru languages"
        
    if (source_lang, target_lang) not in embeddings:
        print('loading embeddings...')
        x_embeds, x_words, y_embeds, y_words = load_data('fasttext_data', source_lang, target_lang)
        embeddings[(source_lang, target_lang)] = x_embeds, x_words, y_embeds, y_words
    
    x_embeds, x_words, y_embeds, y_words = embeddings[(source_lang, target_lang)]

    if (source_lang, target_lang) not in models:
        print('loading wbw model...')
        W = torch.load(f'inference_weights/model-{source_lang}-{target_lang}')['W.weight']
        model = InferenceModel(W)
        models[(source_lang, target_lang)] = model

    model = models[(source_lang, target_lang)]

    source2idx = {word: i for i, word in enumerate(x_words)}

    punctuation = [s for s in list(string.punctuation) if s not in ["'"]]
    text = text.translate(str.maketrans('', '',''.join(punctuation)))
    text = ' '.join([word.lower() if word not in ["I"] else word for word in text.split()])

    text = re.split(" |'", text)

    output = []
    for word in text:
        if word.isdigit():
            output += [word]
            continue
        if word not in source2idx or \
                (source_lang in ['en', 'ru'] and word in stopwords[source_lang]):
            output += ['<NULL>']
            continue

        x_word = x_embeds[source2idx[word]]
        pred = model(x_word)
        nearest = torch.topk(torch.cosine_similarity(pred, y_embeds), 1)[1].item()
        nearest = y_words[nearest]
        output += [nearest]
    
    output = ' '.join(output)
    output = ' '.join([word.lower() if word not in ["<NULL>"] else word for word in output.split()])

    prefix = '<fix>: ' if target_lang == 'en' else '<исправить>: '
    return output if not add_prefix else prefix + output


text_correction_model = None
tokenizer = None

def load_text_correction_model(model_type='large'):
    global text_correction_model, tokenizer
    if not model_type == 'small':
        adapter_model_id = 'gudleifrr/mt5-text-correction-enru'
        tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained('google/mt5-large')
        model.resize_token_embeddings(len(tokenizer))
        text_correction_model = PeftModel.from_pretrained(model, adapter_model_id)
    else:
        model_id = 'gudleifrr/text-correction-en-small'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text_correction_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


def text_correction(text, model_type='large'):
    if text_correction_model is None or tokenizer is None:
        print('loading text correction model...')
        load_text_correction_model(model_type=model_type)

    text_correction_model.eval()

    inputs = tokenizer(text, return_tensors='pt')
    outputs = text_correction_model.generate(**inputs, max_length=3*len(text.split()))

    return tokenizer.decode(outputs[0], skip_special_tokens=True)