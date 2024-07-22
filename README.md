# Unsupervised translation

This repo contains code for unsupervised translation between different languages that is performed in two stages. Firstly, the text is translated word-by-word by a linear model trained with iterative closest point algorithm, the code is in `wbw_translation` . Then language model in target language is applied to fix the translation and make it more readable, code notebooks for that are in `text_correction` . You can find a detailed information about this work in the [report](https://knowing-racer-d3e.notion.site/Report-017cad3a19a54a92bef366392743bba8).

### Environment

clone repository and install required packages:

```bash
git clone https://github.com/humdinger-g/unsupervised_translation
cd unsupervised_translation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Load data

To load fasttext data (embeddings and words) run

```bash
python3 load_data.py --path --languages --num
```

1. `path`  is a path where you want to save, by default saved to ‘fasttext_data’
2. `languages` is a list of language codes without commas. If you don’t know a language code for a specific language, use `utils.lang2iso` function or navigate to `files/lang2iso.json`
3. `num`  is the number of embeddings and words to download (they are sorted by frequency), default is 10_000, but loading all 200_000 is advised for better inference

example:

```bash
python3 load_data.py --path fasttext_data --languages en ru --num 200000
```

### ICP training

To train word-by-word model from scratch, you can run wandb sweep (login before that)

```bash
python3 wbw_translation/icp_sweep.py --source_lang --target_lang --config
```

1. `source_lang` and `target_lang` are language codes for source and target languages respectively
2. `config` is a path to the training config. You can either create this by yourself, you modify existing one in `configs/`

Also, you can run a single iteration right away with such code (ensure you downloaded Russian and English data):

```python
import torch
from torch import nn

from wbw_translation.icp import ICP
from wbw_translation.load_data import load_data
from wbw_translation.models import LinearModel
from utils import (norm_sort, 
                   get_procrustes_mapping, 
                   get_same_words,
                   translate_sample)
                   
# load data from fasttext_data folder
x_embeds, x_words, y_embeds, y_words = load_data(
    'fasttext_data',
    'en',
    'ru'
)

# leave only top 20000 in order to avoid noisy ones
x_embeds = x_embeds[:20000]
y_embeds = y_embeds[:20000]

# sort them by norm and leave top 5000 with least norm
x_norms = norm_sort(x_embeds)[:5000]
y_norms = norm_sort(y_embeds)[:5000]

# instantiate ICP class
icp = ICP(x_norms, y_norms)

# initialize model parameters with mapping of coinciding words
X, Y = get_same_words(x_embeds, y_embeds, x_words[:20000], y_words[:20000])
Wxy, Wyx = get_procrustes_mapping(X, Y)
model = LinearModel(300, init=(Wxy.T, Wyx.T))

# train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = icp.train_iteration(model, optimizer, scheduler=None,
                             num_epochs=200, batch_size=128,
                             random_state=42)
                             
# get sample translations; change seed to get different sample
translate_sample(model, 'en', 'ru', seed=25)
```

for the code above you should get this result:

```
{'work': 'работа, работать, поработать, работаю, работы',
 'post': 'блоге, фраза, топике, процитирую, пишу',
 'government': 'правительства, президента, премьер-министра, чиновников, властей',
 'him': 'ему, он, него, сына, дочь',
 '18': '14, 17, 13, 12, 11',
 '2008': '2013, февраля, апреля, января, 2014',
 'come': 'прийти, придти, пойти, собрался, идти',
 'year': 'полгода, год, месяц, месяца, полмесяца',
 'known': 'называют, знают, называется, именуют, говорят',
 'At': 'На, Тогда, Наконец, Позже, Однако'}
```

### Text correction

Code for dataset creation and model tuning is stored in `text_correction/` . You can access both [dataset](https://huggingface.co/datasets/gudleifrr/text-correction-en) and models ([large](https://huggingface.co/gudleifrr/mt5-text-correction-enru), [small](https://huggingface.co/gudleifrr/text-correction-en-small)) in huggingface. To perform word-by-word translation and text correction use `inference.word_by_word_translation` and `inference.text_correction` . Note that using text correction model requires special tokens `<fix>` or `<исправить>` depending on the language.