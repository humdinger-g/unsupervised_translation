import os
import numpy as np
import torch
import fasttext
import fasttext.util
import requests
import shutil
import gzip
from tqdm.auto import tqdm
import argparse

from typing import List, Tuple


def load_data(path: str, 
              source_lang: str, 
              target_lang: str
        ) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """
    Loads source and target embeddings from specified directory:
    (source embeds, source words, target embeds, target words)



    Args:
        path (str): path to directory with fasttext data
        source_lang (str): source language ISO 639 code
            You can use lang2iso function from utils to get such a code
        target_lang (str): source language ISO 639 code
            You can use lang2iso function from utils to get such a code

    Returns:
        Tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]: _description_
    """

    x_embeds = torch.load(os.path.join(path, f'{source_lang}_embeds.pt'))
    y_embeds = torch.load(os.path.join(path, f'{target_lang}_embeds.pt'))
    x_words = np.load(os.path.join(path, f'{source_lang}_words.npy'))
    y_words = np.load(os.path.join(path, f'{target_lang}_words.npy'))
    return x_embeds, x_words, y_embeds, y_words


def download_fasttext_data(languages: List[str], 
                           path: str, 
                           num: int = 10000) -> None:
    """
    Downloads fasttext embedding matrix and list of words in the same order.
    Saves embeddings as torch.Tensor and words as np.ndarray[str]

    It's reasonable to use all 200k embeddings for inference and only 5-10k for training

    Args:
        languages (List[str]): list of ISO 639 language codes to download
            You can use lang2iso function from utils to get such a code
        path (str): path to where this data will be downloaded
        num (int, optional): number of most frequent embeddings to save. Defaults to 10000.
    """
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for lang in tqdm(languages):
        try:
            if not os.path.exists(os.path.join(path, f'cc.{lang}.300.bin.gz')) and\
                not os.path.exists(os.path.join(path, f'{lang}_embeds.pt')):
                url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz' 
                download_from_fasttext(url, path)
            else:
                print(f'fasttext data for {lang} already downloaded')

            unarchive(os.path.join(path, f'cc.{lang}.300.bin.gz'), path)
            ft = fasttext.load_model(os.path.join(path, f'cc.{lang}.300.bin'))

            embeds = ft.get_input_matrix()[:num, :]
            try:
                translations = ft.get_words()[:num]
            except:
                translations = ft.get_words(on_unicode_error='replace')[:num]

            embeds = torch.from_numpy(embeds)
            torch.save(embeds, os.path.join(path, f'{lang}_embeds.pt'))
            np.save(os.path.join(path, f'{lang}_words.npy'), np.array(translations))
            os.remove(os.path.join(path, f'cc.{lang}.300.bin.gz'))
            os.remove(os.path.join(path, f'cc.{lang}.300.bin'))
            print(f'Successfully downloaded {lang}')
        except:
            print(f'Failed to download {lang}')
            continue


      
def download_from_fasttext(url: str, path: str):
    """Helper function to load fasttext data"""

    try:
        if os.path.isdir(path):
            file_name = url.split("/")[-1]
            path = os.path.join(path, file_name)
        
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"File saved to {path}")
    except Exception as e:
        print(f"Error saving file from URL: {e}")
        
      
def unarchive(gz_file_path: str, output_dir: str):
    """Helper function to unarchive downloaded fasttext data"""

    try:
        os.makedirs(output_dir, exist_ok=True)
        print('Unarchiving...')
        with gzip.open(gz_file_path, 'rb') as f_in:
            file_name = os.path.splitext(os.path.basename(gz_file_path))[0]
            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"File {gz_file_path} unarchived to {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error unarchiving .gz file: {e}")
        return None
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FastText embeddings")
    parser.add_argument("--path", type=str, default='fasttext_data', required=True, help="Path to save the data")
    parser.add_argument("--languages", type=str, nargs='+', required=True, help="List of ISO 639 language codes to download")
    parser.add_argument("--num", type=int, default=10000, help="Number of most frequent embeddings to save")

    args = parser.parse_args()

    download_fasttext_data(args.languages, args.path, args.num)