import pandas as pd
from typing import List
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re
from bs4 import BeautifulSoup

tokenize = get_tokenizer("basic_english", language="en")
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]


class HateSpeechDataset(Dataset):
    def __init__(self, file_loc: str, data_rows: List[str]):
        self.dataframe = pd.read_csv(file_loc)
        self.data_rows = data_rows
        self.data = self.dataframe.loc[:, data_rows].values

    def __call__(self, *args, **kwargs):
        return self.data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return self.data[index]


def tweet_cleaner(t):
    soup = BeautifulSoup(t, 'lxml')
    souped = soup.get_text()
    clean_1 = re.sub(r' @[A-Za-z0-9]+ | https?://[A-Za-z0-9./]+', '', souped)
    try:
        clean_2 = clean_1.replace('ï¿½', '')
    except Exception as e:
        print(e)
        clean_2 = clean_1
    letters_only = re.sub("[^a-zA-Z]", ' ', clean_2)
    lower_case = letters_only.lower()
    final_clean = re.sub('\s+', ' ', lower_case)
    final_result = ' '.join([word for word in final_clean.split() if word not in stopwords])
    return final_result.strip()


def iterate_dataset(dataset: Dataset):
    for sentence, label in dataset:
        yield tokenize(tweet_cleaner(sentence))


def get_data_and_vocab(file_path: str, out_file: str, columns: List):
    hate_dataset = HateSpeechDataset(file_path, columns)
    cleaned_data = {"sentence": [], "label": []}
    for sentence, label in hate_dataset:
        cleaned_data["sentence"].append(tweet_cleaner(sentence))
        cleaned_data["label"].append(label)
    cleaned_df = pd.DataFrame(cleaned_data)
    cleaned_df.to_csv(out_file)
    vocab = build_vocab_from_iterator(iterate_dataset(hate_dataset), min_freq=3,
                                      specials=["<unk>", "<sos>", "<eos>"], special_first=True)
    return vocab, out_file
