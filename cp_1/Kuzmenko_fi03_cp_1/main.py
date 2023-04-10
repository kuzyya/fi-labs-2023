import re
import math
import pandas as pd
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path

class Output:
    @staticmethod
    def get_parity(n):
        parity = False
        while n != 0:
            parity = not parity
            n = n & (n - 1)
        return parity

    @staticmethod
    def sort_by_value(unsorted_dict):
        return {k: v for k, v in sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True)}

    @staticmethod
    def write_to_file(dir_path, name, data, df=None):
        path = Path(dir_path) / name
        try:
            with path.open('w') as file:
                print(f"File created: {name}")

                file.write(f"Data entropy: {{{data['entropy']}}}\n")
                file.write(f"Data redundancy: {{{data['redundancy']}}}\n")

                if df is None:
                    file.write("Ngrams::\n")
                    for k, v in data['ngrams'].items():
                        file.write(f"{k},{v}\n")
                else:
                    df.to_csv(file, index=False)

        except Exception as e:
            print(f"An error occurred: {e}")


class Algorithms:
    @staticmethod
    def entropy(freqs, n):
        temp = [freq * (math.log2(freq) / math.log2(n * 2)) for freq in freqs]
        entropy = -sum(temp)
        return entropy

    @staticmethod
    def redundancy(e):
        return 1 - e / math.log2(32)


class Ngrams:
    def __init__(self, text, n):
        ngrams = self.split(text)
        self.ngrams = self.set_freqs(ngrams)
        freqs = list(self.ngrams.values())
        self.entropy = Algorithms.entropy(freqs, n)
        self.redundancy = Algorithms.redundancy(self.entropy)

    def split(self, text):
        return [text[i:i + 1] for i in range(len(text))]

    @staticmethod
    def set_freqs(ngrams):
        unsorted = {ngram: ngrams.count(ngram) / len(ngrams) for ngram in set(ngrams)}
        return Output.sort_by_value(unsorted)

    def get_data(self):
        return {
            'ngrams': self.ngrams,
            'entropy': self.entropy,
            'redundancy': self.redundancy,
        }


class Bigram(Ngrams):
    def split(self, text):
        if Output.get_parity(len(text)):
            length = len(text) - 1
        else:
            length = len(text)
        return [text[i - 1] + text[i] for i in range(1, length, 1)]


class BigramCross(Ngrams):
    def split(self, text):
        if not Output.get_parity(len(text)):
            length = len(text) - 1
        else:
            length = len(text)
        return [text[i - 1] + text[i] for i in range(1, length, 2)]


class Filter:
    @staticmethod
    def to_lowercase(text):
        lowercase_str = text.lower()
        return lowercase_str

    @staticmethod
    def filter_text(text, whitespaces):
        temp = text.replace("ъ", "ь").replace("ё", "е")
        if whitespaces:
            temp = re.sub(r"[^\u0430-\u044f]+", " ", temp)
            return re.sub(r"\s+", " ", temp)
        else:
            return re.sub(r"[^\u0430-\u044f]+", "", temp)


def bigrams_to_dataframe(bigrams):
    unique_chars = sorted(set([char for bigram in bigrams.keys() for char in bigram]))
    df = pd.DataFrame(index=unique_chars, columns=unique_chars).fillna(0)

    for bigram, freq in bigrams.items():
        row, col = bigram
        df.at[row, col] = freq

    return df

filename = "text"
output_dir = "output"

with open(filename, "r") as file:
        text = file.read()

lower_text = Filter.to_lowercase(text)
filtered_text_no_whitespace = Filter.filter_text(lower_text, whitespaces=False)
filtered_text_with_whitespace = Filter.filter_text(lower_text, whitespaces=True)

ngram_types = [
    ("monogram", Ngrams),
    ("bigram", Bigram),
    ("bigram_cross", BigramCross),
]

versions = [
    ("_no_whitespace", filtered_text_no_whitespace),
    ("_with_whitespace", filtered_text_with_whitespace),
]

for ngram_name, ngram_class in ngram_types:
    for version_suffix, filtered_text in versions:
        ngram_obj = ngram_class(filtered_text, n=len(ngram_name))
        data = ngram_obj.get_data()

        if ngram_name == "bigram":
            # df = bigrams_to_dataframe(data['ngrams'])
            # print(f"{ngram_name + version_suffix} DataFrame:\n")
            # print(df)
            # print("\n")
            Output.write_to_file(output_dir, ngram_name + version_suffix + ".csv", data)
        else:
            Output.write_to_file(output_dir, ngram_name + version_suffix + ".txt", data)
          