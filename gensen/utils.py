import re
import random
import unicodedata
from operator import itemgetter

import nltk
import pyarabic.araby as araby
from langdetect import detect


def normalize_text(text, lang=None):
    text = text.lower().strip()

    # text tokenization depends on the language
    if lang == 'ar':
        text = tokenize_arabic(text)
    elif lang == 'en':
        text = tokenize_english(text)
    else:
        try:
            if detect(text) == 'en':
                text = tokenize_english(text)
            elif detect(text) == 'ar':
                text = tokenize_arabic(text)
        except: # langdetect failed to detect useful features in text
            pass

    # remove extra spaces
    text = re.sub(' +', ' ', text)
    # remove html tags
    text = re.sub(re.compile('<.*?>'), ' ', text)
    # remove twitter usernames, web addresses
    text = re.sub(r"#[\w\d]*|@[.]?[\w\d]*[\'\w*]*|https?:\/\/\S+\b|"
                                              r"www\.(\w+\.)+\S*|", '', text)
    # strip repeated chars (extra vals)
    text = re.sub(r'(.)\1+', r"\1\1", text)
    # separate punctuation from words and remove not included marks
    text = " ".join(re.findall(r"[\w']+|[?!,;:]", text))
    # remove underscores
    text = text.replace('_', ' ')
    # remove double quotes
    text = text.strip('\n').replace('\"', '')
    # remove single quotes
    text = text.replace("'", '')
    # remove numbers
    text = ''.join(i for i in text if not i.isdigit())
    # remove extra spaces
    text = re.sub(' +', ' ', text)
    return text


def tokenize_english(text):
    text = ' '.join(nltk.word_tokenize(text))
    # convert accented characters to ASCII equivalents
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return text


def tokenize_arabic(text):
    text = text.lower().strip()
    text = araby.strip_tashkeel(text)
    text = ' '.join(araby.tokenize(text))
    return text


def extract_test_samples(path, number_samples):
    with open(path) as f:
        lines = f.readlines()

    indices = list(range(0, len(lines)))
    random.seed(42)
    test_indices = random.sample(indices, k=number_samples)

    test_samples = list(itemgetter(*test_indices)(lines))
    train_samples = [line for idx, line in enumerate(lines) if idx not in test_indices]

    with open(path + '.test', 'w') as f:
        for line in test_samples:
            f.write("%s" % line)

    with open(path + '.train', 'w') as f:
        for line in train_samples:
            f.write("%s" % line)
