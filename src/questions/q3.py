from nltk import load, FreqDist, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from numpy import sqrt, sum

FILES = [
    "data/review1.txt",
    "data/review2.txt",
    "data/review3.txt",
]

POSITIVE_WORDS = "data/positive_words.csv"
NEGATIVE_WORDS = "data/negative_words.csv"


def preprocess(data):
    stemmer = SnowballStemmer("english")
    
    return [stemmer.stem(w.lower()) for w in word_tokenize(data) if w not in stopwords.words("english")]


def normalizedVector(voc, bag):
    return [bag[w] / bag.N() for w in voc]


def distance(bag1, bag2):
    voc = bag1 + bag2
    intersection = bag1 & bag2
    vector1 = normalizedVector(voc, bag1)
    vector2 = normalizedVector(voc, bag2)
    
    up = sum(
        x1 * x2 for x1, x2 in zip(vector1, vector2)
    )
    
    down = (
        sqrt(sum(x ** 2 for x in vector1)) *
        sqrt(sum(x ** 2 for x in vector2))
    )
    
    return (up / down), sum(x for x in intersection.values())


def calculateCompound(positive, negative):
    return (positive - negative) / (positive + negative)


def demo():
    positiveBag = FreqDist(
        preprocess(load(POSITIVE_WORDS, format="text"))
    )
    negativeBag = FreqDist(
        preprocess(load(NEGATIVE_WORDS, format="text"))
    )

    for f in FILES:
        words_bag = FreqDist(preprocess(load(f)))
        positive, positiveCount = distance(words_bag, positive_bag)
        negative, negativeCount = distance(words_bag, negative_bag)
        compound = calculateCompound(positive, negative)
        print(f"Arquivo: {f}")
        print(f"\tPositivo: {positive} ({positive_count} palavras)")
        print(f"\tNegativo: {negative} ({negative_count} palavras)")
        print(f"\tClassificação: {compound}")
