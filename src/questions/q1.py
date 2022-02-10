from nltk.corpus import gutenberg
from numpy import average

FILES = [
    "shakespeare-caesar.txt",
    "shakespeare-hamlet.txt",
    "shakespeare-macbeth.txt",
]


def countRepetitions(words: list[str]) -> dict[str, int]:
    counts = {}
    
    for word in words:
        if word in counts:
            counts[word] = counts[word] + 1
        else:
            counts[word] = 1
            
    return counts


def demo():
    for f in FILES:
        words = gutenberg.words(f)
        sents = gutenberg.sents(f)
        wordCount = len(words)
        sentCount = len(sents)
        repetitionCount = countRepetitions(words)
        
        nonRepeatCount = len(
            [word for word, c in repetitionCount.items() if c == 1]
        )
        
        repeatCount = len(
            [word for word, c in repetitionCount.items() if c > 1]
        )
        
        averageSentenceLen = average([len(s) for s in sents])
        
        print(f"Nome do Arquivo: {f}")
        print(f"\tQuantidade de palavras:{wordCount}")
        print(f"\tQuantidade de sentenças: {sentCount}")
        print(f"\tQuantidade de palavras não repetidas: {nonRepeatCount}")
        print(f"\tQuantidade de palavras repetidas: {repeatCount}")
        print(f"\tMédia de palavras por sentença: {averageSentenceLen}")
