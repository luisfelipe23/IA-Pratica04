from cmath import log
from nltk import WordNetLemmatizer, FreqDist, sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords, gutenberg


FILES = [
    "austen-emma.txt",
    "carroll-alice.txt",
    "melville-moby_dick.txt",
    "shakespeare-caesar.txt",
    "shakespeare-hamlet.txt",
]


TAG_DICT = {
    "NN": "n",
    "VB": "v",
    "JJ": "a",
    "RB": "r",
}


def penn2wordnet(penn_tag):
    for t, v in TAG_DICT.items():
        if penn_tag.startswith(t):
            return v
            
    return "n"


def preprocess(data):
    lemma = WordNetLemmatizer()
    stopw = stopwords.words("english")
    # Sentence tokenization
    p = sent_tokenize(data)
    # Word tokenization for each sentence
    p = [word_tokenize(s) for s in p]
    # Tag words in each sentence
    p = [pos_tag(s) for s in p]
    # Flatten senteces
    p = [wt for s in p for wt in s]
    # Cleanup and maps tagset
    p = [
        (w.lower(), penn2wordnet(t))
        for w, t in p if w.isalpha() and w not in stopw
    ]
    # Lemmatization
    p = [lemma.lemmatize(w, t) for w, t in p]
    
    return p


def calculateTfIdf(fds, fd):
    tfIdfs = []
    
    for w in fd:
        tf = fd[w] / fd.N()
        n = len(fds)
        df = sum(1 for fd in fds if fd[w] > 0)
        idf = log((1 + n)/(1 + df)).real
        tfIdfs.append((w, tf * idf))
        
    tfIdfs.sort(key=lambda x: -x[1])
    
    return tfIdfs


def demo():
    fds = {}
    
    for f in FILES:
        print(f"Preprocessando: {f}")
        preprocessed = preprocess(gutenberg.raw(f))
        fds[f] = FreqDist(preprocessed)
        
    for f, fd in fds.items():
        tf_idf = calculateTfIdf(fds.values(), fd)
        print(f"Arquivo: {f}")
        
        for i in range(5):
            w, v = tf_idf[i]
            print(f"\t{i+1}: {w} ({v} TF-IDF)")
