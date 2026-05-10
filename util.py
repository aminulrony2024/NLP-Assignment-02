# Add your import statements here
from collections import Counter
# Add any utility functions here
def generateCorpusStopwords(tokens, top_k=10):
    """
    Generates corpus specific stopwords using a frequency based approach.

    This function identifies the most frequently occurring words in the
    corpus and treats them as stopwords. The intuition is that words
    appearing extremely frequently across documents often carry little
    semantic meaning for information retrieval tasks.

    Design Decision:
    A bottom up (data driven) method is used where stopwords are derived
    directly from the corpus instead of relying only on predefined lists.

    Parameters
    ----------
    tokens : list
        A list containing all tokens from the corpus.

    top_k : int (default = 10)
        The number of most frequent words to consider as corpus stopwords.

    Returns
    -------
    set
        A set containing the most frequent tokens treated as stopwords.
    """
    # Counting frequency of each token in the corpus
    freq = Counter(tokens)
    # Extracting the top_k most frequent words
    most_common = freq.most_common(top_k)
    # Converting these into a set for faster lookup
    corpus_stopwords = set([word for word, count in most_common])

    return corpus_stopwords