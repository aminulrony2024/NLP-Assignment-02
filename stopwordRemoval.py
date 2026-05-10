from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords


class StopwordRemoval():

    def fromList(self, text):
        """
        Stopword Removal using a curated stopword list and a corpus specific list.

        Stopwords are very frequent words for example "the", "is", "and" , that usually
        carry little semantic meaning in Information Retrieval tasks. Removing them
        helps reduce noise and decreases the vocabulary size.

        Design Decisions:
        1. Use the predefined English stopword list provided by NLTK.
        2. Generate additional corpus-specific stopwords based on the dataset
           using the function `generateCorpusStopwords()` from util.py.
        3. Combine both lists to remove both general and corpus-specific stopwords.

        Parameters
        ----------
        text : list
            A list of lists where each sub-list contains tokens representing a sentence.

        Returns
        -------
        list
            A list of lists where stopwords have been removed from each sentence.
        """
		#Fill in code here
        # curated stopwords
        nltk_stopwords = set(stopwords.words('english'))

        # collecting tokens
        all_tokens = []

        for sentence in text:
            for word in sentence:
                all_tokens.append(word)

        # corpus stopwords , this function is defined in util.py file
        # Generating corpus-specific stopwords using frequency statistics
        # corpus_stopwords = generateCorpusStopwords(all_tokens)
        # Combining curated stopwords and corpus-specific stopwords
        # stop_words = nltk_stopwords.union(corpus_stopwords)
        stop_words = nltk_stopwords

        stopwordRemovedText = []
        # Removing stopwords from each sentence
        for sentence in text:

            filtered = []
            for word in sentence:
                # Converting word to lowercase before checking stopword membership
                # This ensures case insensitive comparison
                if word.lower() not in stop_words:
                    filtered.append(word)
            # Storing filtered sentence
            stopwordRemovedText.append(filtered)

        return stopwordRemovedText