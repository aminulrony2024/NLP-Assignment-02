from util import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class InflectionReduction:

    def porterStemmer(self, text):
        """
        Performing stemming using the Porter Stemmer.

        Stemming is the process of reducing inflected or derived words
        to their root form by removing suffixes. The Porter Stemmer
        applies a set of heuristic rules to strip common endings
        such as 'ing', 'ed', 'ly', etc.

        Design Decision:
        Porter Stemmer is used because it is one of the most widely
        used stemming algorithms in Information Retrieval systems.
        It is computationally efficient and reduces vocabulary size.

        Limitations:
        - The resulting stem may not be a valid English word.
        - Over-stemming may occur for example "university" -> "univers" .

        Parameters
        ----------
        text : list
            A list of tokenized sentences (list of lists).

        Returns
        -------
        list
            A list of lists containing stemmed tokens.
        """
        stemmer = PorterStemmer()
        reducedText = []
        # Iterating over each sentence in the tokenized text
        for sentence in text:
            reduced_sentence = []
            # Applying stemming to each word in the sentence
            for word in sentence:
                # Appending the processed sentence to the result
                reduced_sentence.append(stemmer.stem(word))

            reducedText.append(reduced_sentence)

        return reducedText


    def wordnetLemmatizer(self, text):
        """
        Performing lemmatization using WordNet Lemmatizer.

        Lemmatization reduces a word to its dictionary base form
        (lemma) using lexical knowledge from the WordNet database.
        Unlike stemming, the output is usually a valid English word.

        Design Decision:
        WordNet Lemmatizer is chosen because it preserves the
        semantic meaning of words better than stemming.

        Limitations:
        - Requires lexical resources (WordNet).
        - Without POS tagging, the lemmatizer assumes the word
          is a noun which may produce suboptimal results.

        Parameters
        ----------
        text : list
            A list of tokenized sentences (list of lists).

        Returns
        -------
        list
            A list of lists containing lemmatized tokens.
        """
        lemmatizer = WordNetLemmatizer()
        reducedText = []
        # Iterating through each sentence
        for sentence in text:
            reduced_sentence = []
            # Applying lemmatization to each word
            for word in sentence:
                # Here storing the processed sentences
                reduced_sentence.append(lemmatizer.lemmatize(word))

            reducedText.append(reduced_sentence)

        return reducedText


    def reduce(self, text):

        return self.porterStemmer(text)