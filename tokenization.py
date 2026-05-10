from util import *
import re
import spacy
from nltk.tokenize import TreebankWordTokenizer
# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)


class Tokenization():
	def __init__(self):
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Fill in code here
		#spliiting words using regular expression
		tokenizedText = []
		for sentence in text:
			# Regular expression extracts sequences of letters or numbers, that's why using regex
			tokens = re.findall(r'\b\w+\b', sentence)
			tokenizedText.append(tokens)
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Fill in code here
		# Initializing the tokenizer provided by NLTK
		tokenizer = TreebankWordTokenizer()
		tokenizedText = []

		# Tokenizing each sentence separately
		for sentence in text:
			# Applying Penn Treebank tokenization rules
			tokens = tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		# Fill in code here
		tokenizedText = []

		for sentence in text:
			# Converting the sentence into a spaCy document object
			document = self.nlp(sentence)
			# Extracting token text from spaCy token objects
			tokens = [token.text for token in document]
			tokenizedText.append(tokens)
		return tokenizedText