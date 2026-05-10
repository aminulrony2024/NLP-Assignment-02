from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize


class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")
		
	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		# Fill in code here
		# Common abbreviations to protect
		abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'U.S.', 'U.S.A.', 'e.g.', 'i.e.']

		# Replacing abbreviation periods temporarily
		for abbr in abbreviations:
			text = text.replace(abbr, abbr.replace('.', '<prd>'))

		# Protecting decimal numbers for example 3.14
		text = re.sub(r'(\d)\.(\d)', r'\1<prd>\2', text)

		# Splitting sentences using punctuation
		sentences = re.split(r'[.!?]+', text)

		# Restoring periods
		sentences = [s.replace('<prd>', '.') for s in sentences]

		# Removing empty strings and stripping space
		segmentedText = [s.strip() for s in sentences if s.strip()]

		return segmentedText

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# Fill in code here

		#using NLTK's punkt sentence tokenizer
		segmentedText = sent_tokenize(text)

		return segmentedText


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# Fill in code here

		#processing text using spacy
		document = self.nlp(text)

		#extracting sentences
		segmentedText = [sent.text.strip() for sent in document.sents]
		return segmentedText