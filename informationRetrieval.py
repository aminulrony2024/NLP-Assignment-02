from util import *

# Add your import statements here

import math
from collections import defaultdict



class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None      
		self.doc_tfidf = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		# index = None

		#Fill in code here
		self.docIDs = docIDs
		N = len(docs)  # total number of documents
 
		# ── Step 1: flatten each doc into a bag of words ──────────────────────
		# doc_tokens[i] = list of all tokens in docs[i]
		doc_tokens = []
		for doc in docs:
			tokens = []
			for sentence in doc:
				tokens.extend(sentence)
			doc_tokens.append(tokens)
 
		# ── Step 2: compute raw term frequencies per document ─────────────────
		# tf_raw[doc_id][term] = count of term in that doc
		tf_raw = {}
		for doc_id, tokens in zip(docIDs, doc_tokens):
			tf_raw[doc_id] = defaultdict(int)
			for token in tokens:
				tf_raw[doc_id][token] += 1
 
		# ── Step 3: compute document frequency (df) for each term ─────────────
		# df[term] = number of documents containing term
		df = defaultdict(int)
		for doc_id in docIDs:
			for term in tf_raw[doc_id]:
				df[term] += 1
 
		# ── Step 4: compute TF-IDF weights ────────────────────────────────────
		# Using:  tf = (raw_count / doc_length),  idf = log2(N / df)
		# tfidf[term][doc_id] = weight
		tfidf = defaultdict(dict)
		doc_tfidf = {doc_id: {} for doc_id in docIDs}
 
		for doc_id, tokens in zip(docIDs, doc_tokens):
			doc_len = len(tokens) if tokens else 1  # avoid /0
			for term, count in tf_raw[doc_id].items():
				tf  = count / doc_len
				idf = math.log2(N / df[term])       # df >= 1 always
				weight = tf * idf
				tfidf[term][doc_id] = weight
				doc_tfidf[doc_id][term] = weight
 
		self.index    = tfidf
		self.doc_tfidf = doc_tfidf
		


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		N = len(self.docIDs)
 
		# pre-compute doc norms for cosine denominator
		doc_norms = {}
		for doc_id, term_weights in self.doc_tfidf.items():
			doc_norms[doc_id] = math.sqrt(
				sum(w * w for w in term_weights.values())
			)
 
		for query in queries:
			# ── flatten query into bag of words ────────────────────────────
			query_tokens = []
			for sentence in query:
				query_tokens.extend(sentence)
 
			if not query_tokens:
				# empty query → return docs in original order
				doc_IDs_ordered.append(list(self.docIDs))
				continue
 
			# ── compute query TF-IDF vector ────────────────────────────────
			query_tf_raw = defaultdict(int)
			for token in query_tokens:
				query_tf_raw[token] += 1
 
			query_len = len(query_tokens)
			query_vec = {}
			for term, count in query_tf_raw.items():
				if term in self.index:          # only terms in vocab matter
					tf  = count / query_len
					idf = math.log2(N / len(self.index[term]))
					query_vec[term] = tf * idf
 
			if not query_vec:
				# all query terms OOV → no matches, return docs in order
				doc_IDs_ordered.append(list(self.docIDs))
				continue
 
			# ── compute cosine similarity for each doc ─────────────────────
			query_norm = math.sqrt(sum(w * w for w in query_vec.values()))
 
			scores = {}
			for doc_id in self.docIDs:
				dot = 0.0
				for term, q_weight in query_vec.items():
					d_weight = self.doc_tfidf[doc_id].get(term, 0.0)
					dot += q_weight * d_weight
				d_norm = doc_norms.get(doc_id, 0.0)
				if query_norm > 0 and d_norm > 0:
					scores[doc_id] = dot / (query_norm * d_norm)
				else:
					scores[doc_id] = 0.0
 
			# ── sort docs by score descending ──────────────────────────────
			ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
			doc_IDs_ordered.append(ranked)
 
		return doc_IDs_ordered
 
		




