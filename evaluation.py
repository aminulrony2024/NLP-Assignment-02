from util import *

# Add your import statements here
import math

class Evaluation():


	def _get_true_ids(self, query_id, qrels):
		"""
		Extract set of relevant doc IDs for a query from qrels.
 
		Cranfield qrels format:
		  {"query_num": int, "id": int, "position": int}
		  position 1-4 all treated as relevant (binary relevance).
		"""
		return set(
			int(item["id"])
			for item in qrels
			if int(item["query_num"]) == int(query_id) and 1 <= int(item["position"]) <= 4
		)
	


	def _get_grade(self, doc_id, query_id, qrels):
		"""
		Graded relevance for nDCG: grade = 5 - position (so pos 1 → 4, pos 4 → 1).
		Returns 0 if doc not in qrels.
		"""
		for item in qrels:
			if item["query_num"] == query_id and int(item["id"]) == doc_id:
				return 5 - item["position"]   # grades: 4, 3, 2, 1
		return 0


	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		# precision = -1

		#Fill in code here
		true_set = set(true_doc_IDs)
		top_k    = query_doc_IDs_ordered[:k]
		hits     = sum(1 for doc_id in top_k if doc_id in true_set)
		precision = hits / k if k > 0 else 0.0
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		# meanPrecision = -1

		#Fill in code here
		precisions = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_ids(query_id, qrels)
			p = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			precisions.append(p)
		return sum(precisions) / len(precisions) if precisions else 0.0
		

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""
		# recall = -1

		#Fill in code here

		true_set  = set(true_doc_IDs)
		top_k     = query_doc_IDs_ordered[:k]
		hits      = sum(1 for doc_id in top_k if doc_id in true_set)
		recall    = hits / len(true_set) if true_set else 0.0
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		# meanRecall = -1

		#Fill in code here
		recalls = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_ids(query_id, qrels)
			r = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recalls.append(r)
		return sum(recalls) / len(recalls) if recalls else 0.0
		

	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query
		"""
		# fscore = -1

		#Fill in code here
		beta = 0.5
		p = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		denom = (beta ** 2) * p + r
		if denom == 0:
			return 0.0
		fscore = (1 + beta ** 2) * p * r / denom
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		# meanFscore = -1

		#Fill in code here

		fscores = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_ids(query_id, qrels)
			f = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			fscores.append(f)
		return sum(fscores) / len(fscores) if fscores else 0.0
 
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query
		"""
		# nDCG = -1

		#Fill in code here
		true_set = set(true_doc_IDs)
		dcg = 0.0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			rel = 1 if doc_id in true_set else 0
			dcg += rel / math.log2(rank + 1)
 
		# Ideal DCG: place all relevant docs at top
		n_relevant = len(true_set)
		ideal_grades = [1] * min(n_relevant, k)   # binary: each relevant = grade 1
		idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_grades))
 
		nDCG = dcg / idcg if idcg > 0 else 0.0
		return nDCG



	def _queryNDCG_graded(self, query_doc_IDs_ordered, query_id, qrels, k):
		"""
		nDCG@k with actual graded relevance from qrels.
		grade = 5 - position  (position ∈ {1,2,3,4} → grade ∈ {4,3,2,1}).
		"""
		# Build grade lookup for this query
		grade_map = {}
		for item in qrels:
			if int(item["query_num"]) == int(query_id) and 1 <= int(item["position"]) <= 4:
				grade_map[int(item["id"])] = 5 - item["position"]
 
		# DCG
		dcg = 0.0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			rel = grade_map.get(doc_id, 0)
			dcg += rel / math.log2(rank + 1)
 
		# IDCG: sort all relevant docs by grade descending, take top k
		ideal = sorted(grade_map.values(), reverse=True)[:k]
		idcg  = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
 
		return dcg / idcg if idcg > 0 else 0.0


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""
		# meanNDCG = -1

		#Fill in code here

		ndcgs = []
		for i, query_id in enumerate(query_ids):
			n = self._queryNDCG_graded(doc_IDs_ordered[i], query_id, qrels, k)
			ndcgs.append(n)
		return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)
		"""
		# avgPrecision = -1

		#Fill in code here
		true_set = set(true_doc_IDs)
		R        = len(true_set)
		if R == 0:
			return 0.0
 
		hits = 0
		ap   = 0.0
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			if doc_id in true_set:
				hits += 1
				ap   += hits / rank   # precision at this rank
 
		avgPrecision = ap / R

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""
		# meanAveragePrecision = -1

		#Fill in code here

		aps = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_ids(query_id, q_rels)
			ap = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			aps.append(ap)
		return sum(aps) / len(aps) if aps else 0.0



	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query

		Parameters
		----------
		arg1 : list
			Ranked list of document IDs
		arg2 : int
			Query ID
		arg3 : list
			List of relevant document IDs
		arg4 : int
			The k value

		Returns
		-------
		float
			Reciprocal rank value
		"""

		# reciprocalRank = -1

		#Fill in code here

		true_set = set(true_doc_IDs)
		for rank, doc_id in enumerate(query_doc_IDs_ordered[:k], start=1):
			if doc_id in true_set:
				return 1.0 / rank
		return 0.0


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries

		Parameters
		----------
		arg1 : list
			List of ranked document lists
		arg2 : list
			Query IDs
		arg3 : list
			Relevance judgments
		arg4 : int
			The k value

		Returns
		-------
		float
			MRR value
		"""

		# meanReciprocalRank = -1

		#Fill in code here

		rrs = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_ids(query_id, qrels)
			rr = self.queryReciprocalRank(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			rrs.append(rr)
		return sum(rrs) / len(rrs) if rrs else 0.0