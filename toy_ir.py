# Part 1 of Project (Problem Statement): Toy IR System Analysis.
import re
import math
from collections import Counter
from informationRetrieval import InformationRetrieval

# Corpus
DOCS_RAW = [
    "The star in our solar system provides heat and light.",
    "That Hollywood star walked the red carpet for the movie premiere.",
    "Astronomers observe distant stars and galaxies using telescopes.",
]
DOC_IDS = [1, 2, 3]

STOPWORDS = {"the", "in", "our", "and", "that", "for"}


# Preprocessing

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text):
    return remove_stopwords(tokenize(text))

def calc_tfidf(preprocessed_docs, doc_ids):
    N = len(doc_ids)
    raw_tf = {}
    for i, tokens in enumerate(preprocessed_docs):
        raw_tf[doc_ids[i]] = Counter(tokens)
    df = Counter()
    for tf in raw_tf.values():
        for term in tf:
            df[term] += 1
    vocab = sorted(df.keys())
    idf = {term: math.log10(N / df[term]) for term in vocab}
    doc_tfidf = {}
    doc_norms = {}
    for doc_id in doc_ids:
        vec = {}
        for term, tf in raw_tf[doc_id].items():
            tf_w = (1 + math.log10(tf)) if tf > 0 else 0.0
            vec[term] = tf_w * idf[term]
        doc_tfidf[doc_id] = vec
        doc_norms[doc_id] = math.sqrt(sum(w * w for w in vec.values()))
    return vocab, idf, raw_tf, doc_tfidf, doc_norms


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def posting_docids(postings):
    if not postings:
        return []
    return [p[0] if isinstance(p, tuple) else p for p in postings]

def boolean_retrieve(index, query_tokens):
    sets = []
    for term in query_tokens:
        if term not in index:
            return []
        sets.append(set(posting_docids(index[term])))
    if not sets:
        return []
    result = sets[0]
    for s in sets[1:]:
        result = result & s
    return sorted(result)


def main():
    print_header("STEP 1: Tokenization + Stopword Removal")
    preprocessed_docs = []
    flat_docs = []
    for i, doc in enumerate(DOCS_RAW):
        tokens = tokenize(doc)
        filtered = remove_stopwords(tokens)
        preprocessed_docs.append([filtered])
        flat_docs.append(filtered)
        print(f"\nd{DOC_IDS[i]} raw:       {doc}")
        print(f"d{DOC_IDS[i]} tokens:    {tokens}")
        print(f"d{DOC_IDS[i]} filtered:  {filtered}")

    vocab, idf, raw_tf, doc_tfidf, doc_norms = calc_tfidf(flat_docs, DOC_IDS)

    ir = InformationRetrieval()
    ir.buildIndex(preprocessed_docs, DOC_IDS)

    print_header("STEP 2: Inverted Index")
    print(f"{'Term':<15}{'Postings (docID, tf)'}")
    print("-" * 50)
    for term in sorted(ir.index.keys()):
        postings = sorted(posting_docids(ir.index[term]))
        print(f"{term:<15}{postings}")

    print_header("STEP 3: TF Matrix (Raw Counts)")
    header = f"{'Term':<15}" + "".join(f"d{d:<6}" for d in DOC_IDS)
    print(header)
    print("-" * len(header))
    for term in vocab:
        row = f"{term:<15}"
        for d in DOC_IDS:
            row += f"{raw_tf[d].get(term, 0):<7}"
        print(row)

    print_header("STEP 4: IDF Values (log10(N/df))")
    N = len(DOC_IDS)
    print(f"N = {N}")
    print(f"{'Term':<15}{'df':<5}{'N/df':<10}{'IDF':<10}")
    print("-" * 40)
    for term in vocab:
        df_val = sum(1 for d in DOC_IDS if raw_tf[d].get(term, 0) > 0)
        print(f"{term:<15}{df_val:<5}{N/df_val:<10.4f}{idf[term]:<10.4f}")

    print_header("STEP 5: TF-IDF Matrix (1+log10(tf)) * IDF")
    print(header)
    print("-" * len(header))
    for term in vocab:
        row = f"{term:<15}"
        for d in DOC_IDS:
            row += f"{doc_tfidf[d].get(term, 0.0):<7.4f}"
        print(row)

    print("\nDocument vector norms:")
    for d in DOC_IDS:
        print(f"  ||d{d}|| = {doc_norms[d]:.4f}")

    print_header("STEP 6: Boolean Retrieval - Query = 'star light'")
    q1_tokens = preprocess("star light")
    print(f"Query tokens: {q1_tokens}")
    print("\nInverted index lookup:")
    for term in q1_tokens:
        postings = ir.index.get(term, [])
        docs_with_term = sorted(set(posting_docids(postings)))
        print(f"  '{term}' -> {docs_with_term}")
    bool_result = boolean_retrieve(ir.index, q1_tokens)
    print(f"\nBoolean (AND) result: {bool_result}")

    print_header("STEP 7: Cosine Similarity Ranking - Query = 'star light'")
    q1_counter = Counter(q1_tokens)
    q1_tfidf = {}
    for term, tf in q1_counter.items():
        if term in idf:
            q1_tfidf[term] = (1 + math.log10(tf)) * idf[term]
    q1_norm = math.sqrt(sum(w * w for w in q1_tfidf.values())) if q1_tfidf else 0.0
    print("Query TF-IDF vector:")
    for t, w in q1_tfidf.items():
        print(f"  {t}: {w:.4f}")
    print(f"Query norm = {q1_norm:.4f}")
    ranking1 = ir.rank([[q1_tokens]])[0]
    print("\nCosine similarities:")
    for d in DOC_IDS:
        if q1_norm == 0 or doc_norms[d] == 0:
            sim = 0.0
        else:
            dot = sum(q1_tfidf.get(t, 0) * doc_tfidf[d].get(t, 0) for t in q1_tfidf)
            sim = dot / (q1_norm * doc_norms[d])
        print(f"  sim(q, d{d}) = {sim:.4f}")
    print(f"\nFinal ranking: {ranking1}")

    print_header("STEP 8: Word Sense Ambiguity - Query = 'movie star'")
    q2_tokens = preprocess("movie star")
    print(f"Query tokens: {q2_tokens}")
    q2_counter = Counter(q2_tokens)
    q2_tfidf = {}
    for term, tf in q2_counter.items():
        if term in idf:
            q2_tfidf[term] = (1 + math.log10(tf)) * idf[term]
    q2_norm = math.sqrt(sum(w * w for w in q2_tfidf.values())) if q2_tfidf else 0.0
    print("\nQuery TF-IDF vector:")
    for t, w in q2_tfidf.items():
        print(f"  {t}: {w:.4f}")
    print(f"Query norm = {q2_norm:.4f}")
    ranking2 = ir.rank([[q2_tokens]])[0]
    print("\nCosine similarities:")
    for d in DOC_IDS:
        if q2_norm == 0 or doc_norms[d] == 0:
            sim = 0.0
        else:
            dot = sum(q2_tfidf.get(t, 0) * doc_tfidf[d].get(t, 0) for t in q2_tfidf)
            sim = dot / (q2_norm * doc_norms[d])
        print(f"  sim(q, d{d}) = {sim:.4f}")
    print(f"\nFinal ranking: {ranking2}")


if __name__ == "__main__":
    main()