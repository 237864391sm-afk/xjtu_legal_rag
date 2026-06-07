import math
from collections import Counter
from src.data import MOCK_STATUTES, MOCK_CASES

class SimpleBM25:
    def __init__(self, docs):
        self.raw_docs = docs
        # 采用字级别分词
        self.docs = [list(doc["content"]) for doc in docs]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.docs) if self.docs else 1

        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

        self.idf = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log(1 + (len(self.docs) - freq + 0.5) / (freq + 0.5))

        self.k1 = 1.5
        self.b = 0.75

    def retrieve(self, query: str, top_k: int = 2):
        q_tokens = list(query)
        scores = []
        for idx, doc in enumerate(self.docs):
            score = 0
            doc_counter = Counter(doc)
            for q in q_tokens:
                if q in doc_counter:
                    tf = doc_counter[q]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                    score += self.idf[q] * (numerator / denominator)
            scores.append((score, self.raw_docs[idx]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k] if score > 0]


class HybridRetriever:
    def __init__(self):
        self.statute_bm25 = SimpleBM25(MOCK_STATUTES)
        self.case_bm25 = SimpleBM25(MOCK_CASES)

    def retrieve(self, query: str, route_target: str, top_k: int = 2):
        results = []
        if route_target in ["statute", "both"]:
            results.extend(self.statute_bm25.retrieve(query, top_k))
        if route_target in ["case", "both"]:
            results.extend(self.case_bm25.retrieve(query, top_k))
        return results[:top_k]