import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def score_tfidf_relevance(pairs_df):
    df = pairs_df.copy()
    corpus = pd.concat([df["query"], df["title"]]).astype(str).tolist()
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    vec.fit(corpus)
    qv = vec.transform(df["query"].astype(str))
    tv = vec.transform(df["title"].astype(str))
    scores = np.asarray(qv.multiply(tv).sum(axis=1)).reshape(-1)
    df["bm25_tfidf_score"] = scores
    return df
