import numpy as np
import pandas as pd

def generate_synthetic_products(n_products=160, n_queries=60, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["electronics", "home", "beauty", "grocery", "books", "fashion", "sports", "auto"]
    modifiers = ["premium", "budget", "compact", "durable", "wireless", "organic", "portable", "professional"]
    products = []
    for i in range(n_products):
        cat = categories[i % len(categories)]
        mod = modifiers[(i // len(categories)) % len(modifiers)]
        brand = f"brand_{i % 20}"
        products.append({"product_id": str(i), "product_category": cat, "brand": brand, "title": f"{brand} {mod} {cat} product {i}", "modifier": mod, "is_long_tail": i > int(n_products * 0.7)})
    products = pd.DataFrame(products)
    queries = []
    for j in range(n_queries):
        cat = categories[j % len(categories)]; mod = modifiers[j % len(modifiers)]
        queries.append({"query_id": str(j), "query": f"{mod} {cat}", "query_category": cat, "query_modifier": mod})
    queries = pd.DataFrame(queries)
    rows = []
    for _, q in queries.iterrows():
        relevant = products[(products.product_category == q.query_category) & (products.modifier == q.query_modifier)].head(8)
        sub_pool = products[(products.product_category == q.query_category) & (products.modifier != q.query_modifier)]
        hard_pool = products[(products.product_category != q.query_category) & (products.modifier == q.query_modifier)]
        substitutes = sub_pool.sample(min(8, len(sub_pool)), random_state=seed+int(q.query_id)) if len(sub_pool) else sub_pool
        hard_neg = hard_pool.sample(min(8, len(hard_pool)), random_state=seed+100+int(q.query_id)) if len(hard_pool) else hard_pool
        random_neg = products.sample(min(12, len(products)), random_state=seed+200+int(q.query_id))
        sample = pd.concat([relevant, substitutes, hard_neg, random_neg]).drop_duplicates("product_id").head(40)
        for _, p in sample.iterrows():
            label = 3 if (p.product_category == q.query_category and p.modifier == q.query_modifier) else 2 if p.product_category == q.query_category else 1 if p.modifier == q.query_modifier else 0
            rows.append({**q.to_dict(), **p.to_dict(), "esci_label_num": int(label)})
    return products, queries, pd.DataFrame(rows)
