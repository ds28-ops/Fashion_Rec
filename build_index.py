#!/usr/bin/env python3
import os, argparse
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI

def get_args():
    P = argparse.ArgumentParser()
    P.add_argument("--csv", default="data.csv")
    P.add_argument("--images_dir", default="data")   # folder with images
    P.add_argument("--out_index", default="index.faiss")
    P.add_argument("--out_meta",  default="meta.parquet")
    P.add_argument("--limit", type=int, default=0)   # for quick tests; 0 = all
    return P.parse_args()

def main():
    args = get_args()

    # OpenAI key from env
    client = OpenAI()  # expects OPENAI_API_KEY in env

    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.iloc[:args.limit].copy()

    df["description"] = df["description"].fillna(df["display name"])
    df["category"]    = df["category"].fillna("")
    df["combined_text"] = df.apply(
        lambda x: f"{x['display name']}. {x['description']}. Category: {x['category']}",
        axis=1
    )
    df["image_path"] = df["image"].apply(lambda x: os.path.join(args.images_dir, x))

    def embed_batch(texts):
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        embs = [np.array(r.embedding, dtype=np.float32) for r in resp.data]
        return np.vstack(embs)

    # batch to reduce API overhead
    BATCH = 64
    all_embs = []
    for i in tqdm(range(0, len(df), BATCH), desc="Embedding"):
        batch = df["combined_text"].iloc[i:i+BATCH].tolist()
        all_embs.append(embed_batch(batch))
    embs = np.vstack(all_embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    faiss.write_index(index, args.out_index)
    df[["display name","description","category","image_path"]].to_parquet(args.out_meta, index=False)
    print(f"Saved: {args.out_index}, {args.out_meta} (rows={len(df)})")

if __name__ == "__main__":
    main()
