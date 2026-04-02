import json
import numpy as np
import argparse
import re

from tqdm import tqdm
from wordfreq import word_frequency
from keybert import KeyBERT
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

EPSILON = 1e-15

def get_data_from_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def get_keybert_scores(
    data: list, 
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'
):
    st_model = SentenceTransformer(model_name)
    kw_model = KeyBERT(model=st_model)

    tmp_data = defaultdict(lambda: {"scores": [], "language": ""})

    for item in tqdm(data, desc="Extracting Keywords"):
        text = item.get("text", "")
        if not text:
            continue

        keybert_keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words=None,
            use_mmr=True
        )

        lang = item.get("metadata", {}).get("language", "en")

        for key, score in keybert_keywords:
            tmp_data[key]["scores"].append(score)
            tmp_data[key]["language"] = lang

    results = []
    
    for keyword, info in tmp_data.items():
        avg_score = np.mean(info["scores"])

        results.append({
            "keyword": keyword,
            "avg_score": round(float(avg_score), 3),
            "language": info["language"],
            "occurrence": len(info["scores"])
        })

    return results

def min_max_scaler(data):
    if not data:
        return []
    
    raw_scores = np.array([item[1] for item in data]).reshape(-1, 1)
    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_scores = mm_scaler.fit_transform(raw_scores)

    return [
        {"keyword": data[i][0], "score": round(float(normalized_scores[i][0]), 3)}
        for i in range(len(data))
    ]

def get_keywords_with_scores(data, codes, top_k, threshold=0.7):
    keybert_results = get_keybert_scores(data)

    results = []
    codes_set = {item["keyword"].lower() for item in codes}

    for item in tqdm(keybert_results, desc="Computing Final Score"):
        if item["keyword"] in codes_set:
            continue

        wordfreq = word_frequency(item["keyword"], item["language"])
        score = item["avg_score"] / (wordfreq + EPSILON)
        results.append((item["keyword"], score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    results = min_max_scaler(results)
    results = [item for item in results if item["score"] > threshold]
    return results[:top_k]

def get_codes(data):
    re_codes = re.compile(r"^[A-Z]{1,3}[0-9]{3,}$")

    codes = set()
    for item in tqdm(data, desc="Extracting codes"):
        title = item.get("metadata", {}).get("title")
        if title and re_codes.match(title):
            codes.add(title)

    return [
        {"keyword": code, "score": 1.0}
        for code in sorted(list(codes))
    ]

def save_results_jsonl(data_list, path):
    with open(path, "w", encoding="utf-8") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Dataset Path (JSONL file)")
    parser.add_argument("--kw_path", type=str, required=True, help="Keywords final path")
    parser.add_argument("--top_k", type=int, required=True, help="Max number of keywords to extract")
    parser.add_argument("--thr", type=float, default=0.7, help="Threshold")
    args = parser.parse_args()
    
    data = get_data_from_jsonl(args.dataset)
    codes = get_codes(data)

    keywords = get_keywords_with_scores(data, codes=codes, top_k=args.top_k, threshold=args.thr)

    final_results = codes + keywords

    save_results_jsonl(final_results, path=args.kw_path)