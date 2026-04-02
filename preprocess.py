import glob
import os
import nltk
import json
import argparse

from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect

def table_to_markdown(table):
    rows = table.find_all("tr")
    if not rows:
        return ""
    
    md_rows = []
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        cell_texts = [c.get_text(strip=True) for c in cells]
        md_rows.append("| " + " | ".join(cell_texts) + " |")
        
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
    
    return "\n".join(md_rows)

def create_json_data(paths):
    data = []
    for path in tqdm(paths):
        ext = path.split(".")[-1]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if ext == "html":
            soup = BeautifulSoup(content, "html.parser")
            title = soup.title.string if soup.title else None

            if soup.head:
                soup.head.decompose()

            for tag in soup(["aside", "style", "footer", "script", "img", "iframe", "form", "noscript", "a", "button"]):
                tag.decompose()

            selectors = [
                "div.skipnav",
                "div.toolbar.top-nav-on",
                "div.section-toc.section-toc-before",
                "div.breadcrumb-container",
                "div.top-pager",
                "div.feedback-panel"
            ]
            
            for selector in selectors:
                for element in soup.select(selector):
                    element.decompose()

            for table in soup.find_all("table"):
                md = table_to_markdown(table)
                table.replace_with(md)

            text = soup.get_text(separator="\n", strip=True)
        else:
            title = None
            text = content.strip()

        lang = Path(path).parent.name

        if len(lang) != 2 or not lang.isalpha():
            lang = detect(text)
    
        data.append({
            "text": text,
            "metadata": {
                "title": title,
                "language": lang,
                "file_path": path
            }
        })
    return data

def lemmatize_text(text):
    if not text:
        return ""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    return " ".join(lemmatized)

def save_jsonl(data_list, path):
    with open(path, "w", encoding="utf-8") as f:
        for data in data_list:
            item = data.copy()
            item["lemmatized_text"] = lemmatize_text(data["text"])
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root folder Alarms and Manuals")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder JSONL files.")

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    alarms = root / "Alarms"
    manuals = root / "Manuals"
    alarms_jsonl_path = f"{output_dir}/alarms.jsonl"
    manuals_jsonl_path = f"{output_dir}/manuals.jsonl"
    dataset_jsonl_path = f"{output_dir}/dataset.jsonl"
    valid_ext = ["html", "txt"]

    nltk.download('punkt_tab')
    nltk.download('wordnet')

    alarms_paths = []
    manuals_paths = []

    for file_path in glob.iglob(str(alarms / "**" / "*"), recursive=True):
        if os.path.isfile(file_path) and file_path.split(".")[-1] in valid_ext:
            alarms_paths.append(file_path)

    for file_path in glob.iglob(str(manuals / "**" / "*"), recursive=True):
        if os.path.isfile(file_path) and file_path.split(".")[-1] in valid_ext:
            manuals_paths.append(file_path)

    alarms_data = create_json_data(alarms_paths)
    manuals_data = create_json_data(manuals_paths)

    os.makedirs(output_dir, exist_ok=True)

    save_jsonl(alarms_data, alarms_jsonl_path)
    save_jsonl(manuals_data, manuals_jsonl_path)
    save_jsonl(alarms_data + manuals_data, dataset_jsonl_path)
