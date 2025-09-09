from typing import List
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np

from scrape import obtain_text

class Evidence:

    def __init__(self, text: str, source: str, score: float, label: str):
        self.text = text
        self.source = source
        self.score = score
        self.label = label


reasoning_model_name = "FacebookAI/roberta-large-mnli"
reasoning_model = pipeline('text-classification', model=reasoning_model_name, tokenizer=reasoning_model_name)
evidence_model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess_text(text: List[str]) -> List[str]:
    return [t.replace("\n", "").replace("\r", "").replace("\t", "").strip() for t in text]

def pick_evidence(text: List[List[str]], links: List[str], hypothesis_embedding: np.ndarray, k: int, window_size: int) -> List[List[str]]:
    top_k_evidence = [(float("-inf"), None, None, None) for _ in range(k)]
    
    for i in range(len(text)):
        text[i] = preprocess_text(text[i])
        windows = create_windows(text[i], window_size)
        embeddings = evidence_model.encode(windows)
        cosines = util.cos_sim(hypothesis_embedding, embeddings).squeeze()
        ordered = sorted(zip(cosines, range(len(cosines))), key=lambda x: x[0], reverse=True)
        
        for j in range(len(ordered)):
            for l in range(k):
                if ordered[j][0] > top_k_evidence[l][0]:
                    top_k_evidence.insert(l, (ordered[j][0], windows[ordered[j][1]], embeddings[ordered[j][1]], links[i]))
                    top_k_evidence.pop()
                    break

    return top_k_evidence


def create_windows(text: List[str], window_size: int) -> List[List[str]]:
    windows = []
    for i in range(len(text) - window_size + 1):
        windows.append(" ".join(text[i:i+window_size]))

    if len(windows) == 0:
        windows.append(" ".join(text))

    return windows


def embed_hypothesis(hypothesis: str) -> np.ndarray:
    return evidence_model.encode(hypothesis)


def classify_evidence(top_k_evidence: List[List[str]], hypothesis: str) -> List[Evidence]:
    inputs = [evidence[1] + "</s>" + hypothesis for evidence in top_k_evidence]
    outputs = [reasoning_model(inp)[0] for inp in inputs]
    return [Evidence(top_k_evidence[i][1], top_k_evidence[i][3], outputs[i]['score'], outputs[i]['label']) for i in range(len(outputs))]

# If everything is NEUTRAL, then nothing was found
# If something is ENTAILMENT, then something was found
# If something is CONTRADICTION, then something was found but it was the wrong thing