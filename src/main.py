from selenium import webdriver
import time
from typing import List, Tuple

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np
import torch

from rake_nltk import Rake

reasoning_model_name = "FacebookAI/roberta-large-mnli"

reasoning_model = pipeline('text-classification', model=reasoning_model_name, tokenizer=reasoning_model_name)

# Sample evaluation:
#
# QA_input = {
#     'question': 'Why is model conversion important?',
#     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
# }
# res = nlp(QA_input)

def create_search(hypothesis: str) -> str:
    r = Rake()
    r.extract_keywords_from_text(hypothesis)
    return " ".join(r.get_ranked_phrases()[:3])


def process_text(text: str) -> List[str]:
    text = text.split(" ")
    text = [word.strip().lower() for word in text if word.strip()]
    return text


def create_windows(text: List[str], window_size: int) -> List[List[str]]:
    windows = []
    for i in range(len(text) - window_size + 1):
        windows.append(" ".join(text[i:i+window_size]))

    if len(windows) == 0:
        windows.append(" ".join(text))

    return windows


k = 5
window_size = 100 # Number of words to be included as text evidence

top_k_evidence = [(float("-inf"), None, None, None) for _ in range(k)]

evidence_model = SentenceTransformer("all-MiniLM-L6-v2")
hypothesis = input("Enter a hypothesis that can be supported or contradicted by text evidence: ")

hypothesis_embedding = evidence_model.encode(hypothesis)

url = lambda query: 'https://www.congress.gov/search?q={"source"%3A"congrecord"%2C"search"%3A"' + query + '"}'

driver = webdriver.Chrome()

search = create_search(hypothesis)
print(search)
driver.get(url(search))

# For some reason I have to do all of this, but frankly it's fine I don't care
results_list = driver.find_element(By.CSS_SELECTOR, ".basic-search-results-lists")

results: List[WebElement] = []
for result in results_list.find_elements(By.TAG_NAME, "li"):
    if result.get_attribute("class") == "expanded":
        results.append(result)

spans = []
for result in results:
    for span in result.find_elements(By.TAG_NAME, "span"):
        if span.get_attribute("class") == "result-heading congressional-record-heading":
            spans.append(span)
            break

links = [span.find_element(By.TAG_NAME, "a").get_attribute("href") for span in spans][1:5]

for i in range(len(links)):
    driver.get(links[i])

    try:
        content = driver.find_element(By.CSS_SELECTOR, ".styled")

    except:
        print("Source couldn't be accessed: " + links[i])
        continue

    processed = process_text(content.text)
    windows = create_windows(processed, window_size)
    embeddings = evidence_model.encode(windows)
    cosines = util.cos_sim(hypothesis_embedding, embeddings).squeeze()
    ordered = sorted(zip(cosines, range(len(cosines))), key=lambda x: x[0], reverse=True)

    for j in range(len(ordered)):
        for l in range(k):
            if ordered[j][0] > top_k_evidence[l][0]:
                top_k_evidence.insert(l, (ordered[j][0], windows[ordered[j][1]], embeddings[ordered[j][1]], links[i]))
                top_k_evidence.pop()
                break

driver.quit()

inputs = [evidence[1] + "</s>" + hypothesis for evidence in top_k_evidence]
outputs = [reasoning_model(inp)[0] for inp in inputs]

# If everything is NEUTRAL, then nothing was found
# If something is ENTAILMENT, then something was found
# If something is CONTRADICTION, then something was found but it was the wrong thing

labels = [output['label'] for output in outputs]
for i in range(k):
    if labels[i] == 'ENTAILMENT':
        print()
        print("Possible support for hypothesis (p-value: " + str(outputs[i]['score']) + "):")
        print("Source: " + top_k_evidence[i][3])
        print("Evidence: " + top_k_evidence[i][1])
    if labels[i] == 'CONTRADICTION':
        print()
        print("Possible contradiction to hypothesis (p-value: " + str(outputs[i]['score']) + "):")
        print("Source: " + top_k_evidence[i][3])
        print("Evidence: " + top_k_evidence[i][1])

if 'ENTAILMENT' not in labels and 'CONTRADICTION' not in labels:
    print("Nothing was found")