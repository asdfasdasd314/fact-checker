import time
from typing import List, Tuple
from dotenv import load_dotenv
from selenium import webdriver

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np
import torch

from flask import Flask, request, jsonify, send_file

from scrape import obtain_text
from logic import pick_evidence, classify_evidence, embed_hypothesis

load_dotenv()

app = Flask(__name__)

@app.route("/check", methods=["POST"])
def check():
    print(request.json)
    data = request.json
    hypothesis = data["hypothesis"]
    driver = webdriver.Chrome()
    text, links = obtain_text(hypothesis, driver, 4)
    driver.quit()
    evidence = pick_evidence(text, links, embed_hypothesis(hypothesis), 3, 100)
    print("Done picking evidence...")
    evidence = classify_evidence(evidence, hypothesis)
    return jsonify({"hypothesis": hypothesis, "evidence": [e.__dict__ for e in evidence]})


@app.route("/")
def index():
    return send_file("index.html")


if __name__ == "__main__":
    app.run(debug=True)