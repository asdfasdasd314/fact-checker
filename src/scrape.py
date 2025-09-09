from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from typing import List, Tuple

from rake_nltk import Rake

def create_search(hypothesis: str) -> str:
    r = Rake()
    r.extract_keywords_from_text(hypothesis)
    return " ".join(r.get_ranked_phrases()[:3])


def url(query: str) -> str:
    return 'https://www.congress.gov/search?q={"source"%3A"congrecord"%2C"search"%3A"' + query + '"}'


def obtain_result_elements(results_list: WebElement) -> List[WebElement]:
    results: List[WebElement] = []
    for result in results_list.find_elements(By.TAG_NAME, "li"):
        if result.get_attribute("class") == "expanded":
            results.append(result)
    return results


def obtain_span_element(result: WebElement) -> WebElement:
    for span in result.find_elements(By.TAG_NAME, "span"):
        if span.get_attribute("class") == "result-heading congressional-record-heading":
            return span
    return None


def obtain_text(hypothesis: str, driver: webdriver.Chrome, num_results: int) -> Tuple[List[List[str]], List[str]]:
    search = create_search(hypothesis)
    driver.get(url(search))

    # For some reason I have to do all of this, but frankly it's fine I don't care
    results_list = driver.find_element(By.CSS_SELECTOR, ".basic-search-results-lists")

    results = obtain_result_elements(results_list)

    spans = []
    for result in results:
        span = obtain_span_element(result)
        if span:
            spans.append(span)

    links = [span.find_element(By.TAG_NAME, "a").get_attribute("href") for span in spans][:num_results]

    all_sentences = []
    for i in range(len(links)):
        driver.get(links[i])

        try:
            content = driver.find_element(By.CSS_SELECTOR, ".styled")

        except:
            print("Source couldn't be accessed: " + links[i])
            continue

        all_sentences.append(content.text.split(" "))

    return all_sentences, links