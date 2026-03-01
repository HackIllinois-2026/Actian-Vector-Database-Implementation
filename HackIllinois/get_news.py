import sys

from google import genai

import os
import json

from eventregistry import *
import time

# Configuration, if user provided an argument or not for a different URL
SERVER = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
# Use unique name to avoid  file conflicts
COLLECTION = "News_Articles"
DIMENSION = 384  # We are using E5-small-v2


def main():
    article_topics = [
        "deepfake misinformation",
        "AI cyberattack",
        "AI security vulnerability",
        "machine learning security risks",
        "advanced AI capabilities",
        "AI data breach",
        "AI surveillance",
        "AI data center impact",
        "AI infrastructure expansion",
        "AI data privacy concerns",
        "AI generated content risks",
        "AI and IoT security",
        "AI environmental impact",
        "open source AI risks",
        "AI search engine",
        "AI coding assistant",
        "autonomous AI systems",
    ]

    resulting_news = []
    article_titles = []
    article_bodies = []
    article_body_summaries = []

    er_api_key = os.getenv("EVENTREGISTRY_API_KEY")

    er = EventRegistry(apiKey=er_api_key)

    # get the USA URI
    usUri = er.getLocationUri(
        "United_States"
    )  # = http://en.wikipedia.org/wiki/United_States
    print("US URI:", usUri)

    for i in article_topics:
        q = QueryArticlesIter(
            keywords=QueryItems.OR([i]),
            dataType=["news"],
        )
        articles = q.execQuery(er, sortBy="date", maxItems=8)  # maxItems per topic
        resulting_news.extend(articles)  # articles is an iterator

    print(f"Collected {len(resulting_news)} articles")
    for a in resulting_news:
        article_titles.append(a["title"])
        article_bodies.append(a["body"])

    print(article_titles)

    output_dir = "articles_output"
    os.makedirs(output_dir, exist_ok=True)

    # Write all article titles in one file
    with open(f"{output_dir}/all_titles.txt", "w", encoding="utf-8") as f:
        for title in article_titles:
            f.write(title + "\n")

    # Write all article bodies in one file
    with open(f"{output_dir}/all_bodies.txt", "w", encoding="utf-8") as f:
        for body in article_bodies:
            f.write(body + "\n\n")  # separate articles by a blank line

    # Write all article summaries in one file
    # with open(f"{output_dir}/all_summaries.txt", "w", encoding="utf-8") as f:
    #     for summary in article_body_summaries:
    #         f.write(summary + "\n\n")

    # Write full JSON of all articles in one file
    with open(f"{output_dir}/all_articles.json", "w", encoding="utf-8") as f:
        json.dump(resulting_news, f, indent=2)

    print("Files written successfully.")

    BATCH_SIZE = 3
    client = genai.Client()

    for start_idx in range(0, len(article_bodies), BATCH_SIZE):
        batch = article_bodies[start_idx : start_idx + BATCH_SIZE]

        # build a single prompt for the batch
        batch_prompt = ""
        for j, body in enumerate(batch):
            batch_prompt += (
                f"Article {j + 1}:\n"
                f"Summarize the following article body in exactly two sentences. The summary must describe only the outcome or result of the AI-related action discussed. Do not include commentary, explanations, formatting, or any text outside the two sentences. Respond with only the two sentences and nothing else."
                f":\n{body}\n\n"
            )

        # single API call for the batch
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=batch_prompt
        )

        # assuming the model returns all summaries separated by newlines, one per article
        if response.text:
            summaries = response.text.strip().split(
                "\n\n"
            )  # or another delimiter if needed
            article_body_summaries.extend(summaries)

        time.sleep(12)  # ensures max 5 requests/min

    print(article_body_summaries)


if __name__ == "__main__":
    main()
