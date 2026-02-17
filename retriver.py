from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

load_dotenv()

LOGGER = logging.getLogger(__name__)
SEMANTIC_SCHOLAR_SEARCH_URL = "https://www.semanticscholar.org/api/1/search"
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))


def payload(keyword: str, page: int = 1) -> requests.Response:
    headers = {
        "Connection": "keep-alive",
        "sec-ch-ua": '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
        "Cache-Control": "no-cache,no-store,must-revalidate,max-age=-1",
        "Content-Type": "application/json",
        "sec-ch-ua-mobile": "?1",
        "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Mobile Safari/537.36",
        "X-S2-UI-Version": "20166f1745c44b856b4f85865c96d8406e69e24f",
        "sec-ch-ua-platform": '"Android"',
        "Accept": "*/*",
        "Origin": "https://www.semanticscholar.org",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://www.semanticscholar.org",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    }

    data = json.dumps(
        {
            "queryString": keyword.lower(),
            "page": page,
            "pageSize": 10,
            "sort": "relevance",
            "authors": [],
            "coAuthors": [],
            "venues": [],
            "requireViewablePdf": True,
            "fieldsOfStudy": [],
            "hydrateWithDdb": True,
            "includeTldrs": True,
            "performTitleMatch": True,
            "includeBadges": True,
            "getQuerySuggestions": False,
        }
    )

    response = requests.post(
        SEMANTIC_SCHOLAR_SEARCH_URL,
        headers=headers,
        data=data,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


def soup_html(output: requests.Response) -> list[dict[str, str]]:
    """Parse Semantic Scholar response."""
    final_result: list[dict[str, str]] = []
    output_json = output.json()
    for paper in output_json.get("results", []):
        result: dict[str, str] = {}
        result["title"] = paper.get("title", {}).get("text", "No title found")
        result["abstract"] = paper.get("tldr", {}).get("text", "No abstract/TLDR found")

        authors_list = paper.get("authors", [])
        if authors_list:
            author_names = [
                author_group[0].get("name", "N/A")
                for author_group in authors_list
                if author_group
            ]
            result["authors"] = ", ".join(author_names)
        else:
            result["authors"] = "No authors found"

        if paper.get("primaryPaperLink") and paper["primaryPaperLink"].get("url"):
            result["link"] = paper["primaryPaperLink"]["url"]
        elif paper.get("alternatePaperLinks"):
            result["link"] = paper["alternatePaperLinks"][0].get("url", "no_link_found")
        else:
            result["link"] = "no_link_found"

        final_result.append(result)

    return final_result


def retrive_paper(
    keyword: str,
    max_pages: int = 5,
    full_page_result: bool = False,
    api_wait: int = 1,
) -> list[dict[str, str]]:
    """Get paper metadata from Semantic Scholar."""
    del full_page_result  # compatibility placeholder

    all_pages: list[dict[str, str]] = []
    for page in tqdm(range(1, max_pages + 1)):
        try:
            ss_result = soup_html(payload(keyword, page=page))
            all_pages.extend(ss_result)
            time.sleep(api_wait)
        except requests.RequestException as exc:
            LOGGER.exception("Failed to fetch papers from Semantic Scholar on page %s", page)
            raise RuntimeError("Unable to fetch papers from Semantic Scholar") from exc

    return all_pages


def create_vectorstore(url: str, embedding: Any) -> FAISS:
    loader = PyPDFLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embedding=embedding)
    return db


def get_answer(llm: Any, db: FAISS, query: str) -> str:
    retriever = db.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
You are research assisitant.
Answer the user's question clearly and factually using the given context, but do not mention or reference the context explicitly.
If you don't know the answer, just say that you don't know.

<context>
{context}
</context>
Question: {input}
"""
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]
