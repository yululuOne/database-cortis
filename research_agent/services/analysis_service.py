# services/analysis_service.py

"""
Analysis Service

Uses an LLM + FAISS retriever to produce structured report analysis
and answer ad-hoc questions grounded in retrieved chunks.
"""

import re
from typing import List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from prompts.research_prompt import (
    REPORT_SYSTEM,
    REPORT_HUMAN,
    QA_CITE_SYSTEM,
    QA_CITE_HUMAN,
)


def _retrieve_context(store: FAISS, query: str, k: int) -> str:
    """Retrieve top-k chunks and format them as numbered context."""
    docs = store.similarity_search(query, k=k)
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(parts)


def format_docs_with_citations(docs) -> Tuple[str, list]:
    """Format retrieved docs as numbered context string for citation-aware QA."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(parts), docs


def analyze_report(store: FAISS, llm, k: int = 10) -> str:
    context = _retrieve_context(
        store,
        query="这份研报的核心观点、投资逻辑、关键数据和风险因素",
        k=k,
    )
    messages = [
        SystemMessage(content=REPORT_SYSTEM),
        HumanMessage(content=REPORT_HUMAN.format(context=context)),
    ]
    response = llm.invoke(messages)
    return response.content


def answer_question(store: FAISS, llm, question: str, k: int = 4) -> str:
    """Answer an ad-hoc question using retrieved chunks as evidence."""
    context = _retrieve_context(store, query=question, k=k)
    messages = [
        SystemMessage(content=QA_CITE_SYSTEM),
        HumanMessage(content=QA_CITE_HUMAN.format(question=question, context=context)),
    ]
    response = llm.invoke(messages)
    return response.content


def _extract_citation_ids(answer: str) -> List[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", answer)]


def _validate_citations(ids: List[int], k: int) -> bool:
    if not ids:
        return False
    return all(1 <= i <= k for i in ids)


def answer_question_with_citations(store, llm, question: str, k: int = 4) -> Tuple[str, List[int], list]:
    """
    Returns:
    - answer_text (natural language with [n] citations)
    - cited_ids (extracted citation numbers)
    - retrieved_docs (top-k docs for UI display)
    """
    docs = store.similarity_search(question, k=k)
    context, _ = format_docs_with_citations(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_CITE_SYSTEM),
        ("human", QA_CITE_HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": question, "context": context})
    cited = _extract_citation_ids(answer)

    if not _validate_citations(cited, k=k):
        fixer = ChatPromptTemplate.from_messages([
            ("system",
             "你是引用修复器。你必须输出自然语言回答（不要JSON/Markdown），并在关键结论句末尾添加正确的证据编号引用。"
             f"引用编号只能使用[1]到[{k}]。"),
            ("human",
             "问题：{question}\n\n证据片段（带编号）：\n{context}\n\n"
             "下面这段回答缺少或引用错误，请你重写并补上正确引用：\n{bad_answer}"),
        ])
        fixed = (fixer | llm | StrOutputParser()).invoke({
            "question": question,
            "context": context,
            "bad_answer": answer,
        })
        answer = fixed
        cited = _extract_citation_ids(answer)

    return answer, sorted(set(cited)), docs
