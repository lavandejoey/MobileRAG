# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.10.0
"""

import pytest

from core.generator.formatter import AnswerFormatter
from core.retriever.types import Candidate, Evidence


@pytest.fixture
def answer_formatter():
    return AnswerFormatter()


def test_format_answer_with_citations_no_evidence(answer_formatter):
    answer = "This is a test answer."
    formatted_answer = answer_formatter.format_answer_with_citations(answer, [])
    assert formatted_answer == answer


def test_format_answer_with_citations_basic(answer_formatter):
    answer = "This is a test answer."
    evidence = [
        Candidate(
            id="1",
            score=0.9,
            text="",
            evidence=Evidence(file_path="doc1.pdf", page=10),
            lang="en",
            modality="text",
        ),
    ]
    expected_output = "This is a test answer.\n\n---\n\n[Citation 1] (File: doc1.pdf, Page: 10)"
    formatted_answer = answer_formatter.format_answer_with_citations(answer, evidence)
    assert formatted_answer == expected_output


def test_format_answer_with_citations_multiple_evidence(answer_formatter):
    answer = "This is a test answer."
    evidence = [
        Candidate(
            id="1",
            score=0.9,
            text="",
            evidence=Evidence(file_path="doc1.pdf", page=10),
            lang="en",
            modality="text",
        ),
        Candidate(
            id="2",
            score=0.8,
            text="",
            evidence=Evidence(file_path="image.png", caption="A beautiful image."),
            lang="en",
            modality="image",
        ),
        Candidate(
            id="3",
            score=0.7,
            text="",
            evidence=Evidence(file_path="report.docx"),
            lang="en",
            modality="text",
        ),
    ]
    expected_output = (
        "This is a test answer.\n\n---\n\n"
        "[Citation 1] (File: doc1.pdf, Page: 10)\n"
        "[Citation 2] (File: image.png, Caption: A beautiful image.)\n"
        "[Citation 3] (File: report.docx)"
    )
    formatted_answer = answer_formatter.format_answer_with_citations(answer, evidence)
    assert formatted_answer == expected_output


def test_format_answer_with_citations_only_caption(answer_formatter):
    answer = "This is a test answer."
    evidence = [
        Candidate(
            id="1",
            score=0.9,
            text="",
            evidence=Evidence(file_path="diagram.png", caption="A diagram."),
            lang="en",
            modality="image",
        ),
    ]
    expected_output = (
        "This is a test answer.\n\n---\n\n[Citation 1] (File: diagram.png, Caption: A diagram.)"
    )
    formatted_answer = answer_formatter.format_answer_with_citations(answer, evidence)
    assert formatted_answer == expected_output
