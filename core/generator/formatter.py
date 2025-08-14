# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.10.0
"""

from typing import List

from core.retriever.types import Candidate


class AnswerFormatter:
    def format_answer_with_citations(self, answer: str, evidence: List[Candidate]) -> str:
        if not evidence:
            return answer

        citations = []
        for i, item in enumerate(evidence):
            citation_text = f"[Citation {i+1}]"
            link_parts = []
            if item.evidence.file_path:
                link_parts.append(f"File: {item.evidence.file_path.split('/')[-1]}")
            if item.evidence.page is not None:
                link_parts.append(f"Page: {item.evidence.page}")
            if item.evidence.caption:
                link_parts.append(f"Caption: {item.evidence.caption}")

            if link_parts:
                citations.append(f"{citation_text} ({', '.join(link_parts)})")
            else:
                citations.append(citation_text)

        return f"{answer}\n\n---\n\n{'\n'.join(citations)}"
