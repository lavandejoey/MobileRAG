# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.7.0
"""

from unittest.mock import Mock

import pytest
import torch

from core.config.settings import Settings
from core.reranker.reranker import Reranker
from core.retriever.types import Candidate, Evidence


@pytest.fixture
def mock_settings():
    return Settings(device="cpu")


@pytest.fixture
def mock_reranker_model_and_tokenizer():
    mock_tokenizer_instance = Mock()
    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer_instance.items.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }.items()  # Mock the items() method
    mock_tokenizer_from_pretrained = Mock(return_value=mock_tokenizer_instance)

    mock_model_instance = Mock()
    mock_model_instance.logits = torch.tensor([[0.1], [0.9], [0.5]])  # Scores for 3 candidates
    mock_model_from_pretrained = Mock(return_value=mock_model_instance)

    return mock_tokenizer_from_pretrained, mock_model_from_pretrained


@pytest.fixture
def reranker(mock_settings, mock_reranker_model_and_tokenizer):
    mock_tokenizer_instance, mock_model_instance = mock_reranker_model_and_tokenizer
    reranker_instance = Reranker(mock_settings.device)
    reranker_instance.tokenizer = mock_tokenizer_instance
    reranker_instance.model = mock_model_instance
    return reranker_instance


def test_reranker_rank_topr(reranker):
    query = "test query"
    candidates = [
        Candidate(
            id="cand1",
            score=0.5,
            text="Candidate one",
            evidence=Evidence(file_path="f1"),
            lang="en",
            modality="text",
        ),
        Candidate(
            id="cand2",
            score=0.2,
            text="Candidate two",
            evidence=Evidence(file_path="f2"),
            lang="en",
            modality="text",
        ),
        Candidate(
            id="cand3",
            score=0.8,
            text="Candidate three",
            evidence=Evidence(file_path="f3"),
            lang="en",
            modality="text",
        ),
    ]

    # Test with topr = 2
    reranked_candidates = reranker.rank(query, candidates, topr=2)
    assert len(reranked_candidates) == 2
    assert reranked_candidates[0].id == "cand2"  # Score 0.9
    assert reranked_candidates[1].id == "cand3"  # Score 0.5

    # Test with topr = 3
    reranked_candidates = reranker.rank(query, candidates, topr=3)
    assert len(reranked_candidates) == 3
    assert reranked_candidates[0].id == "cand2"
    assert reranked_candidates[1].id == "cand3"
    assert reranked_candidates[2].id == "cand1"  # Score 0.1


def test_reranker_empty_candidates(reranker):
    query = "test query"
    candidates = []
    reranked_candidates = reranker.rank(query, candidates)
    assert len(reranked_candidates) == 0
