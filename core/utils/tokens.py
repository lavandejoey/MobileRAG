# -*- coding: utf-8 -*-
"""
@file: core/utils/tokens.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.1.0
"""

import tiktoken


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Counts the number of tokens in a text string using tiktoken.

    Args:
        text: The text to count tokens for.
        model: The name of the encoding to use.

    Returns:
        The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))
