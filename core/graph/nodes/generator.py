# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from core.config.settings import Settings
from core.generator.llm import LLMGenerator

settings = Settings()
llm_generator = LLMGenerator(settings)


def generator_node(state):
    """
    Generates a response using the LLM.
    """
    prompt = "\n\n".join(
        [
            state["budget"]["summary"],
            "\n".join(state["budget"]["recent_messages"]),
            "\n".join(state["budget"]["memories"]),
            "\n".join(state["budget"]["evidence"]),
            state["normalized_query"],
        ]
    )
    generation = llm_generator.generate(prompt)
    return {"generation": generation}
