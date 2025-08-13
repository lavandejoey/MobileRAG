from core.utils.tokens import count_tokens


def test_count_tokens_simple():
    assert count_tokens("hello world") == 2


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_chinese():
    assert count_tokens("你好世界") == 5
