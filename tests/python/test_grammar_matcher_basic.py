"""Test the basic functionality of GrammarMatcher."""

import sys
from typing import List, Optional, Union

import pytest
import torch
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import (
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar,
    _get_matcher_from_grammar_and_tokenizer_info,
    _is_grammar_accept_string,
)

json_grammar = xgr.Grammar.builtin_json_grammar()


grammar__input__accepted__test_accept_string = [
    ("""root ::= [^a]+""", "bbb", True),
    ("""root ::= [^a]+""", "bba", False),
    ("""root ::= [^a]+""", "©", True),
    ("""root ::= [^a]+""", b"\xe2\xa1\xa1", True),
    ("""root ::= [^a]+""", b"\xe2\xa1\xa1\xa1", False),
    ("""root ::= [^a]+""", b"\xe2\xa1\xe2\xa1", False),
]


@pytest.mark.parametrize("grammar, input, accepted", grammar__input__accepted__test_accept_string)
def test_accept_string(grammar: str, input: Union[str, bytes], accepted: bool):
    matcher = _get_matcher_from_grammar(grammar)
    assert matcher.accept_string(input) == accepted


input_accepted = ['{"name": "John"}', '{ "name" : "John" }']


@pytest.mark.parametrize("input_accepted", input_accepted)
def test_grammar_accept(input_accepted: str):
    assert _is_grammar_accept_string(json_grammar, input_accepted)


input_refused = ('{ name: "John" }', '{ "name": "John" } ')


@pytest.mark.parametrize("input_refused", input_refused)
def test_grammar_refuse(input_refused: str):
    assert not _is_grammar_accept_string(json_grammar, input_refused)


def test_debug_print_internal_state():
    matcher = _get_matcher_from_grammar(json_grammar)
    input_str = '{"name": "John"}'
    for c in input_str:
        assert matcher.accept_string(c)
        internal_state = matcher._debug_print_internal_state()
        assert len(internal_state) > 0


tokenizer_path__input_str__expected_rejected_sizes = [
    (
        "meta-llama/Llama-2-7b-chat-hf",
        '{"id": 1,"name": "Example"}',
        [
            # fmt: off
            31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270, 270, 270,
            270, 31973, 31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263, 31974, 31999,
            # fmt: on
        ],
    ),
    (
        # test for llama 3
        "meta-llama/Meta-Llama-3-8B-Instruct",
        '{"id": 1,"name": "Example哈哈"}',
        [
            # fmt: off
            128235, 127497, 4744, 4744, 4744, 127849, 126399, 126399, 126760, 127499, 4744, 4744,
            4744, 4744, 4744, 127849, 126399, 126399, 4694, 4694, 4694, 4694, 4694, 4694, 4694,
            4694, 128066, 128111, 4694, 128066, 128111, 4694, 127873, 128255,
            # fmt: on
        ],
    ),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path, input_str, expected_rejected_sizes",
    tokenizer_path__input_str__expected_rejected_sizes,
)
def test_fill_next_token_bitmask(
    tokenizer_path: str, input_str: str, expected_rejected_sizes: Optional[List[int]]
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    input_bytes = input_str.encode("utf-8")
    rejected_sizes = []

    for i, c in enumerate(input_bytes):
        matcher.fill_next_token_bitmask(token_bitmask)
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        rejected_sizes.append(len(rejected_token_ids))
        if expected_rejected_sizes is not None:
            assert rejected_sizes[-1] == expected_rejected_sizes[i], (
                rejected_sizes[-1],
                expected_rejected_sizes[i],
            )
        assert matcher.accept_string(bytes([c]))

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    rejected_sizes.append(len(rejected_token_ids))
    if expected_rejected_sizes is not None:
        assert rejected_sizes[-1] == expected_rejected_sizes[-1]


def test_token_operations():
    """Test accepting token and finding the next token mask."""
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    tokenizer_info = xgr.TokenizerInfo(vocab)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    expected = [
        ["{"],
        ['"', "}", "\n", " ", '"a":true'],
        ["<s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        ["<s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        [":", "\n", " ", ':"'],
        ['"', "{", "6", "\n", " "],
        ["}", ", ", "6", "\n", " "],
        [" ", "\n", '"', '"a":true'],
        [" ", "\n", '"', '"a":true'],
        ["}", ", ", "\n", " "],
        ["</s>"],
    ]

    result = []

    for id in input_ids:
        matcher.fill_next_token_bitmask(token_bitmask)
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
        accepted_tokens = [vocab[i] for i in accepted]
        result.append(accepted_tokens)
        assert id in accepted, vocab[id]
        assert matcher.accept_token(id)

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
    accepted_tokens = [vocab[i] for i in accepted]
    result.append(accepted_tokens)

    assert result == expected


def test_rollback():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    tokenizer_info = xgr.TokenizerInfo(vocab)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(
        json_grammar, tokenizer_info, max_rollback_tokens=5
    )

    assert matcher.max_rollback_tokens == 5

    input_ids_splitted = [input_ids[i : i + 2] for i in range(0, len(input_ids), 2)]

    for i_1, i_2 in input_ids_splitted:
        orig_result = []
        token_bitmask1 = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask1)
        orig_result.append(token_bitmask1)
        assert matcher.accept_token(i_1)
        token_bitmask2 = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask2)
        orig_result.append(token_bitmask2)
        assert matcher.accept_token(i_2)

        matcher.rollback(2)
        result_after_rollback = []
        new_token_bitmask1 = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(new_token_bitmask1)
        result_after_rollback.append(new_token_bitmask1)
        assert matcher.accept_token(i_1)
        new_token_bitmask2 = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(new_token_bitmask2)
        result_after_rollback.append(new_token_bitmask2)
        assert matcher.accept_token(i_2)
        for l, r in zip(orig_result, result_after_rollback):
            torch.testing.assert_close(l, r)


def test_graceful_rollback_failure():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", "6:", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", '"', ":"]
    input_ids = [vocab.index(t) for t in input_splitted]

    tokenizer_info = xgr.TokenizerInfo(vocab)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(
        json_grammar, tokenizer_info, max_rollback_tokens=5
    )

    for i in input_ids:
        assert matcher.accept_token(i)

    assert not matcher.accept_token(vocab.index("6:"))

    # The matching should have accepted char '6' but failed to accept char ':'
    # A graceful revert should then occur, where char '6' is rolled back and
    # the state of the matcher is the same as before the failed call to accept_token

    for i in map(vocab.index, ['"', "abc", '"', " ", "}"]):
        assert matcher.accept_token(i)


def test_reset():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    tokenizer_info = xgr.TokenizerInfo(vocab)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)

    orig_result = []

    for i in input_ids:
        token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask)
        orig_result.append(token_bitmask)
        assert matcher.accept_token(i)

    matcher.reset()

    result_after_reset = []

    for i in input_ids:
        token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask)
        result_after_reset.append(token_bitmask)
        assert matcher.accept_token(i)

    for l, r in zip(orig_result, result_after_reset):
        torch.testing.assert_close(l, r)


def test_termination():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", " }", ", ", "6", ":", "\n", " ", '"a"', ':true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a"', ":true", " }", "</s>"]
    input_ids = [vocab.index(t) for t in input_splitted]
    tokenizer_info = xgr.TokenizerInfo(vocab)

    matcher = _get_matcher_from_grammar_and_tokenizer_info(
        json_grammar, tokenizer_info, max_rollback_tokens=5
    )
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    for i in input_ids:
        matcher.fill_next_token_bitmask(token_bitmask)
        assert matcher.accept_token(i)

    assert matcher.is_terminated()

    assert matcher.accept_token(0) is False

    with pytest.raises(RuntimeError):
        matcher.fill_next_token_bitmask(token_bitmask)

    matcher.rollback(2)

    assert not matcher.is_terminated()
    assert matcher.accept_token(input_ids[-2])


def test_get_jump_forward_string():
    grammar_ebnf = r"""root ::= "abb" | "abbd" | other_rule
other_rule ::= "a" sub_rule "b"
sub_rule ::= "b"
"""
    grammar = xgr.Grammar.from_ebnf(grammar_ebnf)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar)
    assert matcher.accept_string("a")
    assert matcher.find_jump_forward_string() == "bb"


def test_vocab_size():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    tokenizer_info = xgr.TokenizerInfo(vocab, vocab_size=64)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    assert token_bitmask.shape == (1, 2)

    rejected_tokens = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert rejected_tokens == [i for i in range(64) if i != 7]


tokenizer_path_override_stop_tokens = [
    ("meta-llama/Llama-2-7b-chat-hf", [2]),
    ("meta-llama/Meta-Llama-3-8B-Instruct", [128001, 128009]),
    ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", [100001]),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path, override_stop_tokens", tokenizer_path_override_stop_tokens
)
def test_override_stop_tokens(tokenizer_path: str, override_stop_tokens: List[int]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info_1 = xgr.TokenizerInfo.from_huggingface(
        tokenizer, stop_token_ids=override_stop_tokens
    )
    matcher_1 = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info_1)
    assert tokenizer_info_1.stop_token_ids == override_stop_tokens
    assert matcher_1.stop_token_ids == override_stop_tokens

    tokenizer_info_2 = xgr.TokenizerInfo.from_huggingface(tokenizer)
    matcher_2 = _get_matcher_from_grammar_and_tokenizer_info(
        json_grammar, tokenizer_info_2, override_stop_tokens=override_stop_tokens
    )
    assert matcher_2.stop_token_ids == override_stop_tokens


if __name__ == "__main__":
    pytest.main(sys.argv)
