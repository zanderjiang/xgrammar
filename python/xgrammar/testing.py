"""Testing utilities."""

import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from pydantic import BaseModel

from .base import _core
from .compiler import CompiledGrammar, GrammarCompiler
from .grammar import Grammar, _convert_schema_to_str
from .matcher import GrammarMatcher, bitmask_dtype
from .tokenizer_info import TokenizerInfo


def _json_schema_to_ebnf(
    schema: Union[str, Type[BaseModel], Dict[str, Any]],
    *,
    any_whitespace: bool = True,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
) -> str:
    """Convert JSON schema string to BNF grammar string. For test purposes.

    Parameters
    ----------
    schema : Union[str, Type[BaseModel], Dict[str, Any]]
        The schema string or Pydantic model or JSON schema dict.

    indent : Optional[int], default: None
        The number of spaces for indentation. If None, the output will be in one line.

    separators : Optional[Tuple[str, str]], default: None
        Two separators used in the schema: comma and colon. Examples: (",", ":"), (", ", ": ").
        If None, the default separators will be used: (",", ": ") when the indent is not None,
        and (", ", ": ") otherwise.

    strict_mode : bool, default: True
        Whether to use strict mode. In strict mode, the generated grammar will not allow
        properties and items that is not specified in the schema. This is equivalent to
        setting unevaluatedProperties and unevaluatedItems to false.

        This helps LLM to generate accurate output in the grammar-guided generation with JSON
        schema.

    Returns
    -------
    bnf_string : str
        The BNF grammar string.
    """
    schema_str = _convert_schema_to_str(schema)
    return _core.testing._json_schema_to_ebnf(
        schema_str, any_whitespace, indent, separators, strict_mode
    )


def _regex_to_ebnf(regex: str, with_rule_name: bool = True) -> str:
    r"""Convert a regex string to BNF grammar string. For test purposes. The regex grammar
    follows the syntax in JavaScript (ECMA 262). Check
    https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions
    for a tutorial. Currently the following features are not supported:
    1. Backreference (\1)
    2. non-capturing group, naming capture groups and assertions ((?...))
    3. Unicode character class escape (\p{...})
    4. Word boundary (\b)
    5. Unicode property escapes (\p{...})
    6. Quantifier with range {x,y}. Now user can just repeat the element as a workaround.

    This method is primarily intended for testing and debugging purposes.

    Parameters
    ----------
    regex : str
        The regex string to be converted.

    Returns
    -------
    bnf_string : str
        The BNF grammar string converted from the input regex.
    """
    return _core.testing._regex_to_ebnf(regex, with_rule_name)


def _ebnf_to_grammar_no_normalization(ebnf_string: str, root_rule_name: str = "root") -> Grammar:
    """Convert a BNF grammar string to a Grammar object without normalization. For test
    purposes. The result grammar cannot be compiled / used in GrammarMatcher.

    Parameters
    ----------
    ebnf_string : str
        The BNF grammar string to be converted.

    Returns
    -------
    grammar : Grammar
        The unnormalized Grammar object converted from the input BNF grammar string.
    """
    return Grammar._create_from_handle(
        _core.testing._ebnf_to_grammar_no_normalization(ebnf_string, root_rule_name)
    )


def _get_matcher_from_grammar(grammar: Union[Grammar, str], **kwargs) -> GrammarMatcher:
    """Create a GrammarMatcher from a grammar. The tokenizer info will be set to an empty
    TokenizerInfo. The result matcher can only accept strings, and cannot accept tokens.

    Parameters
    ----------
    grammar : Union[Grammar, str]
        The grammar to create the matcher from. Can be either a Grammar object or a string
        containing EBNF grammar.

    Returns
    -------
    matcher : GrammarMatcher
        The created grammar matcher.
    """
    tokenizer_info = TokenizerInfo([])
    grammar_compiler = GrammarCompiler(tokenizer_info, cache_enabled=False)
    compiled_grammar = grammar_compiler.compile_grammar(grammar)
    return GrammarMatcher(compiled_grammar, terminate_without_stop_token=True, **kwargs)


def _is_grammar_accept_string(
    grammar: Union[Grammar, str],
    input_str: str,
    *,
    debug_print: bool = False,
    print_time: bool = False,
) -> bool:
    """Check if a grammar accepts a string. For test purposes.

    Parameters
    ----------
    grammar : Union[Grammar, str]
        The grammar to check. Can be either a Grammar object or a BNF grammar string.
    input_str : str
        The input string to check.
    debug_print : bool, default: False
        Whether to print debug information during matching.
    print_time : bool, default: False
        Whether to print timing information.

    Returns
    -------
    bool
        True if the grammar accepts the string, False otherwise.
    """
    grammar_matcher = _get_matcher_from_grammar(grammar)

    if print_time:
        start = time.monotonic_ns()

    accepted = grammar_matcher.accept_string(input_str, debug_print=debug_print)

    if print_time:
        end = time.monotonic_ns()
        print(f"Accepting {input_str}, result: {accepted}, time: {(end - start) / 1e3} us")

    if not accepted:
        return False
    return grammar_matcher.is_terminated()


def _get_masked_tokens_from_bitmask(
    bitmask: torch.Tensor, vocab_size: int, index: int = 0
) -> List[int]:
    """Get the ids of the rejected tokens from the bitmask. Mainly for debug purposes.

    Parameters
    ----------
    bitmask : torch.Tensor
        The rejected token bitmask. Should be generated by allocate_token_bitmask and
        filled by fill_next_token_bitmask. Should be on CPU.

    index : int, default: 0
        The batch index of the bitmask. For batch inference, bitmask[index] will be used.
        Otherwise is ignored.

    Returns
    -------
    rejected_token_ids : List[int]
        A list of rejected token ids.
    """
    if bitmask.device.type != "cpu":
        raise ValueError("bitmask should be on CPU.")
    if bitmask.dtype != bitmask_dtype:
        raise ValueError(f"bitmask should be of type {bitmask_dtype}.")
    return _core.testing._get_masked_tokens_from_bitmask(
        bitmask.data_ptr(), list(bitmask.shape), vocab_size, index
    )


def _is_single_token_bitmask(
    bitmask: torch.Tensor, vocab_size: int, index: int = 0
) -> Tuple[bool, int]:
    """Check if the bitmask is a single token bitmask.

    Parameters
    ----------
    bitmask : torch.Tensor
        The bitmask to check. Should be on CPU.
    vocab_size : int
        The size of the vocabulary.
    index : int, default: 0
        The index of the bitmask.

    Returns
    -------
    is_single_token : bool
        True if the bitmask is a single token bitmask, False otherwise.
    token_id : int
        The id of the token if the bitmask is a single token bitmask, -1 otherwise.
    """
    return _core.testing._is_single_token_bitmask(
        bitmask.data_ptr(), list(bitmask.shape), vocab_size, index
    )


def _bool_mask_to_bitmask(bool_mask: torch.Tensor) -> torch.Tensor:
    """Get the bitmask from bool mask. If the bool mask does not align with the 32-bit block
    size, it will add extra 1 paddings.

    Parameters
    ----------
    bool_mask : torch.Tensor
        The rejected token bool mask. For each element value, True means the token is allowed,
        while False means the token is rejected.

    Returns
    -------
    bitmask : torch.Tensor
        The rejected token bitmask.
    """
    bool_mask_int32 = bool_mask.to(torch.int32)
    # Pad to multiple of 32
    pad_size = (32 - bool_mask.shape[1] % 32) % 32
    if pad_size > 0:
        bool_mask_int32 = torch.nn.functional.pad(bool_mask_int32, (0, pad_size), value=1)
    bool_mask_view = bool_mask_int32.view(bool_mask.shape[0], -1, 32)
    # To avoid error for overflow, we construct int64 weights and convert to int32
    weights = torch.tensor(
        [1 << i for i in range(32)], device=bool_mask.device, dtype=torch.int64
    ).to(torch.int32)
    bitmask = (bool_mask_view * weights).sum(dim=2)
    return bitmask.to(torch.int32)


def _get_matcher_from_grammar_and_tokenizer_info(
    grammar: Union[Grammar, str], tokenizer_info: Optional[TokenizerInfo] = None, **kwargs
) -> GrammarMatcher:
    """Create a GrammarMatcher from a grammar and tokenizer info.

    Parameters
    ----------
    grammar : Union[Grammar, str]
        The grammar to create the matcher from. Can be either a Grammar object or a string
        containing EBNF grammar.
    tokenizer_info : Optional[TokenizerInfo], default: None
        Information about the tokenizer to use with this grammar. If None, an empty
        TokenizerInfo will be created.
    **kwargs
        Additional keyword arguments to pass to the GrammarMatcher constructor.

    Returns
    -------
    matcher : GrammarMatcher
        The created grammar matcher.
    """
    if tokenizer_info is None:
        tokenizer_info = TokenizerInfo([])
    grammar_compiler = GrammarCompiler(tokenizer_info, cache_enabled=False)
    compiled_grammar = grammar_compiler.compile_grammar(grammar)
    return GrammarMatcher(compiled_grammar, **kwargs)


def _get_allow_empty_rule_ids(compiled_grammar: CompiledGrammar) -> List[int]:
    return _core.testing._get_allow_empty_rule_ids(compiled_grammar._handle)


def _generate_range_regex(start: Optional[int] = None, end: Optional[int] = None) -> str:
    return _core.testing._generate_range_regex(start, end)


def _generate_float_regex(start: Optional[float] = None, end: Optional[float] = None) -> str:
    return _core.testing._generate_float_regex(start, end)


class GrammarFunctor:
    """A utility class for transforming grammars. These methods are called during grammar parsing.
    For test purposes."""

    @staticmethod
    def structure_normalizer(grammar: Grammar) -> Grammar:
        """Normalize the structure of the grammar."""
        return Grammar._create_from_handle(
            _core.testing.grammar_functor.structure_normalizer(grammar._handle)
        )

    @staticmethod
    def rule_inliner(grammar: Grammar) -> Grammar:
        """Inline some rule references in the grammar."""
        return Grammar._create_from_handle(
            _core.testing.grammar_functor.rule_inliner(grammar._handle)
        )

    @staticmethod
    def byte_string_fuser(grammar: Grammar) -> Grammar:
        """Fuse the byte string elements in the grammar."""
        return Grammar._create_from_handle(
            _core.testing.grammar_functor.byte_string_fuser(grammar._handle)
        )

    @staticmethod
    def dead_code_eliminator(grammar: Grammar) -> Grammar:
        """Eliminate the not referenced rules in the grammar."""
        return Grammar._create_from_handle(
            _core.testing.grammar_functor.dead_code_eliminator(grammar._handle)
        )

    @staticmethod
    def lookahead_assertion_analyzer(grammar: Grammar) -> Grammar:
        """Analyze and add lookahead assertions in the grammar."""
        return Grammar._create_from_handle(
            _core.testing.grammar_functor.lookahead_assertion_analyzer(grammar._handle)
        )
