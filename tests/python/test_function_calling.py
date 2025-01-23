import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _parse_message


def test_tag_based_simple():
    """Test simple tag-based format with single parameter."""
    input_str = '<function=get_weather>{"location": "SF"}</function>'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("get_weather", {"location": "SF"})


def test_tag_based_multiple_params():
    """Test tag-based format with multiple parameters."""
    input_str = '<function=search>{"query": "python", "limit": "10", "filter": "true"}</function>'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("search", {"query": "python", "limit": "10", "filter": "true"})


def test_json_format_simple():
    """Test simple JSON format."""
    input_str = '{"name": "get_time", "parameters": {"zone": "PST"}}'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("get_time", {"zone": "PST"})


def test_json_format_multiple_params():
    """Test JSON format with multiple parameters."""
    input_str = """{"name": "create_user",
                   "parameters": {
                       "username": "john_doe",
                       "age": "25",
                       "active": "true",
                       "email": "john@example.com"
                   }}"""
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == (
        "create_user",
        {"username": "john_doe", "age": "25", "active": "true", "email": "john@example.com"},
    )


def test_multiple_functions():
    """Test parsing multiple function calls in the same input."""
    input_str = """
    <function=get_weather>{"city": "SF"}</function>
    <function=get_time>{"zone": "PST"}</function>
    """
    result = _parse_message(input_str)
    assert len(result) == 2
    assert result[0] == ("get_weather", {"city": "SF"})
    assert result[1] == ("get_time", {"zone": "PST"})


def test_mixed_formats():
    """Test mixing tag-based and JSON formats."""
    input_str = """
    <function=get_weather>{"city": "SF"}</function>
    {"name": "get_time", "parameters": {"zone": "PST"}}
    """
    result = _parse_message(input_str)
    assert len(result) == 2
    assert result[0] == ("get_weather", {"city": "SF"})
    assert result[1] == ("get_time", {"zone": "PST"})


def test_empty_parameters():
    """Test function calls with empty parameters."""
    input_str = "<function=ping>{}</function>"
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("ping", {})


def test_special_values():
    """Test handling of special JSON values."""
    input_str = """
    <function=test_values>{
        "null_value": null,
        "bool_true": true,
        "bool_false": false,
        "number": 42,
        "float": 3.14
    }</function>
    """
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == (
        "test_values",
        {
            "null_value": "null",
            "bool_true": "true",
            "bool_false": "false",
            "number": "42",
            "float": "3.14",
        },
    )


def test_invalid_json():
    """Test handling of invalid JSON in parameters."""
    input_str = "<function=test>{invalid json}</function>"
    with pytest.raises(Exception):
        _parse_message(input_str)


def test_invalid_json_ignore_error():
    """Test ignore_error parameter with invalid JSON."""
    input_str = """
    <function=valid>{"param": "value"}</function>
    <function=invalid>{invalid json}</function>
    """
    result = _parse_message(input_str, ignore_error=True)
    assert len(result) == 1
    assert result[0] == ("valid", {"param": "value"})


def test_malformed_tag():
    """Test handling of malformed tags."""
    input_str = '<function=test{"param": "value"}'  # Missing closing tag
    with pytest.raises(Exception):
        _parse_message(input_str)


def test_empty_input():
    """Test handling of empty input."""
    result = _parse_message("")
    assert len(result) == 0
    assert result == []


def test_whitespace_handling():
    """Test handling of various whitespace in input."""
    input_str = """
        <function=test>
            {
                "param": "value"
            }
        </function>
    """
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("test", {"param": "value"})


def test_unicode_parameters():
    """Test handling of Unicode characters in parameters."""
    input_str = '<function=translate>{"text": "こんにちは世界", "target": "español"}</function>'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("translate", {"text": "こんにちは世界", "target": "español"})


def test_escaped_characters():
    """Test handling of escaped characters in JSON."""
    input_str = '<function=process>{"path": "C:\\\\Program Files\\\\App", "query": ""quoted string""}</function>'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("process", {"path": "C:\\Program Files\\App", "query": '"quoted string"'})


def test_empty_array_parameters():
    """Test handling of empty arrays in parameters."""
    input_str = '<function=update_list>{"ids": [], "tags": [""], "flags": null}</function>'
    result = _parse_message(input_str)
    assert len(result) == 1
    assert result[0] == ("update_list", {"ids": [], "tags": [""], "flags": "null"})


def test_large_number_of_functions():
    """Test handling of a large number of sequential function calls."""
    functions = [f'<function=func_{i}>{{"id": "{i}"}}</function>' for i in range(100)]
    input_str = "\n".join(functions)
    result = _parse_message(input_str)
    assert len(result) == 100
    for i, (name, params) in enumerate(result):
        assert name == f"func_{i}"
        assert params == {"id": str(i)}


def test_mixed_line_endings():
    """Test handling of different line endings (CRLF, LF)."""
    input_str = '<function=test1>{"a": "1"}</function>\r\n<function=test2>{"b": "2"}</function>\n<function=test3>{"c": "3"}</function>'
    result = _parse_message(input_str)
    assert len(result) == 3
    assert result[0] == ("test1", {"a": "1"})
    assert result[1] == ("test2", {"b": "2"})
    assert result[2] == ("test3", {"c": "3"})


def test_json_with_comments():
    """Test handling of JSON-like content with comments (should fail)."""
    input_str = """
    <function=test>{
        // This is a comment
        "param": "value" /* inline comment */
    }</function>
    """
    with pytest.raises(Exception):
        _parse_message(input_str)


def test_case_sensitivity():
    """Test case sensitivity in function names and parameters."""
    input_str = """
    <function=TestFunction>{"ParamOne": "Value", "paramTwo": "value"}</function>
    {"name": "TestFunction2", "parameters": {"PARAM": "VALUE"}}
    """
    result = _parse_message(input_str)
    assert len(result) == 2
    assert result[0] == ("TestFunction", {"ParamOne": "Value", "paramTwo": "value"})
    assert result[1] == ("TestFunction2", {"PARAM": "VALUE"})


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ('<function=test>{"a": "1"}</function>', [("test", {"a": "1"})]),
        ('{"name": "test", "parameters": {"a": "1"}}', [("test", {"a": "1"})]),
    ],
)
def test_parametrized_formats(input_str, expected):
    """Test both formats with parameterization."""
    result = _parse_message(input_str)
    assert result == expected


if __name__ == "__main__":
    pytest.main(sys.argv)
