import json
from typing import Any, Dict, List, Literal, Optional, Union, cast

from pydantic import BaseModel, Field

from .base import _core


# Input models for structural tag specification
class StructuralTagFormat(BaseModel):
    type: str


class WildcardTextFormat(StructuralTagFormat):
    type: Literal["wildcard_text"]


class LiteralFormat(StructuralTagFormat):
    type: Literal["literal"]
    text: str


class JSONSchemaFormat(StructuralTagFormat):
    type: Literal["json_schema"]
    json_schema: Dict[str, Any]


class TagFormat(StructuralTagFormat):
    type: Literal["tag"]
    begin: str
    content: StructuralTagFormat
    end: str


class SequenceFormat(StructuralTagFormat):
    type: Literal["sequence"]
    elements: List[StructuralTagFormat]


class TagsAndTextFormat(StructuralTagFormat):
    type: Literal["tags_and_text"]
    tags: List[TagFormat]
    triggers: List[str]
    at_least_one: Optional[bool] = False
    stop_after_first: Optional[bool] = False


class TagsWithSeparatorFormat(StructuralTagFormat):
    type: Literal["tags_with_separator"]
    tags: List[TagFormat]
    separator: str
    at_least_one: Optional[bool] = False
    stop_after_first: Optional[bool] = False


class StructuralTag(BaseModel):
    type: Literal["structural_tag"]
    format: StructuralTagFormat


# Output models for structural tag parsing results
class StructuralTagResult(BaseModel):
    type: str


class LiteralResult(StructuralTagResult):
    type: Literal["literal"]
    text: str


class JSONSchemaResult(StructuralTagResult):
    type: Literal["json_schema"]
    value: Union[int, float, bool, str, List, Dict[str, Any]]


class WildcardTextResult(StructuralTagResult):
    type: Literal["wildcard_text"]
    text: str


class SequenceResult(StructuralTagResult):
    type: Literal["sequence"]
    elements: List[StructuralTagResult]


class TagResult(StructuralTagResult):
    type: Literal["tag"]
    begin: str
    content: StructuralTagResult
    end: str


class TagsAndTextResult(StructuralTagResult):
    type: Literal["tags_and_text"]
    tags_and_text: List[Union[str, TagResult]]


class TagsWithSeparatorResult(StructuralTagResult):
    type: Literal["tags_with_separator"]
    tags: List[TagResult]
    separator: str


# Helper function to convert a JSON result back to the appropriate Pydantic model
def _json_to_result(result_json: Dict[str, Any]) -> StructuralTagResult:
    result_type = result_json.get("type")

    if result_type == "literal":
        return LiteralResult(**result_json)
    elif result_type == "json_schema":
        return JSONSchemaResult(**result_json)
    elif result_type == "wildcard_text":
        return WildcardTextResult(**result_json)
    elif result_type == "tag":
        # For tag results, we need to recursively parse the content
        content_data = result_json.get("content", {})
        result_json["content"] = _json_to_result(content_data)
        return TagResult(**result_json)
    elif result_type == "sequence":
        # For sequence results, we need to recursively parse each element
        elements = []
        for element in result_json.get("elements", []):
            elements.append(_json_to_result(element))
        result_json["elements"] = elements
        return SequenceResult(**result_json)
    elif result_type == "tags_and_text":
        # For tags_and_text results, we need to handle both strings and tag results
        tags_and_text = []
        for item in result_json.get("tags_and_text", []):
            if isinstance(item, str):
                tags_and_text.append(item)
            else:
                tags_and_text.append(_json_to_result(item))
        result_json["tags_and_text"] = tags_and_text
        return TagsAndTextResult(**result_json)
    elif result_type == "tags_with_separator":
        # For tags_with_separator results, we need to recursively parse each tag
        tags = []
        for tag in result_json.get("tags", []):
            tags.append(_json_to_result(tag))
        result_json["tags"] = tags
        return TagsWithSeparatorResult(**result_json)
    else:
        raise ValueError(f"Unknown result type: {result_type}")


def structural_tag_parser(response: str, structural_tag: StructuralTag) -> StructuralTagResult:
    """
    Parse a response string according to the structural tag format.

    Args:
        response: The string response from the LLM
        structural_tag: The structural tag specification

    Returns:
        A structured object representing the parsed response
    """
    try:
        # Convert the Pydantic model to a JSON string
        structural_tag_json = structural_tag.model_dump_json()

        # Call the C++ backend through nanobind
        result_json = _core._testing._parse_structural_tag(response, structural_tag_json)

        # Parse the JSON result
        result_dict = json.loads(result_json)

        # Check for errors
        if "error" in result_dict:
            raise ValueError(f"Parsing error: {result_dict['error']}")

        # Convert the JSON result back to a Pydantic model
        return _json_to_result(result_dict)
    except Exception as e:
        if hasattr(e, "__traceback__"):
            import traceback

            traceback.print_exception(type(e), e, e.__traceback__)
        raise ParsingError(f"Failed to parse response with structural tag: {e}")


class ParsingError(Exception):
    """Exception raised when parsing fails."""

    pass
