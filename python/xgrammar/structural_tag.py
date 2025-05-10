from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, root_validator


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


# Sample parser function definition - implementation to be developed
def structural_tag_parser(response: str, structural_tag: StructuralTag) -> StructuralTagResult:
    """
    Parse a response string according to the structural tag format.

    Args:
        response: The string response from the LLM
        structural_tag: The structural tag specification

    Returns:
        A structured object representing the parsed response
    """
    pass
