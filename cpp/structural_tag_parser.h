/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/structural_tag_parser.h
 */

#ifndef XGRAMMAR_STRUCTURAL_TAG_PARSER_H_
#define XGRAMMAR_STRUCTURAL_TAG_PARSER_H_

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "picojson.h"

namespace xgrammar {

/**
 * @brief Base class for format specifications
 */
struct StructuralTagFormat {
  std::string type;

  virtual ~StructuralTagFormat() = default;
  virtual picojson::value to_json() const = 0;
};

struct WildcardTextFormat : public StructuralTagFormat {
  WildcardTextFormat() { type = "wildcard_text"; }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    return picojson::value(obj);
  }
};

struct LiteralFormat : public StructuralTagFormat {
  std::string text;

  LiteralFormat(const std::string& t) : text(t) { type = "literal"; }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["text"] = picojson::value(text);
    return picojson::value(obj);
  }
};

struct JSONSchemaFormat : public StructuralTagFormat {
  picojson::value json_schema;

  JSONSchemaFormat(const picojson::value& schema) : json_schema(schema) { type = "json_schema"; }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["json_schema"] = json_schema;
    return picojson::value(obj);
  }
};

// Forward declaration for recursive structure
struct TagFormat;
struct SequenceFormat;
struct TagsAndTextFormat;
struct TagsWithSeparatorFormat;

using StructuralTagFormatPtr = std::shared_ptr<StructuralTagFormat>;

struct TagFormat : public StructuralTagFormat {
  std::string begin;
  StructuralTagFormatPtr content;
  std::string end;

  TagFormat(const std::string& b, StructuralTagFormatPtr c, const std::string& e)
      : begin(b), content(c), end(e) {
    type = "tag";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["begin"] = picojson::value(begin);
    obj["content"] = content->to_json();
    obj["end"] = picojson::value(end);
    return picojson::value(obj);
  }
};

struct SequenceFormat : public StructuralTagFormat {
  std::vector<StructuralTagFormatPtr> elements;

  SequenceFormat(const std::vector<StructuralTagFormatPtr>& elems) : elements(elems) {
    type = "sequence";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);

    picojson::array elems_array;
    for (const auto& elem : elements) {
      elems_array.push_back(elem->to_json());
    }
    obj["elements"] = picojson::value(elems_array);

    return picojson::value(obj);
  }
};

struct TagsAndTextFormat : public StructuralTagFormat {
  std::vector<std::shared_ptr<TagFormat>> tags;
  std::vector<std::string> triggers;
  bool at_least_one;
  bool stop_after_first;

  TagsAndTextFormat(
      const std::vector<std::shared_ptr<TagFormat>>& t,
      const std::vector<std::string>& trg,
      bool at_least = false,
      bool stop_after = false
  )
      : tags(t), triggers(trg), at_least_one(at_least), stop_after_first(stop_after) {
    type = "tags_and_text";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);

    picojson::array tags_array;
    for (const auto& tag : tags) {
      tags_array.push_back(tag->to_json());
    }
    obj["tags"] = picojson::value(tags_array);

    picojson::array triggers_array;
    for (const auto& trigger : triggers) {
      triggers_array.push_back(picojson::value(trigger));
    }
    obj["triggers"] = picojson::value(triggers_array);

    obj["at_least_one"] = picojson::value(at_least_one);
    obj["stop_after_first"] = picojson::value(stop_after_first);

    return picojson::value(obj);
  }
};

struct TagsWithSeparatorFormat : public StructuralTagFormat {
  std::vector<std::shared_ptr<TagFormat>> tags;
  std::string separator;
  bool at_least_one;
  bool stop_after_first;

  TagsWithSeparatorFormat(
      const std::vector<std::shared_ptr<TagFormat>>& t,
      const std::string& sep,
      bool at_least = false,
      bool stop_after = false
  )
      : tags(t), separator(sep), at_least_one(at_least), stop_after_first(stop_after) {
    type = "tags_with_separator";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);

    picojson::array tags_array;
    for (const auto& tag : tags) {
      tags_array.push_back(tag->to_json());
    }
    obj["tags"] = picojson::value(tags_array);

    obj["separator"] = picojson::value(separator);
    obj["at_least_one"] = picojson::value(at_least_one);
    obj["stop_after_first"] = picojson::value(stop_after_first);

    return picojson::value(obj);
  }
};

struct StructuralTag {
  std::string type;
  StructuralTagFormatPtr format;

  StructuralTag(StructuralTagFormatPtr fmt) : format(fmt) { type = "structural_tag"; }

  picojson::value to_json() const {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["format"] = format->to_json();
    return picojson::value(obj);
  }
};

// Result structures for parsed content

struct StructuralTagResult {
  std::string type;

  virtual ~StructuralTagResult() = default;
  virtual picojson::value to_json() const = 0;
};

struct LiteralResult : public StructuralTagResult {
  std::string text;

  LiteralResult(const std::string& t) : text(t) { type = "literal"; }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["text"] = picojson::value(text);
    return picojson::value(obj);
  }
};

struct JSONSchemaResult : public StructuralTagResult {
  picojson::value value;

  JSONSchemaResult(const picojson::value& v) : value(v) { type = "json_schema"; }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["value"] = value;
    return picojson::value(obj);
  }
};

struct WildcardTextResult : public StructuralTagResult {
  std::string text;

  WildcardTextResult(const std::string& t) : text(t) { type = "wildcard_text"; }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["text"] = picojson::value(text);
    return picojson::value(obj);
  }
};

// Forward declarations for result structures
using StructuralTagResultPtr = std::shared_ptr<StructuralTagResult>;

struct TagResult : public StructuralTagResult {
  std::string begin;
  StructuralTagResultPtr content;
  std::string end;

  TagResult(const std::string& b, StructuralTagResultPtr c, const std::string& e)
      : begin(b), content(c), end(e) {
    type = "tag";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);
    obj["begin"] = picojson::value(begin);
    obj["content"] = content->to_json();
    obj["end"] = picojson::value(end);
    return picojson::value(obj);
  }
};

struct SequenceResult : public StructuralTagResult {
  std::vector<StructuralTagResultPtr> elements;

  SequenceResult(const std::vector<StructuralTagResultPtr>& elems) : elements(elems) {
    type = "sequence";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);

    picojson::array elems_array;
    for (const auto& elem : elements) {
      elems_array.push_back(elem->to_json());
    }
    obj["elements"] = picojson::value(elems_array);

    return picojson::value(obj);
  }
};

struct TagsAndTextResult : public StructuralTagResult {
  using TagsAndTextItem = std::variant<std::string, std::shared_ptr<TagResult>>;
  std::vector<TagsAndTextItem> tags_and_text;

  TagsAndTextResult(const std::vector<TagsAndTextItem>& items) : tags_and_text(items) {
    type = "tags_and_text";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);

    picojson::array items_array;
    for (const auto& item : tags_and_text) {
      if (std::holds_alternative<std::string>(item)) {
        items_array.push_back(picojson::value(std::get<std::string>(item)));
      } else {
        items_array.push_back(std::get<std::shared_ptr<TagResult>>(item)->to_json());
      }
    }
    obj["tags_and_text"] = picojson::value(items_array);

    return picojson::value(obj);
  }
};

struct TagsWithSeparatorResult : public StructuralTagResult {
  std::vector<std::shared_ptr<TagResult>> tags;
  std::string separator;

  TagsWithSeparatorResult(const std::vector<std::shared_ptr<TagResult>>& t, const std::string& sep)
      : tags(t), separator(sep) {
    type = "tags_with_separator";
  }

  picojson::value to_json() const override {
    picojson::object obj;
    obj["type"] = picojson::value(type);

    picojson::array tags_array;
    for (const auto& tag : tags) {
      tags_array.push_back(tag->to_json());
    }
    obj["tags"] = picojson::value(tags_array);

    obj["separator"] = picojson::value(separator);

    return picojson::value(obj);
  }
};

// Parser functions
class ParsingError : public std::runtime_error {
 public:
  explicit ParsingError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Parse structural tags from input string according to format specification
 *
 * @param input The LLM output string to parse
 * @param structural_tag The format specification to use for parsing
 * @return StructuralTagResultPtr The parsed result
 * @throws ParsingError if parsing fails
 */
StructuralTagResultPtr parse_structural_tag(
    const std::string& input, const StructuralTag& structural_tag
);

/**
 * @brief Create a StructuralTagFormat from a JSON representation
 *
 * @param json_value The JSON representation of the format specification
 * @return StructuralTagFormatPtr The created format specification
 * @throws ParsingError if the JSON is invalid
 */
StructuralTagFormatPtr create_format_from_json(const picojson::value& json_value);

/**
 * @brief Parse JSON string into StructuralTag format specification
 *
 * @param json_str The JSON string representing the format specification
 * @return StructuralTag The parsed specification
 * @throws ParsingError if the JSON is invalid
 */
StructuralTag parse_structural_tag_spec(const std::string& json_str);

/**
 * @brief Parse input string using a JSON-specified format
 *
 * @param input The LLM output string to parse
 * @param format_json_str The JSON string representing the format specification
 * @return std::string JSON string representation of the parsing result
 */
std::string parse_with_json(const std::string& input, const std::string& format_json_str);

}  // namespace xgrammar

#endif  // XGRAMMAR_STRUCTURAL_TAG_PARSER_H_
