/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/structural_tag_parser.cc
 */

#include "structural_tag_parser.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace xgrammar {

/**
 * @brief Helper function to get a string from a JSON object
 */
std::string get_string(const picojson::object& obj, const std::string& key) {
  auto it = obj.find(key);
  if (it == obj.end()) {
    throw ParsingError("Missing key '" + key + "' in JSON object");
  }
  if (!it->second.is<std::string>()) {
    throw ParsingError("Key '" + key + "' is not a string");
  }
  return it->second.get<std::string>();
}

/**
 * @brief Helper function to get a boolean from a JSON object
 */
bool get_bool(const picojson::object& obj, const std::string& key, bool default_value = false) {
  auto it = obj.find(key);
  if (it == obj.end()) {
    return default_value;
  }
  if (!it->second.is<bool>()) {
    throw ParsingError("Key '" + key + "' is not a boolean");
  }
  return it->second.get<bool>();
}

/**
 * @brief Helper function to get an array from a JSON object
 */
picojson::array get_array(const picojson::object& obj, const std::string& key) {
  auto it = obj.find(key);
  if (it == obj.end()) {
    throw ParsingError("Missing key '" + key + "' in JSON object");
  }
  if (!it->second.is<picojson::array>()) {
    throw ParsingError("Key '" + key + "' is not an array");
  }
  return it->second.get<picojson::array>();
}

/**
 * @brief Helper function to get an object from a JSON object
 */
picojson::object get_object(const picojson::object& obj, const std::string& key) {
  auto it = obj.find(key);
  if (it == obj.end()) {
    throw ParsingError("Missing key '" + key + "' in JSON object");
  }
  if (!it->second.is<picojson::object>()) {
    throw ParsingError("Key '" + key + "' is not an object");
  }
  return it->second.get<picojson::object>();
}

/**
 * @brief Helper function to get a value from a JSON object
 */
picojson::value get_value(const picojson::object& obj, const std::string& key) {
  auto it = obj.find(key);
  if (it == obj.end()) {
    throw ParsingError("Missing key '" + key + "' in JSON object");
  }
  return it->second;
}

// Forward declarations of internal parsing functions
StructuralTagResultPtr parse_format(
    const std::string& text, const StructuralTagFormatPtr& format_spec
);
StructuralTagResultPtr parse_literal(
    const std::string& text, const std::shared_ptr<LiteralFormat>& format_spec
);
StructuralTagResultPtr parse_json_schema(
    const std::string& text, const std::shared_ptr<JSONSchemaFormat>& format_spec
);
StructuralTagResultPtr parse_wildcard_text(
    const std::string& text, const std::shared_ptr<WildcardTextFormat>& format_spec
);
StructuralTagResultPtr parse_sequence(
    const std::string& text, const std::shared_ptr<SequenceFormat>& format_spec
);
StructuralTagResultPtr parse_tag(
    const std::string& text, const std::shared_ptr<TagFormat>& format_spec
);
StructuralTagResultPtr parse_tags_and_text(
    const std::string& text, const std::shared_ptr<TagsAndTextFormat>& format_spec
);
StructuralTagResultPtr parse_tags_with_separator(
    const std::string& text, const std::shared_ptr<TagsWithSeparatorFormat>& format_spec
);
std::pair<StructuralTagResultPtr, std::string> parse_element_in_sequence(
    const std::string& text, const StructuralTagFormatPtr& format_spec
);

// Helper functions for finding and extracting tags
std::vector<std::tuple<int, std::string>> find_trigger_positions(
    const std::string& text, const std::vector<std::string>& triggers
);

/**
 * @brief Main parsing function that dispatches to specialized parsers
 */
StructuralTagResultPtr parse_structural_tag(
    const std::string& input, const StructuralTag& structural_tag
) {
  try {
    return parse_format(input, structural_tag.format);
  } catch (const ParsingError& e) {
    throw ParsingError("Failed to parse response with structural tag: " + std::string(e.what()));
  } catch (const std::exception& e) {
    throw ParsingError("Unexpected error during parsing: " + std::string(e.what()));
  }
}

/**
 * @brief Parse input according to the given format specification
 */
StructuralTagResultPtr parse_format(
    const std::string& text, const StructuralTagFormatPtr& format_spec
) {
  if (auto literal_format = std::dynamic_pointer_cast<LiteralFormat>(format_spec)) {
    return parse_literal(text, literal_format);
  } else if (auto json_schema_format = std::dynamic_pointer_cast<JSONSchemaFormat>(format_spec)) {
    return parse_json_schema(text, json_schema_format);
  } else if (auto tag_format = std::dynamic_pointer_cast<TagFormat>(format_spec)) {
    return parse_tag(text, tag_format);
  } else if (auto sequence_format = std::dynamic_pointer_cast<SequenceFormat>(format_spec)) {
    return parse_sequence(text, sequence_format);
  } else if (auto tags_and_text_format =
                 std::dynamic_pointer_cast<TagsAndTextFormat>(format_spec)) {
    return parse_tags_and_text(text, tags_and_text_format);
  } else if (auto tags_with_separator_format =
                 std::dynamic_pointer_cast<TagsWithSeparatorFormat>(format_spec)) {
    return parse_tags_with_separator(text, tags_with_separator_format);
  } else if (auto wildcard_text_format =
                 std::dynamic_pointer_cast<WildcardTextFormat>(format_spec)) {
    return parse_wildcard_text(text, wildcard_text_format);
  } else {
    throw ParsingError("Unknown format type: " + format_spec->type);
  }
}

/**
 * @brief Parse literal format - the text must match exactly
 */
StructuralTagResultPtr parse_literal(
    const std::string& text, const std::shared_ptr<LiteralFormat>& format_spec
) {
  const std::string& expected_text = format_spec->text;

  if (text != expected_text) {
    throw ParsingError("Expected literal '" + expected_text + "', got '" + text + "'");
  }

  return std::make_shared<LiteralResult>(text);
}

/**
 * @brief Parse JSON schema format - the text must be valid JSON
 */
StructuralTagResultPtr parse_json_schema(
    const std::string& text, const std::shared_ptr<JSONSchemaFormat>& format_spec
) {
  picojson::value json_data;
  std::string err = picojson::parse(json_data, text);

  if (!err.empty()) {
    throw ParsingError("Invalid JSON: " + err);
  }

  // TODO: Add JSON schema validation using format_spec->json_schema
  // For now, we just check that the JSON is valid

  return std::make_shared<JSONSchemaResult>(json_data);
}

/**
 * @brief Parse wildcard text format - any text is allowed
 */
StructuralTagResultPtr parse_wildcard_text(
    const std::string& text, const std::shared_ptr<WildcardTextFormat>& format_spec
) {
  return std::make_shared<WildcardTextResult>(text);
}

/**
 * @brief Parse sequence format - each element must match in order
 */
StructuralTagResultPtr parse_sequence(
    const std::string& text, const std::shared_ptr<SequenceFormat>& format_spec
) {
  std::vector<StructuralTagResultPtr> elements;
  std::string remaining_text = text;

  for (const auto& element_spec : format_spec->elements) {
    if (remaining_text.empty()) {
      throw ParsingError("Unexpected end of text in sequence");
    }

    // Handle different element types
    if (auto literal_spec = std::dynamic_pointer_cast<LiteralFormat>(element_spec)) {
      // For literals, we can just check if the remaining text starts with the literal
      if (!remaining_text.starts_with(literal_spec->text)) {
        throw ParsingError(
            "Expected literal '" + literal_spec->text + "' at: '" +
            remaining_text.substr(0, std::min(remaining_text.size(), size_t(50))) + "...'"
        );
      }

      auto element_result = std::make_shared<LiteralResult>(literal_spec->text);
      elements.push_back(element_result);
      remaining_text = remaining_text.substr(literal_spec->text.length());
    } else {
      // For other types, we need to find the end of the element
      auto [element_result, new_remaining_text] =
          parse_element_in_sequence(remaining_text, element_spec);
      elements.push_back(element_result);
      remaining_text = new_remaining_text;
    }
  }

  // We've parsed all the elements in the sequence, and it's fine if there's text left over
  return std::make_shared<SequenceResult>(elements);
}

/**
 * @brief Parse a single element in a sequence and return the result and remaining text
 */
std::pair<StructuralTagResultPtr, std::string> parse_element_in_sequence(
    const std::string& text, const StructuralTagFormatPtr& format_spec
) {
  // The strategy depends on the element type
  if (auto tag_spec = std::dynamic_pointer_cast<TagFormat>(format_spec)) {
    // For tags, we can find the beginning and end tags
    if (!text.starts_with(tag_spec->begin)) {
      throw ParsingError(
          "Expected tag beginning '" + tag_spec->begin + "' at: '" +
          text.substr(0, std::min(text.size(), size_t(50))) + "...'"
      );
    }

    // Find the end tag
    size_t content_start = tag_spec->begin.length();
    size_t end_pos = text.find(tag_spec->end, content_start);

    if (end_pos == std::string::npos) {
      throw ParsingError(
          "Could not find end tag '" + tag_spec->end + "' in: '" +
          text.substr(0, std::min(text.size(), size_t(100))) + "...'"
      );
    }

    std::string content_text = text.substr(content_start, end_pos - content_start);
    auto content_result = parse_format(content_text, tag_spec->content);

    auto tag_result = std::make_shared<TagResult>(tag_spec->begin, content_result, tag_spec->end);

    std::string remaining_text = text.substr(end_pos + tag_spec->end.length());
    return {tag_result, remaining_text};
  }

  else if (auto tags_and_text_spec = std::dynamic_pointer_cast<TagsAndTextFormat>(format_spec)) {
    // This is more complex - we need to parse tags and text
    auto result = parse_tags_and_text(text, tags_and_text_spec);

    // For now, assume all text was consumed in a sequence context
    // In a real implementation, this would need to be more sophisticated
    return {result, ""};
  }

  else if (auto tags_with_separator_spec =
               std::dynamic_pointer_cast<TagsWithSeparatorFormat>(format_spec)) {
    // Similar complexity to TagsAndTextFormat
    auto result = parse_tags_with_separator(text, tags_with_separator_spec);

    // For now, assume all text was consumed in a sequence context
    return {result, ""};
  }

  else {
    // For other types, we just parse the whole text
    auto result = parse_format(text, format_spec);
    return {result, ""};
  }
}

/**
 * @brief Parse a tag format - text must start with begin tag and end with end tag
 */
StructuralTagResultPtr parse_tag(
    const std::string& text, const std::shared_ptr<TagFormat>& format_spec
) {
  if (!text.starts_with(format_spec->begin)) {
    throw ParsingError(
        "Expected tag beginning '" + format_spec->begin + "' at: '" +
        text.substr(0, std::min(text.size(), size_t(50))) + "...'"
    );
  }

  if (!text.ends_with(format_spec->end)) {
    throw ParsingError(
        "Expected tag ending '" + format_spec->end + "' at the end of: '..." +
        text.substr(text.length() - std::min(text.length(), size_t(50))) + "'"
    );
  }

  std::string content_text = text.substr(
      format_spec->begin.length(),
      text.length() - format_spec->begin.length() - format_spec->end.length()
  );

  auto content_result = parse_format(content_text, format_spec->content);

  return std::make_shared<TagResult>(format_spec->begin, content_result, format_spec->end);
}

/**
 * @brief Find all positions of triggers in the text
 */
std::vector<std::tuple<int, std::string>> find_trigger_positions(
    const std::string& text, const std::vector<std::string>& triggers
) {
  std::vector<std::tuple<int, std::string>> positions;

  for (const auto& trigger : triggers) {
    size_t start = 0;
    while (true) {
      size_t pos = text.find(trigger, start);
      if (pos == std::string::npos) {
        break;
      }
      positions.emplace_back(pos, trigger);
      start = pos + 1;
    }
  }

  // Sort by position
  std::sort(positions.begin(), positions.end(), [](const auto& a, const auto& b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  return positions;
}

/**
 * @brief Parse a tags_with_separator format - tags separated by a specific separator
 */
StructuralTagResultPtr parse_tags_with_separator(
    const std::string& text, const std::shared_ptr<TagsWithSeparatorFormat>& format_spec
) {
  // Split the text by the separator
  std::vector<std::string> parts;
  size_t start = 0;
  size_t end = 0;

  while ((end = text.find(format_spec->separator, start)) != std::string::npos) {
    parts.push_back(text.substr(start, end - start));
    start = end + format_spec->separator.length();
  }

  // Add the last part
  if (start < text.length()) {
    parts.push_back(text.substr(start));
  }

  // If at_least_one is True, we need at least one part
  if (format_spec->at_least_one && parts.empty()) {
    throw ParsingError("No tags found but at_least_one is True");
  }

  std::vector<std::shared_ptr<TagResult>> tag_results;

  for (const auto& part : parts) {
    if (part.empty()) {
      continue;  // Skip empty parts
    }

    // Try to match this part with each tag
    bool matched = false;
    for (const auto& tag_spec : format_spec->tags) {
      if (part.starts_with(tag_spec->begin) && part.ends_with(tag_spec->end)) {
        std::string content_text = part.substr(
            tag_spec->begin.length(),
            part.length() - tag_spec->begin.length() - tag_spec->end.length()
        );

        try {
          auto content_result = parse_format(content_text, tag_spec->content);

          auto tag_result =
              std::make_shared<TagResult>(tag_spec->begin, content_result, tag_spec->end);

          tag_results.push_back(tag_result);
          matched = true;
          break;
        } catch (const std::exception&) {
          // If parsing fails, try the next tag
          continue;
        }
      }
    }

    if (!matched) {
      throw ParsingError("No matching tag found for part: '" + part + "'");
    }

    // If stop_after_first is True, we're done
    if (format_spec->stop_after_first && matched) {
      break;
    }
  }

  return std::make_shared<TagsWithSeparatorResult>(tag_results, format_spec->separator);
}

/**
 * @brief Create a StructuralTagFormat from a JSON representation
 */
StructuralTagFormatPtr create_format_from_json(const picojson::value& json_value) {
  if (!json_value.is<picojson::object>()) {
    throw ParsingError("Format specification must be an object");
  }

  const auto& obj = json_value.get<picojson::object>();

  // Get the type
  std::string format_type = get_string(obj, "type");

  if (format_type == "literal") {
    std::string text = get_string(obj, "text");
    return std::make_shared<LiteralFormat>(text);
  } else if (format_type == "json_schema") {
    picojson::value json_schema = get_value(obj, "json_schema");
    return std::make_shared<JSONSchemaFormat>(json_schema);
  } else if (format_type == "wildcard_text") {
    return std::make_shared<WildcardTextFormat>();
  } else if (format_type == "tag") {
    std::string begin = get_string(obj, "begin");
    picojson::value content_json = get_value(obj, "content");
    std::string end = get_string(obj, "end");

    auto content = create_format_from_json(content_json);
    return std::make_shared<TagFormat>(begin, content, end);
  } else if (format_type == "sequence") {
    picojson::array elements_json = get_array(obj, "elements");

    std::vector<StructuralTagFormatPtr> elements;
    for (const auto& element_json : elements_json) {
      elements.push_back(create_format_from_json(element_json));
    }

    return std::make_shared<SequenceFormat>(elements);
  } else if (format_type == "tags_and_text") {
    picojson::array tags_json = get_array(obj, "tags");
    picojson::array triggers_json = get_array(obj, "triggers");

    std::vector<std::shared_ptr<TagFormat>> tags;
    for (const auto& tag_json : tags_json) {
      auto tag = std::dynamic_pointer_cast<TagFormat>(create_format_from_json(tag_json));
      if (!tag) {
        throw ParsingError("Each element in 'tags' must be a tag format");
      }
      tags.push_back(tag);
    }

    std::vector<std::string> triggers;
    for (const auto& trigger_json : triggers_json) {
      if (!trigger_json.is<std::string>()) {
        throw ParsingError("Each element in 'triggers' must be a string");
      }
      triggers.push_back(trigger_json.get<std::string>());
    }

    bool at_least_one = get_bool(obj, "at_least_one", false);
    bool stop_after_first = get_bool(obj, "stop_after_first", false);

    return std::make_shared<TagsAndTextFormat>(tags, triggers, at_least_one, stop_after_first);
  } else if (format_type == "tags_with_separator") {
    picojson::array tags_json = get_array(obj, "tags");
    std::string separator = get_string(obj, "separator");

    std::vector<std::shared_ptr<TagFormat>> tags;
    for (const auto& tag_json : tags_json) {
      auto tag = std::dynamic_pointer_cast<TagFormat>(create_format_from_json(tag_json));
      if (!tag) {
        throw ParsingError("Each element in 'tags' must be a tag format");
      }
      tags.push_back(tag);
    }

    bool at_least_one = get_bool(obj, "at_least_one", false);
    bool stop_after_first = get_bool(obj, "stop_after_first", false);

    return std::make_shared<TagsWithSeparatorFormat>(
        tags, separator, at_least_one, stop_after_first
    );
  } else {
    throw ParsingError("Unknown format type: " + format_type);
  }
}

/**
 * @brief Parse JSON string into StructuralTag format specification
 */
StructuralTag parse_structural_tag_spec(const std::string& json_str) {
  picojson::value json_value;
  std::string err = picojson::parse(json_value, json_str);

  if (!err.empty()) {
    throw ParsingError("Invalid JSON: " + err);
  }

  if (!json_value.is<picojson::object>()) {
    throw ParsingError("Format specification must be an object");
  }

  const auto& obj = json_value.get<picojson::object>();

  // Get the type
  std::string type = get_string(obj, "type");

  if (type != "structural_tag") {
    throw ParsingError("Expected top-level type to be 'structural_tag', got '" + type + "'");
  }

  picojson::value format_json = get_value(obj, "format");
  auto format = create_format_from_json(format_json);

  return StructuralTag(format);
}

/**
 * @brief Parse input string using a JSON-specified format
 */
std::string parse_with_json(const std::string& input, const std::string& format_json_str) {
  try {
    auto tag_spec = parse_structural_tag_spec(format_json_str);
    auto result = parse_structural_tag(input, tag_spec);
    return result->to_json().serialize();
  } catch (const ParsingError& e) {
    picojson::object error_obj;
    error_obj["error"] = picojson::value(e.what());
    return picojson::value(error_obj).serialize();
  } catch (const std::exception& e) {
    picojson::object error_obj;
    error_obj["error"] = picojson::value(std::string("Unexpected error: ") + e.what());
    return picojson::value(error_obj).serialize();
  }
}

/**
 * @brief Parse a tags_and_text format - a mixture of text and tags
 */
StructuralTagResultPtr parse_tags_and_text(
    const std::string& text, const std::shared_ptr<TagsAndTextFormat>& format_spec
) {
  // Find all trigger positions
  auto trigger_positions = find_trigger_positions(text, format_spec->triggers);

  // If no triggers found but at_least_one is True, this is an error
  if (trigger_positions.empty() && format_spec->at_least_one) {
    throw ParsingError("No triggers found but at_least_one is True");
  }

  // If no triggers found, it's all text
  if (trigger_positions.empty()) {
    std::vector<TagsAndTextResult::TagsAndTextItem> result_items;
    result_items.emplace_back(text);
    return std::make_shared<TagsAndTextResult>(result_items);
  }

  std::vector<TagsAndTextResult::TagsAndTextItem> result_items;
  size_t last_end = 0;

  for (const auto& [pos, trigger] : trigger_positions) {
    // Add text before the trigger
    if (pos > last_end) {
      result_items.emplace_back(text.substr(last_end, pos - last_end));
    }

    // Find which tag matches this trigger
    std::vector<std::shared_ptr<TagFormat>> matching_tags;
    for (const auto& tag : format_spec->tags) {
      if (tag->begin.starts_with(trigger)) {
        matching_tags.push_back(tag);
      }
    }

    if (matching_tags.empty()) {
      // No matching tag, treat as plain text
      continue;
    }

    // Sort by length of begin string (longer first for specificity)
    std::sort(matching_tags.begin(), matching_tags.end(), [](const auto& a, const auto& b) {
      return a->begin.length() > b->begin.length();
    });

    // Try each tag until one matches
    bool matched = false;
    for (const auto& tag_spec : matching_tags) {
      if (text.substr(pos).starts_with(tag_spec->begin)) {
        // Find the end tag
        size_t content_start = pos + tag_spec->begin.length();
        size_t end_tag_pos = text.find(tag_spec->end, content_start);

        if (end_tag_pos != std::string::npos) {
          std::string content_text = text.substr(content_start, end_tag_pos - content_start);

          try {
            auto content_result = parse_format(content_text, tag_spec->content);

            auto tag_result =
                std::make_shared<TagResult>(tag_spec->begin, content_result, tag_spec->end);

            result_items.emplace_back(tag_result);
            last_end = end_tag_pos + tag_spec->end.length();
            matched = true;

            // If stop_after_first is True, we're done
            if (format_spec->stop_after_first) {
              break;
            }
          } catch (const std::exception&) {
            // If parsing fails, try the next tag
            continue;
          }
        }
      }

      if (matched && format_spec->stop_after_first) {
        break;
      }
    }

    if (format_spec->stop_after_first && matched) {
      break;
    }
  }

  // Add any remaining text
  if (last_end < text.length()) {
    result_items.emplace_back(text.substr(last_end));
  }

  return std::make_shared<TagsAndTextResult>(result_items);
}
