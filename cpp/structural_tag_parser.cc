/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/parser_structural_tag.cc
 */

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "parser_structural_tag.h"
#include "picojson.h"

namespace xgrammar {

// Forward declarations
picojson::value parse_format(const std::string& text, const picojson::value& format_spec);
picojson::value parse_literal(const std::string& text, const picojson::value& format_spec);
picojson::value parse_json_schema(const std::string& text, const picojson::value& format_spec);
picojson::value parse_wildcard_text(const std::string& text, const picojson::value& format_spec);
picojson::value parse_tag(const std::string& text, const picojson::value& format_spec);
picojson::value parse_sequence(const std::string& text, const picojson::value& format_spec);
picojson::value parse_tags_and_text(const std::string& text, const picojson::value& format_spec);
picojson::value parse_tags_with_separator(
    const std::string& text, const picojson::value& format_spec
);
std::vector<std::pair<size_t, std::string>> find_trigger_positions(
    const std::string& text, const std::vector<std::string>& triggers
);

class ParsingError : public std::runtime_error {
 public:
  explicit ParsingError(const std::string& message) : std::runtime_error(message) {}
};

bool starts_with(const std::string& str, const std::string& prefix) {
  return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string parse_structural_tag(
    const std::string& response, const std::string& structural_tag_json
) {
  try {
    // Parse the structural tag JSON
    picojson::value structural_tag;
    std::string err = picojson::parse(structural_tag, structural_tag_json);
    if (!err.empty()) {
      throw ParsingError("Failed to parse structural tag JSON: " + err);
    }

    // Validate it's an object with type "structural_tag" and has a "format" field
    if (!structural_tag.is<picojson::object>() || !structural_tag.contains("type") ||
        !structural_tag.get("type").is<std::string>() ||
        structural_tag.get("type").get<std::string>() != "structural_tag" ||
        !structural_tag.contains("format")) {
      throw ParsingError("Invalid structural tag: missing required fields");
    }

    // Parse the response according to the format
    picojson::value format = structural_tag.get("format");
    picojson::value result = parse_format(response, format);

    // Serialize the result back to JSON
    return result.serialize();
  } catch (const ParsingError& e) {
    // Return error as JSON
    picojson::object error_obj;
    error_obj["error"] = picojson::value(e.what());
    return picojson::value(error_obj).serialize();
  } catch (const std::exception& e) {
    // Return error as JSON
    picojson::object error_obj;
    error_obj["error"] = picojson::value(std::string("Unexpected error: ") + e.what());
    return picojson::value(error_obj).serialize();
  }
}

picojson::value parse_format(const std::string& text, const picojson::value& format_spec) {
  if (!format_spec.is<picojson::object>() || !format_spec.contains("type")) {
    throw ParsingError("Invalid format specification: missing type");
  }

  std::string type = format_spec.get("type").get<std::string>();

  if (type == "literal") {
    return parse_literal(text, format_spec);
  } else if (type == "json_schema") {
    return parse_json_schema(text, format_spec);
  } else if (type == "wildcard_text") {
    return parse_wildcard_text(text, format_spec);
  } else if (type == "tag") {
    return parse_tag(text, format_spec);
  } else if (type == "sequence") {
    return parse_sequence(text, format_spec);
  } else if (type == "tags_and_text") {
    return parse_tags_and_text(text, format_spec);
  } else if (type == "tags_with_separator") {
    return parse_tags_with_separator(text, format_spec);
  } else {
    throw ParsingError("Unknown format type: " + type);
  }
}

picojson::value parse_literal(const std::string& text, const picojson::value& format_spec) {
  if (!format_spec.contains("text") || !format_spec.get("text").is<std::string>()) {
    throw ParsingError("Invalid literal format: missing text field");
  }

  std::string expected_text = format_spec.get("text").get<std::string>();

  if (text != expected_text) {
    throw ParsingError("Expected literal '" + expected_text + "', got '" + text + "'");
  }

  picojson::object result;
  result["type"] = picojson::value("literal");
  result["text"] = picojson::value(text);
  return picojson::value(result);
}

picojson::value parse_json_schema(const std::string& text, const picojson::value& format_spec) {
  if (!format_spec.contains("json_schema") ||
      !format_spec.get("json_schema").is<picojson::object>()) {
    throw ParsingError("Invalid json_schema format: missing json_schema field");
  }

  // Try to parse the text as JSON
  picojson::value json_data;
  std::string err = picojson::parse(json_data, text);
  if (!err.empty()) {
    throw ParsingError("Invalid JSON: " + err);
  }

  // TODO: Add JSON schema validation if needed

  picojson::object result;
  result["type"] = picojson::value("json_schema");
  result["value"] = json_data;
  return picojson::value(result);
}

picojson::value parse_wildcard_text(const std::string& text, const picojson::value& format_spec) {
  picojson::object result;
  result["type"] = picojson::value("wildcard_text");
  result["text"] = picojson::value(text);
  return picojson::value(result);
}

picojson::value parse_tag(const std::string& text, const picojson::value& format_spec) {
  if (!format_spec.contains("begin") || !format_spec.get("begin").is<std::string>() ||
      !format_spec.contains("end") || !format_spec.get("end").is<std::string>() ||
      !format_spec.contains("content")) {
    throw ParsingError("Invalid tag format: missing required fields");
  }

  std::string begin_tag = format_spec.get("begin").get<std::string>();
  std::string end_tag = format_spec.get("end").get<std::string>();

  if (!text.starts_with(begin_tag)) {
    throw ParsingError(
        "Expected tag beginning '" + begin_tag + "' at: '" + text.substr(0, 50) + "...'"
    );
  }

  if (!text.ends_with(end_tag)) {
    throw ParsingError(
        "Expected tag ending '" + end_tag + "' at the end of: '..." +
        text.substr(text.size() - std::min(text.size(), size_t(50))) + "'"
    );
  }

  std::string content_text =
      text.substr(begin_tag.size(), text.size() - begin_tag.size() - end_tag.size());
  picojson::value content_result = parse_format(content_text, format_spec.get("content"));

  picojson::object result;
  result["type"] = picojson::value("tag");
  result["begin"] = picojson::value(begin_tag);
  result["content"] = content_result;
  result["end"] = picojson::value(end_tag);
  return picojson::value(result);
}

picojson::value parse_sequence(const std::string& text, const picojson::value& format_spec) {
  if (!format_spec.contains("elements") || !format_spec.get("elements").is<picojson::array>()) {
    throw ParsingError("Invalid sequence format: missing elements array");
  }

  picojson::array elements_spec = format_spec.get("elements").get<picojson::array>();
  picojson::array result_elements;

  std::string remaining_text = text;

  for (const auto& element_spec : elements_spec) {
    if (remaining_text.empty()) {
      throw ParsingError("Unexpected end of text in sequence");
    }

    // Handle different element types
    std::string element_type = element_spec.get("type").get<std::string>();

    if (element_type == "literal") {
      std::string literal_text = element_spec.get("text").get<std::string>();

      if (!remaining_text.starts_with(literal_text)) {
        throw ParsingError(
            "Expected literal '" + literal_text + "' at: '" +
            remaining_text.substr(0, std::min(remaining_text.size(), size_t(50))) + "...'"
        );
      }

      picojson::object element_result;
      element_result["type"] = picojson::value("literal");
      element_result["text"] = picojson::value(literal_text);
      result_elements.push_back(picojson::value(element_result));

      remaining_text = remaining_text.substr(literal_text.size());
    } else if (element_type == "tag") {
      std::string begin_tag = element_spec.get("begin").get<std::string>();
      std::string end_tag = element_spec.get("end").get<std::string>();

      if (!remaining_text.starts_with(begin_tag)) {
        throw ParsingError(
            "Expected tag beginning '" + begin_tag + "' at: '" +
            remaining_text.substr(0, std::min(remaining_text.size(), size_t(50))) + "...'"
        );
      }

      size_t content_start = begin_tag.size();
      size_t end_pos = remaining_text.find(end_tag, content_start);

      if (end_pos == std::string::npos) {
        throw ParsingError(
            "Could not find end tag '" + end_tag + "' in: '" +
            remaining_text.substr(0, std::min(remaining_text.size(), size_t(100))) + "...'"
        );
      }

      std::string content_text = remaining_text.substr(content_start, end_pos - content_start);
      picojson::value content_result = parse_format(content_text, element_spec.get("content"));

      picojson::object tag_result;
      tag_result["type"] = picojson::value("tag");
      tag_result["begin"] = picojson::value(begin_tag);
      tag_result["content"] = content_result;
      tag_result["end"] = picojson::value(end_tag);

      result_elements.push_back(picojson::value(tag_result));

      remaining_text = remaining_text.substr(end_pos + end_tag.size());
    } else {
      // For other types, parse the whole remaining text
      picojson::value element_result = parse_format(remaining_text, element_spec);
      result_elements.push_back(element_result);
      remaining_text = "";
    }
  }

  picojson::object result;
  result["type"] = picojson::value("sequence");
  result["elements"] = picojson::value(result_elements);
  return picojson::value(result);
}

std::vector<std::pair<size_t, std::string>> find_trigger_positions(
    const std::string& text, const std::vector<std::string>& triggers
) {
  std::vector<std::pair<size_t, std::string>> positions;

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
    return a.first < b.first;
  });

  return positions;
}

picojson::value parse_tags_and_text(const std::string& text, const picojson::value& format_spec) {
  if (!format_spec.contains("tags") || !format_spec.get("tags").is<picojson::array>() ||
      !format_spec.contains("triggers") || !format_spec.get("triggers").is<picojson::array>()) {
    throw ParsingError("Invalid tags_and_text format: missing required fields");
  }

  picojson::array tags_spec = format_spec.get("tags").get<picojson::array>();
  picojson::array triggers_json = format_spec.get("triggers").get<picojson::array>();

  bool at_least_one = false;
  if (format_spec.contains("at_least_one") && format_spec.get("at_least_one").is<bool>()) {
    at_least_one = format_spec.get("at_least_one").get<bool>();
  }

  bool stop_after_first = false;
  if (format_spec.contains("stop_after_first") && format_spec.get("stop_after_first").is<bool>()) {
    stop_after_first = format_spec.get("stop_after_first").get<bool>();
  }

  std::vector<std::string> triggers;
  for (const auto& trigger : triggers_json) {
    triggers.push_back(trigger.get<std::string>());
  }

  // Find all trigger positions
  auto trigger_positions = find_trigger_positions(text, triggers);

  // If no triggers found but at_least_one is true, this is an error
  if (trigger_positions.empty() && at_least_one) {
    throw ParsingError("No triggers found but at_least_one is true");
  }

  // If no triggers found, it's all text
  if (trigger_positions.empty()) {
    picojson::object result;
    result["type"] = picojson::value("tags_and_text");

    picojson::array tags_and_text;
    tags_and_text.push_back(picojson::value(text));

    result["tags_and_text"] = picojson::value(tags_and_text);
    return picojson::value(result);
  }

  picojson::array result_items;
  size_t last_end = 0;

  for (const auto& [pos, trigger] : trigger_positions) {
    // Add text before the trigger
    if (pos > last_end) {
      result_items.push_back(picojson::value(text.substr(last_end, pos - last_end)));
    }

    // Find which tag matches this trigger
    std::vector<picojson::value> matching_tags;
    for (const auto& tag_spec : tags_spec) {
      std::string begin = tag_spec.get("begin").get<std::string>();
      if (begin.starts_with(trigger)) {
        matching_tags.push_back(tag_spec);
      }
    }

    if (matching_tags.empty()) {
      // No matching tag, treat as plain text
      continue;
    }

    // Sort by length of begin string (longer first for specificity)
    std::sort(matching_tags.begin(), matching_tags.end(), [](const auto& a, const auto& b) {
      return a.get("begin").get<std::string>().size() > b.get("begin").get<std::string>().size();
    });

    // Try each tag until one matches
    bool matched = false;
    for (const auto& tag_spec : matching_tags) {
      std::string begin_tag = tag_spec.get("begin").get<std::string>();
      std::string end_tag = tag_spec.get("end").get<std::string>();

      if (text.substr(pos).starts_with(begin_tag)) {
        // Find the end tag
        size_t content_start = pos + begin_tag.size();
        size_t end_tag_pos = text.find(end_tag, content_start);

        if (end_tag_pos != std::string::npos) {
          std::string content_text = text.substr(content_start, end_tag_pos - content_start);

          try {
            picojson::value content_result = parse_format(content_text, tag_spec.get("content"));

            picojson::object tag_result;
            tag_result["type"] = picojson::value("tag");
            tag_result["begin"] = picojson::value(begin_tag);
            tag_result["content"] = content_result;
            tag_result["end"] = picojson::value(end_tag);

            result_items.push_back(picojson::value(tag_result));
            last_end = end_tag_pos + end_tag.size();
            matched = true;

            // If stop_after_first is true, we're done
            if (stop_after_first) {
              break;
            }
          } catch (const std::exception&) {
            // If parsing fails, try the next tag
            continue;
          }
        }
      }

      if (matched && stop_after_first) {
        break;
      }
    }

    if (stop_after_first && matched) {
      break;
    }
  }

  // Add any remaining text
  if (last_end < text.size()) {
    result_items.push_back(picojson::value(text.substr(last_end)));
  }

  picojson::object result;
  result["type"] = picojson::value("tags_and_text");
  result["tags_and_text"] = picojson::value(result_items);
  return picojson::value(result);
}

picojson::value parse_tags_with_separator(
    const std::string& text, const picojson::value& format_spec
) {
  if (!format_spec.contains("tags") || !format_spec.get("tags").is<picojson::array>() ||
      !format_spec.contains("separator") || !format_spec.get("separator").is<std::string>()) {
    throw ParsingError("Invalid tags_with_separator format: missing required fields");
  }

  picojson::array tags_spec = format_spec.get("tags").get<picojson::array>();
  std::string separator = format_spec.get("separator").get<std::string>();

  bool at_least_one = false;
  if (format_spec.contains("at_least_one") && format_spec.get("at_least_one").is<bool>()) {
    at_least_one = format_spec.get("at_least_one").get<bool>();
  }

  bool stop_after_first = false;
  if (format_spec.contains("stop_after_first") && format_spec.get("stop_after_first").is<bool>()) {
    stop_after_first = format_spec.get("stop_after_first").get<bool>();
  }

  // Split the text by the separator
  std::vector<std::string> parts;
  size_t start = 0;
  size_t end;

  while ((end = text.find(separator, start)) != std::string::npos) {
    parts.push_back(text.substr(start, end - start));
    start = end + separator.size();
  }
  parts.push_back(text.substr(start));

  // If at_least_one is true, we need at least one part
  if (at_least_one && parts.empty()) {
    throw ParsingError("No tags found but at_least_one is true");
  }

  picojson::array tag_results;

  for (const auto& part : parts) {
    if (part.empty()) {
      continue;  // Skip empty parts
    }

    // Try to match this part with each tag
    bool matched = false;
    for (const auto& tag_spec : tags_spec) {
      std::string begin_tag = tag_spec.get("begin").get<std::string>();
      std::string end_tag = tag_spec.get("end").get<std::string>();

      if (part.starts_with(begin_tag) && part.ends_with(end_tag)) {
        std::string content_text =
            part.substr(begin_tag.size(), part.size() - begin_tag.size() - end_tag.size());

        try {
          picojson::value content_result = parse_format(content_text, tag_spec.get("content"));

          picojson::object tag_result;
          tag_result["type"] = picojson::value("tag");
          tag_result["begin"] = picojson::value(begin_tag);
          tag_result["content"] = content_result;
          tag_result["end"] = picojson::value(end_tag);

          tag_results.push_back(picojson::value(tag_result));
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

    // If stop_after_first is true, we're done
    if (stop_after_first && matched) {
      break;
    }
  }

  picojson::object result;
  result["type"] = picojson::value("tags_with_separator");
  result["tags"] = picojson::value(tag_results);
  result["separator"] = picojson::value(separator);
  return picojson::value(result);
}

}  // namespace xgrammar
