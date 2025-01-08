/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/function_calling.cc
 */

#include <xgrammar/function_calling.h>

#include <unordered_map>

#include "picojson.h"

namespace xgrammar {

/******************* Parser *******************/

std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> parse_message(
    const std::string& message_body
) {
  std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> tool_calls;
  size_t i = 0;
  const size_t length = message_body.length();

  while (i < length) {
    // <function> tags
    if (i + 10 < length && message_body.substr(i, 10) == "<function=") {
      size_t name_start = i + 10;
      size_t name_end = message_body.find('>', name_start);
      if (name_end == std::string::npos) {
        i++;
        continue;
      }

      std::string function_name = message_body.substr(name_start, name_end - name_start);

      size_t params_start = name_end + 1;
      size_t params_end = message_body.find("</function>", params_start);
      if (params_end == std::string::npos) {
        i++;
        continue;
      }

      std::string params_str = message_body.substr(params_start, params_end - params_start);
      size_t pos = 0;
      while ((pos = params_str.find('\'', pos)) != std::string::npos) {
        params_str.replace(pos, 1, "\"");
        pos++;
      }

      picojson::value v;
      std::string err = picojson::parse(v, params_str);
      if (!err.empty()) {
        i++;
        continue;
      }

      if (v.is<picojson::object>()) {
        const picojson::object& obj = v.get<picojson::object>();
        std::unordered_map<std::string, std::string> params;

        for (const auto& pair : obj) {
          if (pair.second.is<std::string>()) {
            params[pair.first] = pair.second.get<std::string>();
          }
        }
        tool_calls.emplace_back(function_name, params);
      }

      i = params_end + 11;
    }
    // json format
    else if (i < length && message_body[i] == '{') {
      int bracket_count = 1;
      size_t j = i + 1;

      while (j < length && bracket_count > 0) {
        if (message_body[j] == '{') {
          bracket_count++;
        } else if (message_body[j] == '}') {
          bracket_count--;
        }
        j++;
      }

      if (bracket_count == 0) {
        std::string json_str = message_body.substr(i, j - i);
        picojson::value v;
        std::string err = picojson::parse(v, json_str);
        if (!err.empty()) {
          i = j;
          continue;
        }

        if (v.is<picojson::object>()) {
          const picojson::object& obj = v.get<picojson::object>();
          auto name_it = obj.find("name");
          auto params_it = obj.find("parameters");

          if (name_it != obj.end() && params_it != obj.end() && name_it->second.is<std::string>() &&
              params_it->second.is<picojson::object>()) {
            std::string name = name_it->second.get<std::string>();
            const picojson::object& params_obj = params_it->second.get<picojson::object>();

            std::unordered_map<std::string, std::string> params;
            for (const auto& pair : params_obj) {
              if (pair.second.is<std::string>()) {
                params[pair.first] = pair.second.get<std::string>();
              }
            }
            tool_calls.emplace_back(name, params);
          }
        }
        i = j;
      } else {
        i++;
      }
    } else {
      i++;
    }
  }

  return tool_calls;
}

}  // namespace xgrammar
