/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/parser_structural_tag.h
 */

#ifndef XGRAMMAR_PARSER_STRUCTURAL_TAG_H_
#define XGRAMMAR_PARSER_STRUCTURAL_TAG_H_

#include <string>
#include <vector>

namespace xgrammar {

/**
 * @brief Parse a response string according to a structural tag format.
 *
 * @param response The string response from the LLM
 * @param structural_tag_json The structural tag specification as a JSON string
 *
 * @return A JSON string representing the parsed response
 */
std::string parse_structural_tag(
    const std::string& response, const std::string& structural_tag_json
);

}  // namespace xgrammar

#endif  // XGRAMMAR_PARSER_STRUCTURAL_TAG_H_
