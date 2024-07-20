/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_serializer.cc
 */

#include "grammar_serializer.h"

#include <picojson.h>
#include <xgrammar/support/encoding.h>

namespace xgrammar {

std::string BNFGrammarPrinter::PrintRule(const Rule& rule) {
  std::string res = rule.name + " ::= " + PrintRuleExpr(rule.body_expr_id);
  if (rule.lookahead_assertion_id != -1) {
    res += " (=" + PrintRuleExpr(rule.lookahead_assertion_id) + ")";
  }
  return res;
}

std::string BNFGrammarPrinter::PrintRule(int32_t rule_id) {
  return PrintRule(grammar_->GetRule(rule_id));
}

std::string BNFGrammarPrinter::PrintRuleExpr(const RuleExpr& rule_expr) {
  std::string result;
  switch (rule_expr.type) {
    case RuleExprType::kByteString:
      return PrintByteString(rule_expr);
    case RuleExprType::kCharacterClass:
      return PrintCharacterClass(rule_expr);
    case RuleExprType::kCharacterClassStar:
      return PrintCharacterClassStar(rule_expr);
    case RuleExprType::kEmptyStr:
      return PrintEmptyStr(rule_expr);
    case RuleExprType::kRuleRef:
      return PrintRuleRef(rule_expr);
    case RuleExprType::kSequence:
      return PrintSequence(rule_expr);
    case RuleExprType::kChoices:
      return PrintChoices(rule_expr);
    default:
      XGRAMMAR_LOG(FATAL) << "Unexpected RuleExpr type: " << static_cast<int>(rule_expr.type);
  }
}

std::string BNFGrammarPrinter::PrintRuleExpr(int32_t rule_expr_id) {
  return PrintRuleExpr(grammar_->GetRuleExpr(rule_expr_id));
}

std::string BNFGrammarPrinter::PrintByteString(const RuleExpr& rule_expr) {
  std::string internal_str;
  internal_str.reserve(rule_expr.data_len);
  for (int i = 0; i < rule_expr.data_len; ++i) {
    internal_str += static_cast<char>(rule_expr[i]);
  }
  auto codepoints = ParseUTF8(internal_str.c_str(), UTF8ErrorPolicy::kReturnByte);
  std::string result;
  for (auto codepoint : codepoints) {
    result += PrintAsEscapedUTF8(codepoint);
  }
  return "\"" + result + "\"";
}

std::string BNFGrammarPrinter::PrintCharacterClass(const RuleExpr& rule_expr) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {
      {'-', "\\-"}, {']', "\\]"}};
  std::string result = "[";
  bool is_negative = static_cast<bool>(rule_expr[0]);
  if (is_negative) {
    result += "^";
  }
  for (auto i = 1; i < rule_expr.data_len; i += 2) {
    result += PrintAsEscapedUTF8(rule_expr[i], kCustomEscapeMap);
    if (rule_expr[i] == rule_expr[i + 1]) {
      continue;
    }
    result += "-";
    result += PrintAsEscapedUTF8(rule_expr[i + 1], kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string BNFGrammarPrinter::PrintCharacterClassStar(const RuleExpr& rule_expr) {
  return PrintCharacterClass(rule_expr) + "*";
}

std::string BNFGrammarPrinter::PrintEmptyStr(const RuleExpr& rule_expr) { return "\"\""; }

std::string BNFGrammarPrinter::PrintRuleRef(const RuleExpr& rule_expr) {
  return grammar_->GetRule(rule_expr[0]).name;
}

std::string BNFGrammarPrinter::PrintSequence(const RuleExpr& rule_expr) {
  std::string result;
  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " ";
    }
  }
  result += ")";
  return result;
}

std::string BNFGrammarPrinter::PrintChoices(const RuleExpr& rule_expr) {
  std::string result;

  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " | ";
    }
  }
  result += ")";
  return result;
}

std::string BNFGrammarPrinter::ToString() {
  std::string result;
  int num_rules = grammar_->NumRules();
  for (auto i = 0; i < num_rules; ++i) {
    result += PrintRule(grammar_->GetRule(i)) + "\n";
  }
  return result;
}

// TVM_REGISTER_GLOBAL("mlc.grammar.BNFGrammarToString").set_body_typed([](const BNFGrammar&
// grammar) {
//   return BNFGrammarPrinter(grammar).ToString();
// });

}  // namespace xgrammar
