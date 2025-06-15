/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/persistent_stack.h
 * \brief The header for the definition of the persistent stack and stack elements.
 */
#ifndef XGRAMMAR_PERSISTENT_STACK_H_
#define XGRAMMAR_PERSISTENT_STACK_H_

#include <xgrammar/xgrammar.h>

#include <queue>
#include <unordered_set>
#include <vector>

#include "grammar_data_structure.h"
#include "grammar_serializer.h"

namespace xgrammar {

/*! \brief Specifies a stack element. It is mainly a position in a rule. */
struct StackElement {
  /*! \brief The rule's id. Used for debug purposes. */
  int32_t rule_id = -1;
  /*! \brief Which choice in this rule is selected. */
  int32_t sequence_id = -1;
  /*! \brief Which element of the choice sequence is to be visited. When the current sequence is
   * a tag dispatch rule, this element id the currently visited node. */
  int32_t element_id = -1;

  /*! \brief The number of left utf8 bytes in the current element. Used when the element is
   * a character class or a character class star. */
  int32_t left_utf8_bytes = 0;
  /*! \brief The next position to match in the current byte string. Used when the element is
   * a byte string. */
  int32_t element_in_string = 0;

  /*! \brief The id of the parent node in the PersistentStack. */
  int32_t parent_id = -1;

  /*! \brief The reference count of this StackElement. If reduces to zero, the node will be
   * removed from the StackElementBuffer. */
  int reference_count = 0;

  /*! \brief A parent_id value of kNoParent means this StackElement is the root of the tree. */
  static constexpr int32_t kNoParent = -1;

  constexpr StackElement() = default;
  constexpr StackElement(
      int32_t rule_id, int32_t sequence_id, int32_t element_id, int32_t parent_id = kNoParent
  )
      : rule_id(rule_id), sequence_id(sequence_id), element_id(element_id), parent_id(parent_id) {}

  // The element is invalid when sequence_id is -1.
  bool IsInvalid() const { return sequence_id == -1; }

  bool operator==(const StackElement& other) const {
    return rule_id == other.rule_id && sequence_id == other.sequence_id &&
           element_id == other.element_id && parent_id == other.parent_id &&
           left_utf8_bytes == other.left_utf8_bytes && element_in_string == other.element_in_string;
  }

  inline constexpr static int32_t kUnexpandedRuleStartSequenceId = 128000;

  inline constexpr static int32_t kDispatchedTagDispatchElementId = -1;
};

/*! \brief A special value for invalid StackElement. */
inline constexpr StackElement kInvalidStackElement(-1, -1, -1, -1);

/*! \brief A buffer to manage all StackElements. */
class StackElementBuffer {
 public:
  /*!
   * \brief Allocate a new StackElement. with given initial value.
   * \returns The id of the allocated node.
   */
  int32_t Allocate(StackElement stack_element) {
    int32_t id;
    if (free_nodes_.empty()) {
      buffer_.emplace_back();
      id = static_cast<int32_t>(buffer_.size()) - 1;
    } else {
      id = free_nodes_.back();
      XGRAMMAR_DCHECK(buffer_[id].IsInvalid());
      free_nodes_.pop_back();
    }
    stack_element.reference_count = 0;
    buffer_[id] = stack_element;
    return id;
  }

  /*! \brief Free the StackElement with the given id. */
  void Free(int32_t id) {
    XGRAMMAR_DCHECK(!buffer_[id].IsInvalid());
    buffer_[id] = kInvalidStackElement;
    free_nodes_.push_back(id);
  }

  /*! \brief Get the capacity of the buffer. */
  size_t Capacity() const { return buffer_.size(); }

  /*! \brief Get the number of allocated nodes. */
  size_t Size() const {
    XGRAMMAR_DCHECK(buffer_.size() >= free_nodes_.size());
    return buffer_.size() - free_nodes_.size();
  }

  /*! \brief Get the StackElement with the given id. */
  StackElement& operator[](int32_t id) {
    XGRAMMAR_DCHECK(id >= 0 && id < static_cast<int32_t>(buffer_.size()));
    XGRAMMAR_DCHECK(!buffer_[id].IsInvalid());
    return buffer_[id];
  }
  const StackElement& operator[](int32_t id) const {
    XGRAMMAR_DCHECK(id >= 0 && id < static_cast<int32_t>(buffer_.size()));
    XGRAMMAR_DCHECK(!buffer_[id].IsInvalid());
    return buffer_[id];
  }

  void Reset() {
    buffer_.clear();
    free_nodes_.clear();
  }

  friend class PersistentStack;

 private:
  /*! \brief The buffer to store all StackElements. */
  std::vector<StackElement> buffer_;
  /*! \brief A stack to store all free node ids. */
  std::vector<int32_t> free_nodes_;
};

/*!
 * \brief A tree structure to store all stacks. Every stack contains several StackElements, and
 * is represented as a path from the root to a leaf node.
 */
class PersistentStack {
 public:
  /*! \brief Construct a PersistentStack associated with the given grammar. */
  PersistentStack(const Grammar& grammar) : grammar_(grammar) {}

  /*!
   * \brief Create a new node with the given StackElement. The reference count of the new node
   * is zero.
   *
   * \note Later, this node should either be pointed by some child rule, or become a stack top
   * node (so it will be pointed to by an attached pointer) to be maintained in the
   * reference-counting based memory management.
   */
  int32_t NewNode(const StackElement& stack_element) {
    auto id = node_buffer_.Allocate(stack_element);
    if (stack_element.parent_id != StackElement::kNoParent) {
      XGRAMMAR_DCHECK(
          stack_element.parent_id < static_cast<int32_t>(node_buffer_.Capacity()) &&
          !node_buffer_[stack_element.parent_id].IsInvalid()
      );
      node_buffer_[stack_element.parent_id].reference_count++;
    }
    return id;
  }

  /*!
   * \brief Check if the given StackElement points to the end of the grammar. For a stack element,
   * if its rule id is the root rule id, and the element id equals to the length of the sequence it
   * refers to, it would be the end of the grammar.
   */
  bool IsEndOfGrammar(const StackElement& stack_element) const;

  /*! \brief Attach an additional reference to the node with the given id. */
  void AttachRefTo(int32_t id) {
    XGRAMMAR_DCHECK(id != StackElement::kNoParent);
    node_buffer_[id].reference_count++;
  }

  /*! \brief Remove a reference to the node with the given id. If the reference count becomes zero,
   * free the node and recursively all its ancestors with zero reference count. */
  void RemoveRefTo(int32_t id) {
    XGRAMMAR_DCHECK(id != StackElement::kNoParent);
    auto cur_node = id;
    while (cur_node != StackElement::kNoParent) {
      node_buffer_[cur_node].reference_count--;
      if (node_buffer_[cur_node].reference_count != 0) {
        break;
      }
      auto next_node = node_buffer_[cur_node].parent_id;
      node_buffer_.Free(cur_node);
      cur_node = next_node;
    }
  }

  /*! \brief Get the StackElement with the given id. */
  const StackElement& operator[](int32_t id) const {
    XGRAMMAR_DCHECK(id != StackElement::kNoParent);
    XGRAMMAR_DCHECK(!node_buffer_[id].IsInvalid());
    return node_buffer_[id];
  }

  /*! \brief Print the given stack_element to a string. */
  std::string PrintStackElement(const StackElement& stack_element) const;

  /*! \brief Print the stack_element associated with the given id to a string. */
  std::string PrintStackElement(int32_t id) const;

  /*! \brief Print the stack with the given top id to a string. */
  std::string PrintStackByTopId(int32_t top_id) const;

  /*!
   * \brief Check the well-formedness of the tree and the associated buffer. For debug purpose.
   * \details This function checks the following properties:
   * 1. Every node is pointed directly or indirectly by a outside pointer.
   * 2. Every node's reference count is consistent with the actual reference count.
   * 3. All ids and stack elements are valid.
   * 4. If a node in the buffer is free, it should be equal to kInvalidStackElement.
   */
  void CheckWellFormed(const std::vector<int32_t>& outside_pointers) const;

  /*! \brief Reset the tree and the associated buffer. */
  void Reset() { node_buffer_.Reset(); }

 private:
  /*! \brief The grammar associated with this PersistentStack. */
  Grammar grammar_;
  /*! \brief The buffer to store all StackElements. */
  StackElementBuffer node_buffer_;
};

/*!
 * \brief A class to maintain the stack tops and its history to support rollback.
 * \details This class helps to maintain nodes by automatically maintaining the attached references.
 * If a node is not existing in any stack in the history record, it will be freed.
 *
 * It can store up to the previous max_rollback_tokens + 1 steps of history, and thus supports
 * rolling back up to max_rollback_tokens steps.
 */
class StackTopsHistory {
 public:
  /*!
   * \param tree The PersistentStack to be associated with. Possibly modify the tree by
   * attaching and removing references to the stack top nodes.
   * \param max_rollback_tokens The maximum number of rollback tokens to be supported.
   */
  StackTopsHistory(PersistentStack* tree) : persistent_stack_(tree) {}

  /*!
   * \brief Push a new history record consisting a list of stack tops. These nodes will be recorded
   * as existing in a stack (by attaching a reference to them).
   * \param stack_tops The stack tops to be pushed.
   * \param drop_old Whether to drop the oldest history record if the history size exceeds the
   * limit. If the history is dropped, node that do not exist in any stack any more will be freed.
   */
  void PushHistory(const std::vector<int32_t>& stack_tops) {
    stack_tops_history_.push_back(stack_tops);
    for (auto id : stack_tops) {
      persistent_stack_->AttachRefTo(id);
    }
  }

  /*! \brief Roll back to several previous steps. Possibly frees node that do not exist in any stack
   * any more. */
  void Rollback(int rollback_steps) {
    XGRAMMAR_DCHECK(rollback_steps < static_cast<int>(stack_tops_history_.size()))
        << "The number of requested rollback tokens is greater than or equal to the current "
           "history "
        << "size: " << rollback_steps << " vs " << stack_tops_history_.size() << ".";
    while (rollback_steps--) {
      PopLatest();
    }
  }

  /*! \brief Discard the earliest several steps. Possibly frees node that do not exist in any stack
   * any more. */
  void DiscardEarliest(int discard_steps) {
    XGRAMMAR_DCHECK(discard_steps < static_cast<int>(stack_tops_history_.size()))
        << "The number of requested discard steps is greater than or equal to the current "
           "history "
        << "size: " << discard_steps << " vs " << stack_tops_history_.size() << ".";
    while (discard_steps--) {
      PopEarliest();
    }
  }

  /*! \brief Get the latest stack tops. */
  const std::vector<int32_t>& GetLatest() const { return stack_tops_history_.back(); }

  /*!
   * \brief Print one history record.
   * \param steps_ago The number of steps behind the latest record. 0 means the
   * latest record.
   */
  std::string PrintHistory(int steps_ago = 0) const;

  /*! \brief Get the number of history records. */
  int Size() const { return stack_tops_history_.size(); }

  /*! \brief Check the well-formedness of the tree and the associated buffer. */
  void CheckWellFormed() const;

  /*! \brief Reset the history and the associated node tree. */
  void Reset() {
    stack_tops_history_.clear();
    persistent_stack_->Reset();
  }

 private:
  /*! \brief Pop the oldest history record. Possibly frees node that do not exist in any stack any
   * more. */
  void PopEarliest() {
    const auto& old_stack_tops = stack_tops_history_.front();
    for (auto id : old_stack_tops) {
      persistent_stack_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_front();
  }

  /*! \brief Pop the latest history record. Possibly frees node that do not exist in any stack any
   * more. */
  void PopLatest() {
    const auto& new_stack_tops = stack_tops_history_.back();
    for (auto id : new_stack_tops) {
      persistent_stack_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_back();
  }

  /*! \brief Modifiable pointer to the PersistentStack. */
  PersistentStack* persistent_stack_;
  /*! \brief The history of stack tops. */
  std::deque<std::vector<int32_t>> stack_tops_history_;
};

inline bool PersistentStack::IsEndOfGrammar(const StackElement& stack_element) const {
  if (stack_element.parent_id != StackElement::kNoParent) {
    return false;
  }
  auto seq_expr = grammar_->GetRuleExpr(stack_element.sequence_id);
  if (seq_expr.type == Grammar::Impl::RuleExprType::kTagDispatch) {
    return stack_element.element_id != -1;
  } else {
    return seq_expr.size() == stack_element.element_id;
  }
}

inline std::string PersistentStack::PrintStackElement(int32_t id) const {
  return "id: " + std::to_string(id) + ", " + PrintStackElement(node_buffer_[id]);
}

inline std::string PersistentStack::PrintStackElement(const StackElement& stack_element) const {
  std::stringstream ss;
  ss << "StackElement: rule " << stack_element.rule_id;
  if (stack_element.rule_id != -1) {
    ss << ": " << grammar_->GetRule(stack_element.rule_id).name;
  }

  if (stack_element.sequence_id == StackElement::kUnexpandedRuleStartSequenceId) {
    ss << ", unexpanded rule start sequence";
  } else {
    ss << ", sequence " << stack_element.sequence_id << ": "
       << GrammarPrinter(grammar_).PrintRuleExpr(stack_element.sequence_id);

    if (stack_element.sequence_id == StackElement::kDispatchedTagDispatchElementId) {
      ss << ", dispatched tag dispatch";
    } else {
      ss << ", element id: " << stack_element.element_id;
      auto sequence = grammar_->GetRuleExpr(stack_element.sequence_id);
      if (sequence.type != Grammar::Impl::RuleExprType::kTagDispatch &&
          stack_element.element_id < static_cast<int32_t>(sequence.size())) {
        auto element = grammar_->GetRuleExpr(sequence[stack_element.element_id]);
        if (element.type == Grammar::Impl::RuleExprType::kByteString) {
          ss << ", element in string: " << stack_element.element_in_string;
        } else if (element.type == Grammar::Impl::RuleExprType::kCharacterClass ||
                   element.type == Grammar::Impl::RuleExprType::kCharacterClassStar) {
          ss << ", left utf8 bytes: " << stack_element.left_utf8_bytes;
        }
      }
    }
  }

  ss << ", parent id: " << stack_element.parent_id
     << ", ref count: " << stack_element.reference_count;
  return ss.str();
}

inline std::string PersistentStack::PrintStackByTopId(int32_t top_id) const {
  std::stringstream ss;
  std::vector<int32_t> stack;
  for (auto cur_id = top_id; cur_id != StackElement::kNoParent;
       cur_id = node_buffer_[cur_id].parent_id) {
    stack.push_back(cur_id);
  }
  ss << "{\n";
  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    ss << PrintStackElement(*it) << "\n";
  }
  ss << "}";
  return ss.str();
}

inline void PersistentStack::CheckWellFormed(const std::vector<int32_t>& outside_pointers) const {
  const auto& buffer = node_buffer_.buffer_;
  std::unordered_set<int32_t> free_nodes_set(
      node_buffer_.free_nodes_.begin(), node_buffer_.free_nodes_.end()
  );
  int buffer_size = static_cast<int>(buffer.size());
  std::vector<int> new_reference_counter(buffer_size, 0);
  std::vector<bool> visited(buffer_size, false);
  std::queue<int> visit_queue;
  for (auto id : outside_pointers) {
    XGRAMMAR_CHECK(id >= 0 && id < buffer_size);
    XGRAMMAR_CHECK(!buffer[id].IsInvalid());
    new_reference_counter[id]++;
    if (visited[id] == false) {
      visited[id] = true;
      visit_queue.push(id);
    }
  }
  while (!visit_queue.empty()) {
    auto cur_id = visit_queue.front();
    visit_queue.pop();
    const auto& stack_element = buffer[cur_id];
    if (stack_element.parent_id != StackElement::kNoParent) {
      XGRAMMAR_CHECK(stack_element.parent_id >= 0 && stack_element.parent_id < buffer_size);
      XGRAMMAR_CHECK(!buffer[stack_element.parent_id].IsInvalid());
      new_reference_counter[stack_element.parent_id]++;
      if (visited[stack_element.parent_id] == false) {
        visited[stack_element.parent_id] = true;
        visit_queue.push(stack_element.parent_id);
      }
    }
  }

  for (int i = 0; i < static_cast<int32_t>(buffer.size()); ++i) {
    if (free_nodes_set.count(i)) {
      XGRAMMAR_CHECK(buffer[i].IsInvalid());
      XGRAMMAR_CHECK(visited[i] == false);
    } else {
      XGRAMMAR_CHECK(visited[i] == true);
      XGRAMMAR_CHECK(!buffer[i].IsInvalid());
      XGRAMMAR_CHECK(new_reference_counter[i] == buffer[i].reference_count)
          << "Reference counters unmatch for node #" << i << ": Updated "
          << new_reference_counter[i] << ", Original " << buffer[i].reference_count;
    }
  }
}

inline std::string StackTopsHistory::PrintHistory(int steps_ago) const {
  const auto& latest_tops =
      stack_tops_history_[static_cast<int64_t>(stack_tops_history_.size()) - 1 - steps_ago];
  std::stringstream ss;
  ss << "Num of stacks: " << latest_tops.size() << std::endl;
  int cnt = 0;
  for (auto id : latest_tops) {
    ss << "Stack #" << cnt << ": " << persistent_stack_->PrintStackByTopId(id) << "\n";
    ++cnt;
  }
  return ss.str();
}

inline void StackTopsHistory::CheckWellFormed() const {
  std::vector<int32_t> outside_pointers;
  for (const auto& stack_tops : stack_tops_history_) {
    outside_pointers.insert(outside_pointers.end(), stack_tops.begin(), stack_tops.end());
  }
  persistent_stack_->CheckWellFormed(outside_pointers);
}

}  // namespace xgrammar

#endif  // XGRAMMAR_PERSISTENT_STACK_H_
