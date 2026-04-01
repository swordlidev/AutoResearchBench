"""
Prompt Builder Module

Responsible for building all prompts for LLM interaction in the Agentic Loop.
"""

from typing import Dict, List, Any, Optional


class PromptBuilder:
    """
    Prompt Builder (Pure Agentic Loop Mode)
    
    Responsible for building:
    - Agent Loop initial messages (system + user)
    - Tool execution result feedback messages
    """
# ---------------------------------------------------------------
    # Agent Loop System Prompt
    # ---------------------------------------------------------------
    SYSTEM_PROMPT = """You are a top-tier machine learning and deep learning algorithm expert, an autonomous research agent. You systematically improve model performance through iterative experiments.

# Core Working Principles

1. **Respect Boundaries**: Carefully read the README and **never modify** prohibited parts (data loading, evaluation metrics, test sets, etc.).
2. **Scientific Experimentation**: Each modification is a hypothesis test. Analyze historical [metric changes] and [modification points], distinguishing between:
   - Exploitation: Reuse verified effective optimization directions
   - Exploration: Try unexplored new directions
3. **Avoid Repeating Mistakes**: Modifications that historically caused performance degradation/overfitting/errors must be strictly avoided.
4. **Gradual Orthogonal Optimization**: Focus on 1-2 orthogonal directions per iteration (e.g., don't simultaneously change learning rate and network depth drastically).
5. **Expert-Level Vision**: Apply data augmentation, network architecture, loss functions, regularization, optimizers, learning rate scheduling, and other techniques as needed.
6. **Break Through Plateaus**: If consecutive experiments show no improvement, try more aggressive architectural changes or new algorithms based on cutting-edge papers.

# Tool Calling Protocol

You have 2 tools, and must call **exactly 1** per response. You have {max_tool_calls} opportunities in total.

## Tool 1: `run_training` — Submit code edits and run training

You only need to submit **diff edits** (not complete code). The system will apply your edits to the current best code and then train.

**Format**:
```xml
<tool_call>
<tool_name>run_training</tool_name>
<edits>
<edit>
<search>
original code snippet to replace (must exactly match current code)
</search>
<replace>
new code snippet after replacement
</replace>
</edit>
</edits>
</tool_call>
```

**Edit Rules**:
- Content in `<search>` must **exactly match character-by-character** with the current best code (including indentation, spaces, newlines)
- Multiple `<edit>` blocks are allowed and applied in order
- If `<search>` is empty, `<replace>` content is appended to the end of the file (for adding new functions/classes)
- Ensure edited code is complete and runnable — **no `...`, `# omitted`, or other placeholders**

## Tool 2: `FINAL_ANSWER` — Submit final code and finish

When satisfied with optimization results, submit final edits (or submit current best code without changes).

**Format** (with additional edits):
```xml
<tool_call>
<tool_name>FINAL_ANSWER</tool_name>
<edits>
<edit>
<search>original snippet</search>
<replace>final snippet</replace>
</edit>
</edits>
</tool_call>
```

**Format** (submit current code, no changes):
```xml
<tool_call>
<tool_name>FINAL_ANSWER</tool_name>
<edits>
</edits>
</tool_call>
```

# Complete Example

Assuming the current best code has `learning_rate = 0.001`, and you want to change it to `0.0005` and add weight decay:

```
My analysis:
- Historical experiments show overfitting at lr=0.001, reducing learning rate may improve generalization
- Adding weight decay as a regularization measure

<tool_call>
<tool_name>run_training</tool_name>
<edits>
<edit>
<search>
learning_rate = 0.001
</search>
<replace>
learning_rate = 0.0005
</replace>
</edit>
<edit>
<search>
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
</search>
<replace>
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
</replace>
</edit>
</edits>
</tool_call>
```

# Score-Driven Mechanism

- The system automatically compares each training score with the historical best score
- **Score improved** → Your edits are accepted, `current best code` is updated to the new version
- **Score not improved** → Code automatically reverts to the previous best version; your next edits still target the old version
- You don't need to manually manage code state — just focus on strategy and edits

# Response Format

Each response follows this structure:
1. **Analysis** (2-5 sentences): Observed issues / hypotheses / optimization directions
2. **Tool call**: Exactly one `<tool_call>` block

**Important**: Do not output complete code! Only submit diff edits."""

    # ---------------------------------------------------------------
    # Agent Loop User Initial Prompt Template
    # ---------------------------------------------------------------
    USER_TEMPLATE = """{readme_section}
## Current Best Code (your <search> must exactly match this code)
```python
{code_content}
```

## Start Experiment

Please briefly analyze (3-5 sentences), then submit your first edit.

Analysis points:
- Read the code, identify current bottlenecks (overfitting? underfitting? training instability? simplistic architecture?)
- Propose 1-2 orthogonal optimization hypotheses
- Consider README constraints and choose the most promising direction

Then use `<tool_call>` to invoke `run_training` to start the experiment."""

    # ---------------------------------------------------------------
    # Tool Result Feedback Template
    # ---------------------------------------------------------------
    TOOL_RESULT_TEMPLATE = """<tool_result>
<tool_name>{tool_name}</tool_name>
<status>{status}</status>
{edit_feedback}<output>
{output}
</output>
{metrics_section}
{score_feedback}</tool_result>

{action_hint}
Remaining tool calls: {remaining_calls}"""

    def __init__(self, max_code_length: int = 40000):
        """
        Initialize Prompt Builder
        
        Args:
            max_code_length: Maximum code length (truncated if exceeded)
        """
        self.max_code_length = max_code_length
    
    # ==================================================================
    # Build Initial Messages
    # ==================================================================
    
    def build_agent_messages(
        self,
        code_content: str,
        readme_content: Optional[str] = None,
        max_tool_calls: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Build initial message list for the Agent Loop
        
        Args:
            code_content: Current best code content
            readme_content: README.md content (optional)
            max_tool_calls: Maximum tool call count
            
        Returns:
            OpenAI-compatible message list [{"role": ..., "content": ...}, ...]
        """
        # Truncate code
        truncated_code = code_content[:self.max_code_length]
        if len(code_content) > self.max_code_length:
            truncated_code += "\n# ... (code truncated)"
        
        # System prompt
        system_prompt = self.SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls)
        
        # User initial prompt
        user_prompt = self.USER_TEMPLATE.format(
            readme_section=self._build_readme_section(readme_content),
            code_content=truncated_code,
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    # Backward compatibility alias
    def build_agent_loop_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Backward compatibility alias (deprecated, use build_agent_messages)"""
        return self.build_agent_messages(**kwargs)
    
    # ==================================================================
    # Build Tool Result Feedback
    # ==================================================================
    
    def build_tool_result_message(
        self,
        tool_name: str,
        success: bool,
        output: str,
        metrics: Optional[Dict[str, Any]] = None,
        remaining_calls: int = 0,
        edit_feedback: Optional[str] = None,
        score_feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build tool execution result feedback message
        
        Args:
            tool_name: Tool name
            success: Whether succeeded
            output: Tool output content
            metrics: Extracted metrics dict
            remaining_calls: Remaining tool call count
            edit_feedback: Edit application result feedback
            score_feedback: Score change info dict
            
        Returns:
            Formatted tool result message text
        """
        status = "success" if success else "failed"
        
        # Edit feedback
        edit_feedback_section = ""
        if edit_feedback:
            edit_feedback_section = f"<edit_result>\n{edit_feedback}\n</edit_result>\n"
        
        # Metrics
        metrics_section = ""
        if metrics:
            internal_keys = {'success', 'return_code', 'stdout', 'stderr', 'error', 'experiment_dir'}
            clean_metrics = {k: v for k, v in metrics.items() if k not in internal_keys}
            if clean_metrics:
                metrics_lines = '\n'.join(f"  {k}: {v}" for k, v in clean_metrics.items())
                metrics_section = f"<metrics>\n{metrics_lines}\n</metrics>"
        
        # Score feedback
        score_feedback_section = ""
        if score_feedback:
            code_updated = score_feedback.get("code_updated", False)
            best_score = score_feedback.get("best_score", 0)
            new_score = score_feedback.get("new_score", 0)
            score_delta = score_feedback.get("score_delta", 0)
            higher_is_better = score_feedback.get("higher_is_better", True)
            
            direction_hint = "(higher is better)" if higher_is_better else "(lower is better)"
            
            if code_updated:
                if higher_is_better:
                    delta_str = f"+{score_delta:.4f}"
                else:
                    delta_str = f"{score_delta:.4f}"  # negative delta is improvement for lower-is-better
                score_feedback_section = (
                    f"<score_update>\n"
                    f"  ✅ Score improved! Code has been updated to the new version.\n"
                    f"  This score: {new_score:.4f} | Change: {delta_str} | Current best: {best_score:.4f} {direction_hint}\n"
                    f"  Your next <search> must match the updated new code.\n"
                    f"</score_update>"
                )
            else:
                score_feedback_section = (
                    f"<score_update>\n"
                    f"  ❌ Score did not improve, code has been automatically reverted.\n"
                    f"  This score: {new_score:.4f} | Change: {score_delta:+.4f} | Current best: {best_score:.4f} {direction_hint}\n"
                    f"  Your next <search> must still match the pre-revert code (i.e., the previous best version).\n"
                    f"</score_update>"
                )
        
        # Action hint
        action_hint = self._build_action_hint(
            success=success,
            remaining_calls=remaining_calls,
            score_feedback=score_feedback,
        )
        
        return self.TOOL_RESULT_TEMPLATE.format(
            tool_name=tool_name,
            status=status,
            output=output,
            edit_feedback=edit_feedback_section,
            metrics_section=metrics_section,
            score_feedback=score_feedback_section,
            action_hint=action_hint,
            remaining_calls=remaining_calls,
        )
    
    def _build_action_hint(
        self,
        success: bool,
        remaining_calls: int,
        score_feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate action hint based on training results and remaining calls"""
        if remaining_calls <= 0:
            return "⏰ This is the last chance. Please call FINAL_ANSWER to submit your best code."
        
        if not success:
            return (
                "Training failed. Please carefully read the error message above, fix the bug, and resubmit.\n"
                "Common causes: syntax error, dimension mismatch, OOM, missing import."
            )
        
        code_updated = score_feedback.get("code_updated", False) if score_feedback else False
        
        if remaining_calls == 1:
            if code_updated:
                return (
                    "Score improved! This is the second-to-last chance.\n"
                    "You can: (1) continue optimizing with run_training, or (2) call FINAL_ANSWER to lock in current results."
                )
            else:
                return (
                    "Score did not improve. This is the second-to-last chance, try a different direction.\n"
                    "Note: Code has been reverted, your <search> must match the reverted version."
                )
        
        if code_updated:
            return (
                "Score improved! Analyze the metrics and decide next steps: continue optimizing or submit final code?\n"
                "Note: Current best code has been updated, your <search> must match the new version."
            )
        else:
            return (
                "Score did not improve, code has been reverted. Analyze the cause and try a different optimization direction.\n"
                "Hint: Your <search> must match the reverted code (the unchanged version)."
            )
    
    # ==================================================================
    # Internal Helper Methods
    # ==================================================================
    
    def _build_readme_section(self, readme_content: Optional[str]) -> str:
        """Build README section"""
        if not readme_content:
            return ""
        return f"""
## Algorithm Description (README)
```
{readme_content.strip()}
```

"""
    

