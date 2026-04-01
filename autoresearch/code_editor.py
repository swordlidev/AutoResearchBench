"""
Code Editing Engine

Responsible for parsing LLM output search/replace edit blocks and applying them to code.
Uses an Aider-like search/replace block editing protocol.
"""

import re
import logging
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EditBlock:
    """Single edit block"""
    search: str   # Original code snippet to find
    replace: str  # New code snippet to replace with


@dataclass
class EditResult:
    """Edit application result"""
    applied_code: str                          # Complete code after applying edits
    applied_count: int = 0                     # Number of successfully applied edit blocks
    total_count: int = 0                       # Total number of edit blocks
    errors: List[str] = field(default_factory=list)  # List of error messages for failed applications

    @property
    def all_applied(self) -> bool:
        """Whether all edit blocks were successfully applied"""
        return self.applied_count == self.total_count

    @property
    def has_errors(self) -> bool:
        """Whether any edit blocks failed to apply"""
        return len(self.errors) > 0


class CodeEditor:
    """
    Code Editing Engine

    Core methods:
    - parse_edits(raw_content) : Extract EditBlock list from XML format
    - apply_edits(code, edits) : Apply edit blocks to code sequentially

    Fault tolerance strategy (two levels):
    1. Exact match: Direct text search in code
    2. Fuzzy match: Normalize whitespace then retry
    """

    # ----------------------------------------------------------------
    # Parsing
    # ----------------------------------------------------------------

    @staticmethod
    def parse_edits(raw_content: str) -> List[EditBlock]:
        """
        Parse EditBlock list from XML format content

        Supported format:
            <edits>
            <edit>
            <search>
            code to find
            </search>
            <replace>
            replacement code
            </replace>
            </edit>
            ...
            </edits>

        Also supports <edit> blocks without outer <edits> wrapper.

        Args:
            raw_content: Raw text output from LLM

        Returns:
            List of EditBlock (may be empty)
        """
        edits: List[EditBlock] = []

        # Match all <edit>...</edit> blocks
        edit_pattern = re.compile(
            r'<edit>\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>\s*</edit>',
            re.DOTALL,
        )

        for match in edit_pattern.finditer(raw_content):
            search_text = CodeEditor._clean_content(match.group(1))
            replace_text = CodeEditor._clean_content(match.group(2))
            edits.append(EditBlock(search=search_text, replace=replace_text))

        if edits:
            logger.info(f"Parsed {len(edits)} edit blocks")
        else:
            logger.warning("No <edit> blocks found in LLM output")

        return edits

    # ----------------------------------------------------------------
    # Application
    # ----------------------------------------------------------------

    @staticmethod
    def apply_edits(code: str, edits: List[EditBlock]) -> EditResult:
        """
        Apply edit blocks to code sequentially

        Application strategy:
        1. Exact match: str.find + replace (first occurrence only)
        2. Fuzzy match: Normalize whitespace then retry
        3. All failed: Record error, skip this edit

        Args:
            code:  Current code text
            edits: List of EditBlock to apply

        Returns:
            EditResult containing applied code, success count, error messages
        """
        result = EditResult(
            applied_code=code,
            applied_count=0,
            total_count=len(edits),
        )

        current_code = code

        for idx, edit in enumerate(edits, start=1):
            applied, new_code = CodeEditor._apply_single_edit(current_code, edit)

            if applied:
                current_code = new_code
                result.applied_count += 1
                logger.debug(f"Edit block #{idx} applied successfully")
            else:
                # Truncate search snippet to first 80 chars for logging
                snippet = edit.search[:80].replace('\n', '\\n')
                error_msg = f"Edit block #{idx} match failed, search snippet: \"{snippet}...\""
                result.errors.append(error_msg)
                logger.warning(error_msg)

        result.applied_code = current_code
        return result

    # ----------------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------------

    @staticmethod
    def _apply_single_edit(code: str, edit: EditBlock) -> tuple:
        """
        Try to apply a single edit block

        Returns:
            (success: bool, new_code: str)
        """
        search = edit.search
        replace = edit.replace

        # --- Special case: empty search → append to end of file ---
        if not search.strip():
            new_code = code.rstrip('\n') + '\n\n' + replace.strip() + '\n'
            return True, new_code

        # --- 1. Exact match ---
        if search in code:
            new_code = code.replace(search, replace, 1)
            return True, new_code

        # --- 2. Fuzzy match (strip leading/trailing whitespace + normalize blank lines) ---
        normalized_search = CodeEditor._normalize_whitespace(search)
        normalized_code = CodeEditor._normalize_whitespace(code)

        if normalized_search in normalized_code:
            # Found normalized match, locate corresponding region in original code
            new_code = CodeEditor._fuzzy_replace(code, search, replace)
            if new_code is not None:
                return True, new_code

        # --- All failed ---
        return False, code

    @staticmethod
    def _fuzzy_replace(code: str, search: str, replace: str) -> str | None:
        """
        Fuzzy replace: normalize code lines then match, locate corresponding
        region in original code and replace

        Returns:
            Replaced code, or None (match failed)
        """
        code_lines = code.split('\n')
        search_lines = [line.strip() for line in search.strip().split('\n') if line.strip()]

        if not search_lines:
            return None

        # Sliding window match across code lines
        for start_idx in range(len(code_lines)):
            match_len = CodeEditor._try_match_lines(code_lines, start_idx, search_lines)
            if match_len > 0:
                # Found match region [start_idx, start_idx + match_len)
                before = '\n'.join(code_lines[:start_idx])
                after = '\n'.join(code_lines[start_idx + match_len:])
                parts = [p for p in [before, replace.strip(), after] if p]
                return '\n'.join(parts) + ('\n' if code.endswith('\n') else '')

        return None

    @staticmethod
    def _try_match_lines(code_lines: list, start: int, search_lines: list) -> int:
        """
        Starting from code_lines[start], try to match search_lines line by line
        (ignoring blank lines and leading/trailing whitespace)

        Returns:
            Number of code lines consumed by match (> 0 means success), 0 means failure
        """
        si = 0  # search_lines pointer
        ci = start  # code_lines pointer

        while si < len(search_lines) and ci < len(code_lines):
            code_stripped = code_lines[ci].strip()

            # Skip blank lines in code
            if not code_stripped:
                ci += 1
                continue

            if code_stripped == search_lines[si]:
                si += 1
                ci += 1
            else:
                return 0

        if si == len(search_lines):
            return ci - start

        return 0

    @staticmethod
    def _clean_content(text: str) -> str:
        """
        Clean content extracted from XML tags:
        - Remove leading and trailing blank lines
        - Preserve internal indentation
        """
        # Remove leading/trailing blank lines but preserve indentation
        lines = text.split('\n')

        # Remove leading blank lines
        while lines and not lines[0].strip():
            lines.pop(0)

        # Remove trailing blank lines
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace:
        - Strip leading/trailing whitespace from each line
        - Merge consecutive blank lines into a single blank line
        - Remove leading/trailing blank lines
        """
        lines = [line.strip() for line in text.split('\n')]

        # Merge consecutive blank lines
        normalized = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    normalized.append('')
                prev_empty = True
            else:
                normalized.append(line)
                prev_empty = False

        # Remove leading/trailing blank lines
        while normalized and not normalized[0]:
            normalized.pop(0)
        while normalized and not normalized[-1]:
            normalized.pop()

        return '\n'.join(normalized)
