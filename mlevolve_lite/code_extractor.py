from __future__ import annotations

import re


def extract_python_code(text: str) -> str | None:
    """Extract the last Python code block from markdown-style response."""
    # Prefer ```python ... ```
    matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback to plain ``` ... ```
    matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Some long LLM responses are truncated after opening the final code fence.
    # Salvage the code so the run can still be audited and evaluated.
    match = re.search(r"```(?:python)?\s*(.*)\Z", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
