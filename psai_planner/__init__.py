"""Optional local LLM planning with source-span validation.

The public API imports only the Python standard library. Model libraries are
loaded inside ``plan_project`` only after input and context checks succeed.
"""

from .planner import (PlannerValidationError, build_catalog, build_prompt,
                      plan_project, validate_response)
from .context import build_review_context, validate_review_context

__all__ = ["PlannerValidationError", "build_catalog", "build_prompt",
           "plan_project", "validate_response", "build_review_context", "validate_review_context"]
