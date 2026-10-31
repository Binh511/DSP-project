"""Deterministic source facts; imports no ML libraries and never infers schedules."""
from .facts import extract_facts, resource_warnings

__all__ = ["extract_facts", "resource_warnings"]
