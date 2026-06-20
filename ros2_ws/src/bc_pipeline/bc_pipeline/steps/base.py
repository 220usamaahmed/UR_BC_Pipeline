#!/usr/bin/env python3
"""
Step base class + type registry.

ADDING A NEW STEP TYPE
----------------------
1. Create a module under bc_pipeline/steps/.
2. Subclass Step, decorate it with @register("YourType").
3. Import the module in steps/__init__.py so the decorator runs.

That's it — the factory (build_step) picks it up by its 'type' string from the
YAML.  No central if/elif to edit.

Each Step:
  • parses + validates its own args in validate() (called at construction, so
    a bad config fails before any motion), and
  • performs its behaviour in execute(), returning True on success.

The shared Context gives every step the MoveIt clients, checkpoint table, and
node/logger it needs (see context.py).
"""

from abc import ABC, abstractmethod


STEP_REGISTRY: dict[str, type] = {}


class StepConfigError(Exception):
    """Raised when a step's YAML arguments are missing or invalid."""


def register(type_name: str):
    """Class decorator: register a Step subclass under its YAML 'type' string."""
    def decorator(cls):
        if type_name in STEP_REGISTRY:
            raise ValueError(f"Duplicate step type registered: '{type_name}'")
        cls.type_name = type_name
        STEP_REGISTRY[type_name] = cls
        return cls
    return decorator


class Step(ABC):
    #: Set by @register; the YAML 'type' string this class handles.
    type_name: str = '<unregistered>'

    def __init__(self, cfg: dict, ctx):
        self.cfg = cfg
        self.ctx = ctx
        self.validate()

    def validate(self):
        """Parse + validate args from self.cfg. Override in subclasses."""

    @abstractmethod
    def execute(self) -> bool:
        """Perform the step. Return True on success, False on failure."""

    @property
    def label(self) -> str:
        return self.type_name

    def _require(self, key: str):
        """Return cfg[key] or raise a StepConfigError naming the step."""
        if key not in self.cfg:
            raise StepConfigError(
                f"{self.type_name} step missing required key '{key}'."
            )
        return self.cfg[key]


def build_step(entry: dict, ctx) -> Step:
    """Factory: construct the Step subclass named by entry['type']."""
    type_name = entry['type']
    if type_name not in STEP_REGISTRY:
        raise StepConfigError(
            f"Unknown step type '{type_name}'. "
            f"Known types: {sorted(STEP_REGISTRY)}."
        )
    return STEP_REGISTRY[type_name](entry, ctx)
