#!/usr/bin/env python3
"""
Checkpoint — joint-space move to a named checkpoint.

The checkpoint name must exist in the config's 'checkpoints' table; an unknown
name is a hard error at load time (fail fast, before any motion).  Joint values
live only in the table so they stay reusable across many steps.
"""

from .base import Step, StepConfigError, register


@register('Checkpoint')
class Checkpoint(Step):
    def validate(self):
        self.name = self._require('checkpoint')
        if self.name not in self.ctx.checkpoints:
            raise StepConfigError(
                f"Checkpoint '{self.name}' is not defined in the checkpoints "
                f"table. Defined: {sorted(self.ctx.checkpoints)}."
            )
        self.angles = self.ctx.checkpoints[self.name]

    @property
    def label(self) -> str:
        return f"Checkpoint '{self.name}'"

    def execute(self) -> bool:
        self.ctx.logger.info(f"Moving to checkpoint '{self.name}' → {self.angles}")
        return self.ctx.plan_and_execute_joints(self.angles)
