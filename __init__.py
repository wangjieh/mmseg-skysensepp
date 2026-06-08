"""Optimizer constructors for the SkySense++ project.

The class name intentionally uses a project-specific prefix to avoid
registration conflicts with MMSegmentation's built-in
``LearningRateDecayOptimizerConstructor``.
"""

from .layer_decay_optimizer_constructor import SkySenseLayerDecayOptimizerConstructor

__all__ = ['SkySenseLayerDecayOptimizerConstructor']
