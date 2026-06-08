"""SkySense++ optimizer wrapper constructors.

This module avoids registering a class called
``LearningRateDecayOptimizerConstructor`` because MMSegmentation 1.2.x already
registers that exact name. Registering the same name twice causes errors like:

    KeyError: 'LearningRateDecayOptimizerConstructor is already registered ...'

Use ``SkySenseLayerDecayOptimizerConstructor`` in configs if a project-specific
layer-decay constructor is needed.
"""

from __future__ import annotations

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS

try:
    # Reuse MMSegmentation's official implementation when available. This keeps
    # the behavior compatible with mmsegmentation==1.2.2 and prevents duplicate
    # registration under the official name.
    from mmseg.engine.optimizers.layer_decay_optimizer_constructor import (  # noqa: E501
        LearningRateDecayOptimizerConstructor as _MMSegLearningRateDecayOptimizerConstructor,  # noqa: E501
    )
except Exception as exc:  # pragma: no cover
    _MMSegLearningRateDecayOptimizerConstructor = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


if _MMSegLearningRateDecayOptimizerConstructor is not None:

    @OPTIM_WRAPPER_CONSTRUCTORS.register_module(
        name='SkySenseLayerDecayOptimizerConstructor')
    class SkySenseLayerDecayOptimizerConstructor(
            _MMSegLearningRateDecayOptimizerConstructor):
        """Project-prefixed alias of MMSeg's layer-decay constructor.

        In your config, use::

            optim_wrapper = dict(
                constructor='SkySenseLayerDecayOptimizerConstructor',
                optimizer=dict(type='AdamW', lr=6e-5, weight_decay=0.01),
                paramwise_cfg=dict(...))

        For normal fine-tuning configs that use ``type='OptimWrapper'`` and do
        not specify ``constructor``, this class is not used.
        """

        pass

else:

    @OPTIM_WRAPPER_CONSTRUCTORS.register_module(
        name='SkySenseLayerDecayOptimizerConstructor')
    class SkySenseLayerDecayOptimizerConstructor:  # pragma: no cover
        """Fallback that reports a clear import error if used."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                'Failed to import MMSegmentation official '
                'LearningRateDecayOptimizerConstructor. Original error: '
                f'{_IMPORT_ERROR!r}')
