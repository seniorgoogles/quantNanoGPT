from brevitas.inject.enum import RestrictValueType, ScalingImplType
from brevitas.quant.fixed_point import BiasQuantSolver, MaxStatsScaling, NarrowIntQuant, WeightQuantSolver
from brevitas.inject import ExtendedInjector
from brevitas.core.function_wrapper.ops_ste import CeilSte

from quantizer_config import DYNAMIC_QUANTIZER_BIT_WIDTH

class BiasQuantizer(BiasQuantSolver):
    requires_input_bit_width = False
    requires_input_scale = False

class PerTensorPoTScaling16bit(ExtendedInjector):
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    bit_width = 16
    restrict_value_float_to_int_impl = CeilSte

class IntDynamicWeightPerTensorFixedPoint(NarrowIntQuant,
                                 MaxStatsScaling,
                                 PerTensorPoTScaling16bit,
                                 WeightQuantSolver):
    bit_width = DYNAMIC_QUANTIZER_BIT_WIDTH
