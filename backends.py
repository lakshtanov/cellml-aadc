"""
AD-backend abstraction for the fair AADC-vs-CasADI comparison.

The model math (cvs3_model.py, and the Lotka model in compare_fair.py) is written
once against this tiny interface, then run through EITHER backend. The only thing
that differs between an AADC run and a CasADI run is which backend object is
passed in — same source, same integrator, same conditionals.

Interface:
  B.iif(cond, a, b)   conditional select        (aadc.iif      / ca.if_else)
  B.cos(x)            cosine                     (aadc.math.cos / ca.cos)
  B.floor(x)          floor                      (passive value / ca.floor)
  B.const(v)          wrap a scalar constant     (aadc.idouble  / identity)
"""
import math
from types import SimpleNamespace


def get_aadc_backend():
    import aadc
    return SimpleNamespace(
        name="aadc",
        iif=aadc.iif,
        cos=aadc.math.cos,
        # AADC uses the passive (numeric) value of floor at record time. The floor
        # argument (S_HEART / CHI_*) advances at fixed rates independent of the
        # calibration parameters, so this does not affect the parameter gradient.
        floor=lambda x: math.floor(float(x)),
        const=aadc.idouble,
    )


def get_numeric_backend():
    """Plain-float backend — for independent finite-difference reference values."""
    def _iif(cond, a, b):
        return a if cond else b
    return SimpleNamespace(
        name="numeric",
        iif=_iif,
        cos=math.cos,
        floor=lambda x: math.floor(float(x)),
        const=float,
    )


def get_casadi_backend():
    import casadi as ca
    return SimpleNamespace(
        name="casadi",
        iif=ca.if_else,          # <-- the analogue the original comparison never used
        cos=ca.cos,
        floor=ca.floor,          # symbolic, derivative 0 a.e. — matches AADC intent
        const=lambda v: v,       # CasADI promotes Python floats automatically
    )
