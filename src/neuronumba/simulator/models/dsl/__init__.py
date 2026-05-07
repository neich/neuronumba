"""
Declarative model DSL for neuronumba.

A `ModelSpec` describes a whole-brain model in terms of state variables,
coupling variables, parameters, and equations. `build_model(spec)` compiles it
into a `Model` subclass that plugs into `Simulator` exactly like a hand-written
model.

Public API:
    ModelSpec, StateVar, CouplingVar, Parameter
    build_model, dump_generated, get_source_file
    get_cache_dir, cleanup_cache
"""
from .builder import build_model, dump_generated, get_source_file
from .materialize import cleanup_cache, get_cache_dir
from .spec import CouplingVar, ModelSpec, Parameter, StateVar

__all__ = [
    "CouplingVar",
    "ModelSpec",
    "Parameter",
    "StateVar",
    "build_model",
    "cleanup_cache",
    "dump_generated",
    "get_cache_dir",
    "get_source_file",
]
