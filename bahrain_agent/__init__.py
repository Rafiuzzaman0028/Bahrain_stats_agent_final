# bahrain_agent/__init__.py
"""
bahrain_agent package

This initializer is intentionally minimal to avoid:
- circular imports,
- "attempted relative import with no known parent package",
- RuntimeWarning about modules being loaded early.

All submodules can still be imported normally:
    from bahrain_agent.agent import BahrainStatsAgent
    from bahrain_agent.data_layer import load_all_data
    ...

Nothing here will interfere with any other file.
"""

__version__ = "1.0.0"

# Expose the most important names for convenience (optional)
__all__ = [
    "agent",
    "data_layer",
    "query_layer",
    "describe_layer",
    "nlu_router",
    "DataRepository",
    "BahrainStatsAgent",
]
