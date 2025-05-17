"""
The package for the Dagster Cloud pipeline.
"""

# Dagster Cloud is importing your package like: import dagster_pipelines
# and expects to find dagster_pipelines.defs
# This re-exports your defs object from definitions.py at the package level,
# which Dagster Cloud requires.
from .definitions import defs
