"""
The definitions for the Dagster Cloud pipeline.
"""

from dagster import Definitions

from dagster_pipelines.assets.portfolio_asset import sector_portfolios
from dagster_pipelines.schedules.portfolio_schedule import portfolio_schedule

defs = Definitions(
    assets=[sector_portfolios],
    schedules=[portfolio_schedule],
)
