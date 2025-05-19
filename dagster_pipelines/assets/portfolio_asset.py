"""
This asset is used to generate a position for the SPY ETF.
"""

import os
import pprint
from datetime import datetime

from dagster import DailyPartitionsDefinition, asset, build_op_context
from dotenv import load_dotenv
from vbase import (
    ForwarderCommitmentService,
    VBaseClient,
    VBaseDataset,
    VBaseStringObject,
)

from .portfolio_producer import produce_sector_portfolios, get_run_logger

# The name of the portfolio set (collection).
PORTFOLIO_NAME = "SectorPortfolios"

# Define a daily partition for portfolio rebalancing.
partitions_def = DailyPartitionsDefinition(start_date="2025-01-01")

# The vBase forwarder URL for making commitments via the vBase forwarder.
VBASE_FORWARDER_URL = "https://api.vbase.com/forwarder/"


@asset(partitions_def=partitions_def)
def sector_portfolios(context):
    """
    This asset generates market-neutral long and short portfolios for 11 sector ETFs.
    """
    load_dotenv()

    required_settings = [
        "VBASE_API_KEY",
        "VBASE_COMMITMENT_SERVICE_PRIVATE_KEY",
    ]
    for setting in required_settings:
        if setting not in os.environ:
            raise ValueError(f"{setting} environment variable is not set.")

    partition_date = context.asset_partition_key_for_output()
    context.log.info("Starting sector portfolios generation for %s", partition_date)

    # Initialize per-run file logger
    file_logger = get_run_logger(partition_date)
    file_logger.info("Starting sector portfolios generation for %s", partition_date)

    try:
        df_portfolios = produce_sector_portfolios(partition_date, logger=file_logger, half_life=30)
        context.log.info(f"{partition_date}: portfolios_df = \n{df_portfolios}")
        file_logger.info("Generated portfolios:\n%s", df_portfolios)

        # Save portfolios locally
        filename = f"sector_portfolios--{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        path = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        df_portfolios.to_csv(path, index=False)
        file_logger.info(f"Saved portfolios to {path}")
        context.log.info(f"Saved portfolios to {path}")

        # Convert to CSV string
        body = df_portfolios.to_csv(index=False)
        context.log.info(f"{partition_date}: CSV body = \n{body}")
        file_logger.info("Portfolios CSV content:\n%s", body)

        # vBase stamping
        vbc = VBaseClient(
            ForwarderCommitmentService(
                forwarder_url=VBASE_FORWARDER_URL,
                api_key=os.environ["VBASE_API_KEY"],
                private_key=os.environ["VBASE_COMMITMENT_SERVICE_PRIVATE_KEY"],
            )
        )
        ds = VBaseDataset(vbc, PORTFOLIO_NAME, VBaseStringObject)
        receipt = ds.add_record(body)
        context.log.info(f"ds.add_record() receipt:\n{pprint.pformat(receipt)}")
        file_logger.info("vBase receipt:\n%s", pprint.pformat(receipt))

    except Exception as e:
        context.log.error(f"Error generating sector portfolios: {e}")
        file_logger.exception("Error during sector portfolios generation")
        raise


def debug_portfolio(date_str: str = None) -> None:
    """
    Materialize the portfolio asset for a specific date or today's date.

    Args:
        date_str: Optional date string in YYYY-MM-DD format. If None, uses today's date.
    """
    partition_date = date_str or datetime.now().strftime("%Y-%m-%d")
    context = build_op_context(partition_key=partition_date)
    sector_portfolios(context)


if __name__ == "__main__":
    # Run for today's date.
    debug_portfolio()

#     # Run for a specific past date.
#     debug_portfolio("2025-04-04")
