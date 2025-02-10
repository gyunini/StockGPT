from prefect import task
from prefect.logging import get_run_logger

from src.stockgpt.load_stooq import load_stooq

@task
def task_3():
    """Simulates fetching data."""
    logger = get_run_logger()
    load_stooq(logger)
    return {"message": "Coinbase data loaded successfully"}