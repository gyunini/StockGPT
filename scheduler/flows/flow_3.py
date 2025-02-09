from prefect import flow
from scheduler.tasks.task_3 import task_3

@flow(name="load_stooq")
def flow_3():
    """Fetches stooq data"""
    result = task_3()
    print(f"Processing Output: {result}")