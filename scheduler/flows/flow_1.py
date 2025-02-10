from prefect import flow
from scheduler.tasks.task_1 import fetch_data

@flow(name="data_pipeline_flow")
def flow_1():
    """Main workflow that fetches data."""
    result = fetch_data()
    print(f"Flow Output: {result}")