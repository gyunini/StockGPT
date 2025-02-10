from prefect import flow
from scheduler.tasks.task_2 import process_data

@flow(name="data_processing_flow")
def flow_2():
    """Main workflow that processes data."""
    result = process_data()
    print(f"Processing Output: {result}")