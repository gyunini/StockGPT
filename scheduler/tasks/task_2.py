from prefect import task

@task
def process_data():
    """Simulates fetching data."""
    return {"message": "Data fetched successfully"}
