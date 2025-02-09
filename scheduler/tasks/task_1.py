from prefect import task

@task
def fetch_data():
    """Simulates fetching data."""
    return {"message": "Data fetched successfully"}
