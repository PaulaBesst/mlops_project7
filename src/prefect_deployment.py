from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from batch_predict import batch_prediction_pipeline

deployment = Deployment.build_from_flow(
    flow=batch_prediction_pipeline,
    name="breast_cancer_prediction_batch",
    schedule=CronSchedule(cron="0 0 * * *"),  # Run daily at midnight
    work_queue_name="ml_queue"
)

if __name__ == "__main__":
    deployment.apply()