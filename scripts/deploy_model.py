import os

from azureml.core import Environment, Model, Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")

ws = Workspace.get(name="WorkspaceML", subscription_id=subscription_id, resource_group="ResourceGroupML")

# Get the registered model
model = Model(ws, name="AR_model")

# Define environment
env = Environment.from_conda_specification(name="ml-env", file_path="environment.yml")

# Define inference configuration
inference_config = InferenceConfig(entry_script="scripts/score.py", environment=env)

# Define deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=8)

# Deploy the model
service = Model.deploy(
    workspace=ws,
    name="fin-ml-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
)

service.wait_for_deployment(show_output=True)

# Print the logs
print(service.get_logs())

# Print the scoring URI
print(f"Deployment successful! Endpoint: {service.scoring_uri}")
