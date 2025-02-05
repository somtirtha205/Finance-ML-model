from azureml.core import Environment, Model, Workspace
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

ws = Workspace.from_config()

# Get the registered model
model = Model(ws, name="AR_model")

# Define environment
env = Environment.from_conda_specification(name="ml-env", file_path="environment.yml")

# Define inference configuration
inference_config = InferenceConfig(entry_script="scripts/score.py", environment=env)

# Define deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(
    workspace=ws,
    name="fin-ml-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
)

service.wait_for_deployment(show_output=True)
print(f"Deployment successful! Endpoint: {service.scoring_uri}")
