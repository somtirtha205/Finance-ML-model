import os

from azureml.core import Model, Workspace

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")

ws = Workspace.get(name="WorkspaceML", subscription_id=subscription_id, resource_group="ResourceGroupML")

model = Model.register(workspace=ws, model_name="AR_model", model_path="armodel.pkl", description="Finance ML model")

print(f"Model {model.name} registered successfully!")
