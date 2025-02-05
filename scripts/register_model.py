import os

from azureml.core import Model, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
tenant_id = os.getenv("AZURE_Tenant_ID")
sp_id = os.getenv("AZURE_SP_ID")
sp_secret = os.getenv("AZURE_SP_SECRET")

sp = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=sp_id,
    subscription_id=subscription_id,
    service_principal_password=sp_secret,
)

ws = Workspace.get(name="WorkspaceML", resource_group="ResourceGroupML", subscription_id=subscription_id, auth=sp)

model = Model.register(workspace=ws, model_name="AR_model", model_path="./model/ar.pkl", description="Finance ML model")

print(f"Model {model.name} registered successfully!")
