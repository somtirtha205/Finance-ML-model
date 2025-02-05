from azureml.core import Model, Workspace

ws = Workspace.get(name="WorkspaceML", resource_group="ResourceGroupML")

model = Model.register(workspace=ws, model_name="AR_model", model_path="./model/ar.pkl", description="Finance ML model")

print(f"Model {model.name} registered successfully!")
