from azureml.core import Model, Workspace

ws = Workspace.from_config()

model = Model.register(workspace=ws, model_name="AR_model", model_path="./model/ar.pkl", description="Finance ML model")

print(f"Model {model.name} registered successfully!")
