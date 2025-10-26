import os
from pathlib import Path
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Workspace, Environment, Datastore
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
from azureml.data.datapath import DataPath, DataPathComputeBinding
from dotenv import load_dotenv, dotenv_values


NAME = "Debug-STAGE-fine-tuning-labse"
DESCRIPTION = "fine tuning model for matching"
COMPUTE_CLUSTER = "test-gpu"
allow_reuse = False

# 1. Load env vars
load_dotenv()
loaded_vars = dotenv_values()

# 2. Get Creds
auth = InteractiveLoginAuthentication()

# 3. Get Workspace
workspace = Workspace.get(
        name=os.environ["WORKSPACE_NAME"],
        resource_group=os.environ["RESOURCE_GROUP"],
        subscription_id=os.environ["SUBSCRIPTION_ID"],
        auth=auth)

# 4. Create RunConfig
run_config = RunConfiguration()
env = Environment.from_conda_specification(name="matching_env", 
                                           file_path="conda_dependencies.yml")

compute_binding = DataPathComputeBinding(mode="mount")

# 5. Migrate env vars to compute cluster
run_config.environment = env
run_config.environment.environment_variables.update(loaded_vars)

# DATA INPUTS
#   datastores
project_ds = Datastore.get(workspace, "mp_ds_matching_stage")


#   data paths
root = DataPath(datastore=project_ds, path_on_datastore="").create_data_reference(
    data_reference_name="root", datapath_compute_binding=compute_binding)

# Pipeline Steps
matcher_step = PythonScriptStep(
    name="fine_tuning",
    script_name = "src/main.py",
    compute_target=COMPUTE_CLUSTER,
    source_directory="./",
    inputs=[root],
    outputs = [],
    arguments= [
        "--root", root
    ],
    runconfig=run_config,
    allow_reuse=allow_reuse,
)

pipeline = Pipeline(workspace=workspace, 
                    steps=[matcher_step, ],
                    description=DESCRIPTION)
pipeline.validate()
pipeline.publish(name=NAME, description=DESCRIPTION)

# submit
_submit = input("Submit Pipeline? (y/n): ")
if _submit.lower().strip() == "y":
    pipeline.submit(experiment_name=NAME)




