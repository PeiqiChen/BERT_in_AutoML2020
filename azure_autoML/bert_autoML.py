# -*- coding: utf-8 -*-
'''
pip3 install azureml-train-automl 
pip3 uninstall -y numpy
pip3 uninstall -y setuptools
pip3 install setuptools
pip3 install numpy
pip3 install azureml.core
pip3 install azureml.train.automl 
pip3 install azureml.train
'''

import logging
import os
import shutil
import pandas as pd
# import numpy as np
# print(numpy.__file__)
import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.run import Run
from azureml.widgets import RunDetails
from azureml.core.model import Model
from helper import run_inference, get_result_df
from azureml.train.automl import AutoMLConfig
from sklearn.datasets import fetch_20newsgroups


print("This notebook was created using version 1.11.0 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")

subscription_id = os.getenv("SUBSCRIPTION_ID", default="5e031869-5b88-4c8c-aac3-52f689fd05eb")
resource_group = os.getenv("RESOURCE_GROUP", default="re1")
workspace_name = os.getenv("WORKSPACE_NAME", default="wo1")
workspace_region = os.getenv("WORKSPACE_REGION", default="eastus")

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")


# Choose an experiment name.
experiment_name = 'automl-classification-text-dnn'

experiment = Experiment(ws, experiment_name)

output = {}
output['Subscription ID'] = ws.subscription_id
output['Workspace Name'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your cluster.
amlcompute_cluster_name = "text-cluster"

# Verify that cluster does not exist already
# try:
# compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
# print('Found existing cluster, use it.')
# except ComputeTargetException:
compute_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2", # CPU for BiLSTM, such as "STANDARD_D2_V2"
                                                       # To use BERT (this is recommended for best performance), select a GPU such as "STANDARD_NC6"
                                                       # or similar GPU option
                                                       # available in your workspace
                                                       max_nodes = 4)
compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)

############# get data#####################
data_dir = "text-qishi"  # Local directory to store data
blobstore_datadir = data_dir  # Blob store directory to store data in
target_column_name = 'target' # 目标列


def get_data():
    data = pd.read_csv("text/train.csv")
    data = data.loc[:600, ['target', 'comment_text']]
    # print(data)
    data_train = pd.DataFrame(data)
    return data_train

data_train = get_data()

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

train_data_fname = data_dir + '/train.csv'
data_train.to_csv(train_data_fname, index=False)

datastore = ws.get_default_datastore()

####数据存储
datastore.upload(src_dir=data_dir, target_path=blobstore_datadir,
                 overwrite=True)

train_dataset = Dataset.Tabular.from_delimited_files(path = [(datastore, blobstore_datadir + '/train.csv')])


###################### prepare AUTOML run ############################3
automl_settings = {
    "experiment_timeout_minutes": 20,
    "primary_metric": 'accuracy',
    "max_concurrent_iterations": 4,
    "max_cores_per_iteration": -1,
    "enable_dnn": True,
    "enable_early_stopping": True,
    "validation_size": 0.3,
    "verbosity": logging.INFO,
    "enable_voting_ensemble": False,
    "enable_stack_ensemble": False,
}

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             compute_target=compute_target,
                             training_data=train_dataset,
                             label_column_name=target_column_name,
                             **automl_settings
                            )


automl_run = experiment.submit(automl_config, show_output=True)

