import glob
import pandas as pd
from azureml.core import Run, Datastore
from pathlib import Path


from azureml.data.dataset_factory import FileDatasetFactory


def read_concat_parquet(files, return_types=False):
    dfs = [pd.read_parquet(f,engine='fastparquet') for f in files]
    df = pd.concat(dfs)
    if return_types:
        dtypes_df = pd.concat([df.dtypes for df in dfs], axis=1).T
        return df, dtypes_df
    else:
        return df


def read_concat_csv(files, return_types=False, sep = ','):
    li = []
    for filename in files:
        df = pd.read_csv(filename, index_col = None, header = 0, sep = sep)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame

def read_concat_excel(files, return_types=False):
    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs)
    if return_types:
        dtypes_df = pd.concat([df.dtypes for df in dfs], axis=1).T
        return df, dtypes_df
    else:
        return df

def recursive_glob_list(folders):
    files = []
    for f in folders:
        files += glob.glob(f"{f}/**/*.parquet", recursive=True)
    return files


def upload_data(datastore_name, files, target_path="."):
    run = Run.get_context()

    datastore = Datastore.get(run.experiment.workspace, datastore_name)
    datastore.upload_files(
        files,
        target_path=target_path,
        overwrite=True,
        show_progress=True,
    )




def upload_directory(datastore_name, local_directory, target_path=""):
    run = Run.get_context()

    # Get the datastore
    datastore = Datastore.get(run.experiment.workspace, datastore_name)

    # Upload the directory using FileDatasetFactory
    FileDatasetFactory.upload_directory(
        src_dir=local_directory, 
        target=(datastore, target_path), 
        overwrite=True,  
        show_progress=True
    )




