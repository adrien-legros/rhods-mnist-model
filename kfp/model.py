import argparse
import os

import kfp
from kfp import dsl, kubernetes
from kfp.client import Client

import pre_process
import train
import evaluate

from kubernetes.client.models import V1Volume, V1PersistentVolumeClaimVolumeSource, \
    V1PersistentVolumeClaimSpec, V1ResourceRequirements

@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def mnist_pipeline(model_obc: str = "mnist-model", tag: str = "latest"):
    # Pipeline conf
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    # Pipeline steps
    pre_process_task = pre_process.pre_process(train_path=train_path, test_path=test_path)
    X_train_file = pre_process_task.outputs["X_train_file"]
    y_train_file = pre_process_task.outputs["y_train_file"]
    X_val_file = pre_process_task.outputs["X_val_file"]
    y_val_file = pre_process_task.outputs["y_val_file"]
    X_test_file = pre_process_task.outputs["X_test_file"]
    train_task = train.train(X_train_file=X_train_file, y_train_file=y_train_file, X_val_file=X_val_file, y_val_file=y_val_file, tag=tag)
    model_file = train_task.outputs["model_file"]
    evaluate_task = evaluate.evaluate(X_val_file=X_val_file, y_val_file=y_val_file, X_test_file=X_test_file, model_file=model_file)
    # Pipeline env var
    kubernetes.use_secret_as_env(
        task=pre_process_task,
        secret_name='aws-connection-data',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name='aws-connection-data',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })


if __name__ == '__main__':
    host = "http://ds-pipeline-dspa.mnist:8888"
    parser = argparse.ArgumentParser(
                        prog='Model.py',
                        description='Digit recognition model and pipeline triggering')
    parser.add_argument('-t', '--tag')
    args = parser.parse_args()
    tag = args.tag
    client = Client(host=host)
    run = client.create_run_from_pipeline_func(mnist_pipeline, arguments={"tag": tag})
    print(f"RUN_ID: {run.run_id}")
