
import argparse
import kfp
import kfp.components as comp
import kfp_tekton
import kubernetes
import os

import pre_process
import train
import evaluate

from kfp.dsl import data_passing_methods
from kubernetes.client.models import V1Volume, V1PersistentVolumeClaimVolumeSource, \
    V1PersistentVolumeClaimSpec, V1ResourceRequirements

pre_process_op = kfp.components.create_component_from_func(
    pre_process.pre_process,
    base_image="quay.io/alegros/runtime-image:rhods-mnist-cpu",
)

train_op = kfp.components.create_component_from_func(
    train.train,
    base_image="quay.io/alegros/runtime-image:rhods-mnist-cpu",
    packages_to_install=["mlflow"],
)

evalue_op = kfp.components.create_component_from_func(
    evaluate.evaluate,
    base_image="quay.io/alegros/runtime-image:rhods-mnist-cpu",
)

@kfp.dsl.pipeline(
    name="Mnist Pipeline",
)
def mnist_pipeline(model_obc: str = "mnist-model", tag: str = "latest"):
    accesskey = kubernetes.client.V1EnvVar(
        name="AWS_ACCESS_KEY_ID",
        value_from=kubernetes.client.V1EnvVarSource(
            secret_key_ref=kubernetes.client.V1SecretKeySelector(
                name="mlpipeline-minio-artifact", key="accesskey"
            )
        ),
    )
    host = kubernetes.client.V1EnvVar(
        name="AWS_S3_HOST",
        value_from=kubernetes.client.V1EnvVarSource(
            secret_key_ref=kubernetes.client.V1SecretKeySelector(
                name="mlpipeline-minio-artifact", key="host"
            )
        ),
    )
    port = kubernetes.client.V1EnvVar(
        name="AWS_S3_PORT",
        value_from=kubernetes.client.V1EnvVarSource(
            secret_key_ref=kubernetes.client.V1SecretKeySelector(
                name="mlpipeline-minio-artifact", key="port"
            )
        ),
    )
    secretkey = kubernetes.client.V1EnvVar(
        name="AWS_SECRET_ACCESS_KEY",
        value_from=kubernetes.client.V1EnvVarSource(
            secret_key_ref=kubernetes.client.V1SecretKeySelector(
                name="mlpipeline-minio-artifact", key="secretkey"
            )
        ),
    )
    bucket = kubernetes.client.V1EnvVar(name="AWS_S3_BUCKET", value="rhods")
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    pre_process_task = pre_process_op(
        train_path = train_path,
        test_path = test_path
    ).add_env_variable(accesskey).add_env_variable(host).add_env_variable(port).add_env_variable(secretkey).add_env_variable(bucket)
    train_task = train_op(
        pre_process_task.outputs["X_train"],
        pre_process_task.outputs["y_train"],
        pre_process_task.outputs["X_val"],
        pre_process_task.outputs["y_val"],
        pre_process_task.outputs["X_test"],
        tag = tag
    ).add_env_variable(accesskey).add_env_variable(host).add_env_variable(port).add_env_variable(secretkey).add_env_variable(bucket)
    evaluate_task = evalue_op(
        pre_process_task.outputs["X_val"],
        pre_process_task.outputs["y_val"],
        pre_process_task.outputs["X_test"],
        train_task.outputs["model"]
    ).add_env_variable(accesskey).add_env_variable(host).add_env_variable(port).add_env_variable(secretkey).add_env_variable(bucket)


if __name__ == '__main__':
    host = "http://ds-pipeline-pipelines-definition.mnist:8888"
    parser = argparse.ArgumentParser(
                        prog='Model.py',
                        description='Digit recognition model and pipeline triggering')
    parser.add_argument('-t', '--tag')
    args = parser.parse_args()
    tag = args.tag
    volume_based_data_passing_method = data_passing_methods.KubernetesVolume(
        volume=V1Volume(
            name='data',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                claim_name='ml-pipeline'),
        )
    )
    pipeline_conf = kfp.dsl.PipelineConf()
    pipeline_conf.data_passing_method = volume_based_data_passing_method
    client = kfp_tekton.TektonClient(host=host)
    result = client.create_run_from_pipeline_func(
        mnist_pipeline, arguments={"tag": tag}, experiment_name="pre-prod", pipeline_conf=pipeline_conf
    )
    print(f"RUN_ID: {result.run_id}")