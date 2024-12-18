import argparse
import kfp 

from kfp import dsl, Client
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact, ClassificationMetrics
from datetime import datetime

@dsl.component(
    base_image="quay.io/alegros/mnist-runtime-image:latest",
    packages_to_install=[],
)
def load_datasets(train_ds: Output[Dataset], test_ds: Output[Dataset]):
    import pandas as pd

    train_csv = 'https://minio-s3-alegros-loan-prediction.apps.prod.rhoai.rh-aiservices-bu.com/rhods/data/train.csv'
    df = pd.read_csv(train_csv)
    with open(train_ds.path, 'w') as f:
        df.to_csv(f, index=False)
    test_csv = 'https://minio-s3-alegros-loan-prediction.apps.prod.rhoai.rh-aiservices-bu.com/rhods/data/test.csv'
    df = pd.read_csv(test_csv)
    with open(test_ds.path, 'w') as f:
        df.to_csv(f, index=False)
    train_ds.metadata["version"] = "1.0"
    test_ds.metadata["version"] = "1.0"
    train_ds.metadata["foo"] = "bar"
    test_ds.metadata["foo"] = "bar"

@dsl.component(
    base_image="quay.io/alegros/mnist-runtime-image:latest",
    packages_to_install=[],
)
def pre_process(
        train_ds: Input[Dataset],
        test_ds: Input[Dataset],
        X_train_out: Output[Artifact],
        y_train_out: Output[Artifact],
        X_val_out: Output[Artifact],
        y_val_out: Output[Artifact]
):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from tensorflow import keras


    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import OneHotEncoder

    import pickle

    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    df_train = pd.read_csv(train_ds.path)
    df_test = pd.read_csv(test_ds.path)

    X = df_train.iloc[:,1:]
    y = df_train.iloc[:, 0]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=15)

    class ReshapeFunc(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            X = X.reshape((-1,28,28,1))
            return X
    features_pipeline = Pipeline(steps=[
        ('Normalize', MinMaxScaler()),
        ('Reshape', ReshapeFunc())
    ])
    target_pipeline = Pipeline(steps=[
        ('OneHot', OneHotEncoder())
    ])

    X_train = features_pipeline.fit_transform(X_train)
    y_train = target_pipeline.fit_transform(y_train.values.reshape(-1,1))
    y_train = y_train.toarray()
    X_val = features_pipeline.fit_transform(X_val)
    y_val = target_pipeline.fit_transform(y_val.values.reshape(-1, 1))
    y_val = y_val.toarray()

    save_pickle(X_train_out.path, X_train)
    save_pickle(y_train_out.path, y_train)
    save_pickle(X_val_out.path, X_val)
    save_pickle(y_val_out.path, y_val)

@dsl.component(
    base_image="quay.io/alegros/mnist-runtime-image:latest",
    packages_to_install=[],
)
def train(
        X_train_out: Input[Artifact],
        y_train_out: Input[Artifact],
        X_val_out: Input[Artifact],
        y_val_out: Input[Artifact],
        model_onnx_out: Output[Model],
        model_tf_out: Output[Model],
        tag: str):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    import subprocess
    import pickle
    import sys
    import datetime

    # Pickle helpers
    def save_pickle(object_file, target_object):
        with open(object_file, "wb") as f:
            pickle.dump(target_object, f)

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)
        return target_object            
    # Load pre processed data
    X_train = load_pickle(X_train_out.path)
    y_train = load_pickle(y_train_out.path)
    X_val = load_pickle(X_val_out.path)
    y_val = load_pickle(y_val_out.path)
    # Model architecture
    def build_model():
        inp = keras.Input(shape=(28,28,1), name="input_1")
        x = keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1),padding='SAME', 
                                activation='relu')(inp)
        x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='SAME', activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        output = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inp, outputs=output)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            metrics=['accuracy'])
        return model, inp, output
    # Model training
    model, inp, out = build_model()
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=10, 
                                                            min_delta=0.005, restore_best_weights=True),
                            keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 3)])
    # Metadata declaration
    metadata = {
        "framework": tf.__version__,
        "python_version": sys.version,
        "creation_date": str(datetime.datetime.now()),
        "tag": tag,
        "dummy": "foo"
    }
    model_tf_out.metadata = metadata
    model_onnx_out.metadata = metadata
    # Save model
    tf.saved_model.save(model, model_tf_out.path)
    # Convert and export model
    cmd = 'python -m tf2onnx.convert --saved-model ' + model_tf_out.path + ' --output ' + model_onnx_out.path + ' --opset 13'
    proc = subprocess.run(cmd.split(), capture_output=True)
    print(proc.returncode)
    print(proc.stdout.decode('ascii'))
    print(proc.stderr.decode('ascii'))
    

@dsl.component(
    base_image="quay.io/alegros/mnist-runtime-image:latest",
    packages_to_install=[],
)
def evaluate(
    X_val_out: Input[Artifact],
    y_val_out: Input[Artifact],
    model_onnx_out: Input[Model],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
):
    import numpy as np
    import pandas as pd
    import pickle
    import tensorflow as tf
    import onnxruntime as ort
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, confusion_matrix
    from keras import backend as K

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)
        return target_object
    

    X_val = load_pickle(X_val_out.path)
    y_val = load_pickle(y_val_out.path)
    
    ort_sess = ort.InferenceSession(model_onnx_out.path)
    outputs = ort_sess.run(None, {'inputs': X_val.astype(np.float32)})
    
    y_val_pred = np.argmax(outputs[0], axis=1)
    y_val_true = np.argmax(y_val,axis=1)
    
    # Precision (using keras backend)
    def precision_metric(y_true, y_pred):
        threshold = 0.5  # Training threshold 0.5
        y_pred_y = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())

        true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
        false_negatives = K.sum(K.clip(y_true * (1-y_pred), 0, 1))
        false_positives = K.sum(K.clip((1-y_true) * y_pred, 0, 1))
        true_negatives = K.sum(K.clip((1 - y_true) * (1-y_pred), 0, 1))

        precision = true_positives / (true_positives + false_positives + K.epsilon())
        return precision

    # Recall (using keras backend)
    def recall_metric(y_true, y_pred):
        threshold = 0.5 #Training threshold 0.5
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())

        true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
        false_negatives = K.sum(K.clip(y_true * (1-y_pred), 0, 1))
        false_positives = K.sum(K.clip((1-y_true) * y_pred, 0, 1))
        true_negatives = K.sum(K.clip((1 - y_true) * (1-y_pred), 0, 1))

        recall = true_positives / (true_positives + false_negatives + K.epsilon())
        return recall

    # F1-score (using keras backend)
    def f1_metric(y_true, y_pred):
        precision = precision_metric(y_true, y_pred)
        recall = recall_metric(y_true, y_pred)
        f1 = 2 * ((precision * recall) / (recall+precision+K.epsilon()))
        return f1
    acc = accuracy_score(y_val_true, y_val_pred)
    metrics.log_metric("accuracy", acc * 100)
    f1_macro = f1_score(y_val_true, y_val_pred, average="macro")
    metrics.log_metric("f1_score", f1_macro)
    rec = recall_score(y_val_true, y_val_pred, average="macro")
    metrics.log_metric("recall", rec)
    prec = precision_score(y_val_true, y_val_pred, average="macro")
    metrics.log_metric("precision", prec)
    cm = confusion_matrix(np.argmax(y_val,axis=1), y_val_pred)
    classification_metrics.log_confusion_matrix([f"{i}" for i in range(0, 10)], cm.tolist())
    fpr, tpr, thresholds = roc_curve(
        y_true=y_val_true, y_score=y_val_pred, pos_label=True)
    classification_metrics.log_roc_curve(fpr[1:], tpr[1:], thresholds[1:]) # [1:] Workaround to slip Infinity value
    print(f'accuracy_score: {acc}')
    print(f'f1_score_macro: {f1_macro}')
    print(f'precision_score: {prec}')
    print(f'recall_score: {rec}')

@dsl.component(
    base_image="quay.io/alegros/mnist-runtime-image:latest",
    packages_to_install=[],
)
def register_model(tag: str, model: Input[Model], metrics: Input[Metrics], classification_metrics: Input[ClassificationMetrics], app_domain: str, user_token: str):
    from model_registry import ModelRegistry
    from datetime import datetime
    # Register model
    model_path=model.path
    version=datetime.now().strftime('%y%m%d%H%M')
    model_name="mnist"
    author_name="ds@redhat.com"
    model_regitry_endpoint = f"https://model-registry-rest.{app_domain}"
    registry = ModelRegistry(server_address=model_regitry_endpoint, port=443, author=author_name, is_secure=False, user_token=user_token)
    registered_model_name = model_name
    metadata = {
        # "metrics": metrics,
        "license": "apache-2.0",
        "commit": tag,
        "classification_metrics": classification_metrics.path,
        "metrics": metrics.path
    }
    rm = registry.register_model(
        registered_model_name,
        model_path,
        model_format_name="onnx",
        model_format_version="1",
        version=version,
        description=f"Mnist Model version {version}",
        metadata=metadata
    )
    print("Model registered successfully")


@dsl.pipeline(name="mnist")
def mnist_pipeline(app_domain: str, user_token: str, model_obc: str = "mnist-model", tag: str = "latest"):
    # Pipeline steps
    load_datasets_task = load_datasets()
    pre_process_task = pre_process(train_ds=load_datasets_task.outputs["train_ds"], test_ds=load_datasets_task.outputs["test_ds"])
    X_train_out = pre_process_task.outputs["X_train_out"]
    y_train_out = pre_process_task.outputs["y_train_out"]
    X_val_out = pre_process_task.outputs["X_val_out"]
    y_val_out = pre_process_task.outputs["y_val_out"]
    train_task = train(X_train_out=X_train_out, y_train_out=y_train_out, X_val_out=X_val_out, y_val_out=y_val_out, tag=tag)
    model_onnx_out = train_task.outputs["model_onnx_out"]
    evaluate_task = evaluate(X_val_out=X_val_out, y_val_out=y_val_out, model_onnx_out=model_onnx_out)
    register_model_task = register_model(tag=tag, model=model_onnx_out, metrics=evaluate_task.outputs["metrics"], classification_metrics=evaluate_task.outputs["classification_metrics"], app_domain=app_domain, user_token=user_token)

if __name__ == '__main__':
    host = "https://ds-pipeline-dspa.mnist:8888"
    parser = argparse.ArgumentParser(
                        prog='Model.py',
                        description='Digit recognition model and pipeline triggering')
    parser.add_argument('-t', '--tag')
    parser.add_argument('-d', '--app_domain')
    parser.add_argument('-r', '--register_model', action='store_true')
    parser.add_argument('--user_token')
    args = parser.parse_args()
    tag = args.tag
    now = str(datetime.now())
    if not tag:
        tag = now
    register_model = args.register_model
    if (args.register_model) and (args.app_domain is not None):
        app_domain = args.app_domain
    else:
        app_domain = None
    kfp.compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path='mnist-pipeline.yaml',
        pipeline_parameters={'tag': tag, 'app_domain': app_domain, 'user_token': args.user_token},
    )
    client = Client(host=host, verify_ssl=False)
    pipeline_name = "Digit recognition pipeline - KFP SDK"
    try:
        pipeline = client.upload_pipeline(pipeline_package_path="mnist-pipeline.yaml", pipeline_name=pipeline_name, description="KFP created digit recognition pipeline")
        print(pipeline)
        pipeline_id = pipeline.pipeline_id
        pipeline_versions = client.list_pipeline_versions(pipeline_id=pipeline_id)
        pipeline_version_id = pipeline_versions.pipeline_versions[0].pipeline_version_id
    except Exception as e:
        print(f"Exception raised: {e}")
        pipeline_id = client.get_pipeline_id(name=pipeline_name)
        pipeline_version = client.upload_pipeline_version(pipeline_id=pipeline_id, pipeline_version_name=now, pipeline_package_path="mnist-pipeline.yaml")
        pipeline_version_id = pipeline_version.pipeline_version_id
    print(f"Pipeline Id: {pipeline_id}")
    print(f"Pipeline Version Id: {pipeline_version_id}")
    experiment = client.create_experiment(name="git", description=f"Experiment created on git PR")
    experiment_id = experiment.experiment_id
    print(f"Experiment Id: {experiment_id}")
    pipeline_run = client.run_pipeline(job_name=tag, pipeline_id=pipeline_id, experiment_id=experiment_id, version_id=pipeline_version_id, enable_caching=True)
    print(pipeline_run)
