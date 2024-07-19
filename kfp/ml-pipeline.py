import argparse

from kfp import dsl, Client
from kfp.dsl import Input, Output, Dataset, Model, HTML, ClassificationMetrics, Artifact


@dsl.component(
    base_image="quay.io/modh/runtime-images@sha256:de57a9c7bd6a870697d27ba0af4e3ee5dc2a2ab05f46885791bce2bffb77342d",
    packages_to_install=["numpy", "pandas", "matplotlib", "seaborn", "tensorflow", "scikit-learn"],
)
def pre_process(
        train_ds: Input[Dataset],
        test_ds: Input[Dataset],
        X_train_out: Output[Artifact],
        y_train_out: Output[Artifact],
        X_val_out: Output[Artifact],
        y_val_out: Output[Artifact],
        X_test_out: Output[Artifact]
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
    X_test = features_pipeline.fit_transform(df_test)

    save_pickle(X_train_out.path, X_train)
    save_pickle(y_train_out.path, y_train)
    save_pickle(X_val_out.path, X_val)
    save_pickle(y_val_out.path, y_val)
    save_pickle(X_test_out.path, X_test)


@dsl.component(
    base_image="quay.io/modh/runtime-images@sha256:de57a9c7bd6a870697d27ba0af4e3ee5dc2a2ab05f46885791bce2bffb77342d",
    packages_to_install=["numpy", "pandas", "tensorflow", "tf2onnx"],
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
    base_image="quay.io/modh/runtime-images@sha256:de57a9c7bd6a870697d27ba0af4e3ee5dc2a2ab05f46885791bce2bffb77342d",
    packages_to_install=["numpy", "pandas", "tensorflow", "scikit-learn", "onnxruntime"],
)
def evaluate(
    X_val_out: Input[Artifact],
    y_val_out: Input[Artifact],
    model_onnx_out: Input[Model]
):
    import numpy as np
    import pandas as pd
    import pickle
    import tensorflow as tf
    import onnxruntime as ort
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import confusion_matrix
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
    f1_macro = f1_score(y_val_true, y_val_pred, average="macro")
    rec = recall_score(y_val_true, y_val_pred, average="macro")
    prec = precision_score(y_val_true, y_val_pred, average="macro")
    print(f'accuracy_score: {acc}')
    print(f'f1_score_macro: {f1_macro}')
    print(f'precision_score: {prec}')
    print(f'recall_score: {rec}')


@dsl.pipeline(name="mnist")
def mnist_pipeline(model_obc: str = "mnist-model", tag: str = "latest"):
    # Pipeline steps
    import_train_ds = dsl.importer(
        artifact_uri='s3://rhods/data/train.csv',
        artifact_class=dsl.Dataset,
        reimport=True,
        metadata={})
    import_test_ds = dsl.importer(
        artifact_uri='s3://rhods/data/test.csv',
        artifact_class=dsl.Dataset,
        reimport=True,
        metadata={})
    pre_process_task = pre_process(train_ds=import_train_ds.output, test_ds=import_test_ds.output)
    X_train_out = pre_process_task.outputs["X_train_out"]
    y_train_out = pre_process_task.outputs["y_train_out"]
    X_val_out = pre_process_task.outputs["X_val_out"]
    y_val_out = pre_process_task.outputs["y_val_out"]
    X_test_out = pre_process_task.outputs["X_test_out"]
    train_task = train(X_train_out=X_train_out, y_train_out=y_train_out, X_val_out=X_val_out, y_val_out=y_val_out, tag=tag)
    model_onnx_out = train_task.outputs["model_onnx_out"]
    evaluate_task = evaluate(X_val_out=X_val_out, y_val_out=y_val_out, model_onnx_out=model_onnx_out)

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
    print(run)
    print(f"RUN_ID: {run.run_id}")
