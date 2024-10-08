from kfp.dsl import InputPath
from kfp import dsl

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["tf2onnx", "seaborn"],
)
def evaluate(
    X_val_file: InputPath(),
    y_val_file: InputPath(),
    X_test_file: InputPath(),
    model_file: InputPath()
):
    import numpy as np
    import pandas as pd
    import pickle
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    from sklearn.metrics import confusion_matrix
    from keras import backend as K

    def load_pickle(object_file):
        with open(object_file, "rb") as f:
            target_object = pickle.load(f)
        return target_object


    X_val = load_pickle(X_val_file)
    y_val = load_pickle(y_val_file)
    X_test = load_pickle(X_test_file)
    model = load_pickle(model_file)

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

    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_val_pred
    y_val_true = np.argmax(y_val,axis=1)
    y_val_true

    acc = accuracy_score(y_val_true, y_val_pred)
    f1_macro = f1_score(y_val_true, y_val_pred, average="macro")
    rec = recall_score(y_val_true, y_val_pred, average="macro")
    prec = precision_score(y_val_true, y_val_pred, average="macro")
    print(f'accuracy_score: {acc}')
    print(f'f1_score_macro: {f1_macro}')
    print(f'precision_score: {prec}')
    print(f'recall_score: {rec}')

