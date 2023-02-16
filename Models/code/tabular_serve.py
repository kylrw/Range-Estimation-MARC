import ast
import json
import os
from io import StringIO
from typing import Any, List

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

HEADER = ['V', 'I', 'T']
LABELS = "labels"
PROBABILITY = "probability"
PROBABILITIES = "probabilities"
PREDICTED_LABEL = "predicted_label"
AGT_MODEL_NAME = "MODEL_NAME"
SAGEMAKER_INFERENCE_OUTPUT = "SAGEMAKER_INFERENCE_OUTPUT"
DATA = "data"
FEATURES = "features"
VALUES = "values"
SCORE = "score"
PREDICTIONS = "predictions"
TEXT_CSV = "text/csv"
APPLICATION_JSONLINES = "application/jsonlines"
APPLICATION_JSON = "application/json"


def generate_single_csv_line_inference_selection(data: List):
    """
    Generate a single csv line response.
    :param data: list of output generated from the tmodel.
    :return: a generator that produces a csv line for each datapoint.
    """
    for single_prediction in data:
        # Wrap lists with double quotes to avoid confusion in the csv file
        contents = '"{}"'.format(single_prediction) if isinstance(single_prediction, list) else str(single_prediction)
        yield contents


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir, require_version_match=False)
    model.persist_models()
    globals()["column_names"] = HEADER if HEADER else model.feature_metadata_in.get_features()

    return model


def transform_fn(model, request_body, input_content_type, output_content_type=TEXT_CSV):
    if input_content_type.lower() == TEXT_CSV:
        pass
    elif input_content_type.lower() in (APPLICATION_JSON, APPLICATION_JSONLINES):
        request_body = ast.literal_eval(request_body)
        try:
            if FEATURES in request_body:
                request_body = request_body[FEATURES]
            else:
                request_body = request_body[DATA][FEATURES][VALUES]
        except KeyError:
            raise Exception(f"{input_content_type} content type does not has correct format")
        request_body = ",".join(str(e) for e in request_body)
    else:
        raise Exception(f"{input_content_type} input content type not supported")

    if output_content_type.lower() not in (TEXT_CSV, APPLICATION_JSON, APPLICATION_JSONLINES):
        raise Exception(f"{output_content_type} output content type not supported")

    buf = StringIO(request_body)
    data = pd.read_csv(buf, header=None)
    num_cols = len(data.columns)
    global_column_names = globals()["column_names"]
    if num_cols != len(global_column_names):  # noqa: F821
        raise Exception(
            "Invalid data format. Input data has {} while the model expects {} {} {}"
            .format(num_cols, len(global_column_names), global_column_names, type(global_column_names))  # noqa: F821
        )
    else:
        data.columns = global_column_names  # noqa: F821

    MODEL_NAME_ENV_VALUE = os.getenv(AGT_MODEL_NAME)
    MODEL_NAME = None
    if MODEL_NAME_ENV_VALUE:
        MODEL_NAME = MODEL_NAME_ENV_VALUE.replace("-", "_")

    # Prepare inference output
    # Predictions should be stored in exact order as given in INFERENCE_OUTPUT
    INFERENCE_OUTPUT = os.getenv(SAGEMAKER_INFERENCE_OUTPUT)
    response: Any = None
    if INFERENCE_OUTPUT:
        return_list = []
        for inference_output in INFERENCE_OUTPUT.split(","):
            if inference_output == PREDICTED_LABEL:
                prediction = model.predict(data, model=MODEL_NAME)
            elif inference_output == PROBABILITIES:
                prediction = model.predict_proba(data, model=MODEL_NAME).to_numpy()
            elif inference_output == LABELS:
                num_rows = len(data)
                class_labels_single = pd.DataFrame(model.class_labels).T
                prediction = pd.concat([class_labels_single] * num_rows).to_numpy().astype('str')
            else:
                prediction = model.predict_proba(data, model=MODEL_NAME).to_numpy()
                prediction = np.max(prediction, axis=1)
            return_list.append(generate_single_csv_line_inference_selection(prediction.tolist()))

        if output_content_type.lower() == TEXT_CSV:
            csv_lines = []
            for r_value in zip(*return_list):
                line = ",".join(r_value)
                csv_lines.append(line)
            response = "\n".join(csv_lines) + "\n"
        else:
            dict_lines = []
            for r_value in zip(*return_list):
                r_dict = dict()
                for idx, key in enumerate(INFERENCE_OUTPUT.split(",")):
                    if key == PREDICTED_LABEL:
                        r_dict[key] = r_value[idx]
                    else:
                        r_dict[key] = ast.literal_eval(r_value[idx])
                dict_lines.append(r_dict)
            if output_content_type.lower() == APPLICATION_JSONLINES:
                response = "\n".join([json.dumps(line) for line in dict_lines]) + "\n"
            else:
                response = json.dumps(dict({PREDICTIONS: dict_lines}))
    else:
        prediction = model.predict(data, model=MODEL_NAME)
        if output_content_type.lower() == TEXT_CSV:
            response = "\n".join(generate_single_csv_line_inference_selection(prediction.tolist())) + "\n"
        elif output_content_type.lower() == APPLICATION_JSONLINES:
            response = "\n".join([json.dumps(line) for line in prediction.tolist()]) + "\n"
        else:
            output_dict = [{SCORE: value} for value in prediction.tolist()]
            response = json.dumps(dict({PREDICTIONS: output_dict}))

    return response, output_content_type
