from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

jeopardy_dataframe = pd.read_csv("jeopardy_data.csv", sep=",")


def preprocess_features(jeopardy_dataframe):
    """Prepares input features from Jeopardy data set.

     Args:
       jeopardy_dataframe: A Pandas DataFrame expected to contain data
         from the California housing data set.
     Returns:
       A DataFrame that contains the features to be used for the model, including
       synthetic features.
     """

    selected_features = jeopardy_dataframe[
    ["correct_a",
     "correct_b",
     "correct_c",
     "incorrect_a",
     "incorrect_b",
     "incorrect_c",
     "num_since_correct_a",
     "num_since_correct_b",
     "num_since_correct_c",
     "dd_a",
     "dd_b",
     "dd_c",
     "num_wins_row_champ",
     "amount_before_final_a",
     "amount_before_final_b",
     "amount_before_final_c",
     "amount_leader_wagered"
     ]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["audacity_a"] = (
            jeopardy_dataframe["incorrect_a"] /
            jeopardy_dataframe["correct_a"])
    processed_features["audacity_b"] = (
            jeopardy_dataframe["incorrect_b"] /
            jeopardy_dataframe["correct_b"])
    processed_features["audacity_a"] = (
            jeopardy_dataframe["incorrect_b"] /
            jeopardy_dataframe["correct_b"])
    return processed_features


def preprocess_targets(jeopardy_dataframe):
    """Prepares target features (i.e., labels) from Jeopardy data set.

    Args:
      jeopardy_dataframe: A Pandas DataFrame expected to contain data
        from the Jeopardy data set.
    Returns:
      A DataFrame that contains the target feature.
    """

    output_targets = pd.DataFrame()
    # Create a boolean categorical feature representing whether the
    # leader before Final Jeopardy won the show
    output_targets["leader_win"] = jeopardy_dataframe["leader_win_flag"].astype(float)

    return output_targets


# Choose the first 160 (out of 200) examples for training.
training_examples = preprocess_features(jeopardy_dataframe.head(160))
training_targets = preprocess_targets(jeopardy_dataframe.head(160))

# Choose the last 40 (out of 200) examples for validation.
validation_examples = preprocess_features(jeopardy_dataframe.tail(40))
validation_targets = preprocess_targets(jeopardy_dataframe.tail(40))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
     input_features: The names of the numerical input features to use.
    Returns:
    A set of feature columns
  """

    for my_feature in input_features:
        if my_feature in ("correct_a",
            "correct_b",
            "correct_c",
            "incorrect_a",
            "incorrect_b",
            "incorrect_c",
            "num_since_correct_a",
            "num_since_correct_b",
            "num_since_correct_c",
            "dd_a",
            "dd_b",
            "dd_c",
            "num_wins_row_champ",
            "amount_before_final_a",
            "amount_before_final_b",
            "amount_before_final_c",
            "amount_leader_wagered"):
            f_list = set([tf.feature_column.numeric_column(my_feature)])
    occupation_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='occupation_leader',
        vocabulary_file="unique_occupations.txt",
        vocabulary_size=86)
    f_list.union(occupation_feature_column)
    final_category_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='final_category',
        vocabulary_file="unique_final_jeopardy_categories.txt",
        vocabulary_size=181)
    f_list.union(final_category_feature_column)

    return f_list

