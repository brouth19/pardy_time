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
    """Prepares input features from California housing data set.

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

    return selected_features


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
    # median_house_value is above a set threshold.
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

