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


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a logistic regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_linear_classifier_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear classification model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `jeopardy_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `jeopardy_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `jeopardy_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `jeopardy_dataframe` to use as target for validation.

    Returns:
      A `LinearClassifier` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["leader_win"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["leader_win"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["leader_win"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classifier


linear_classifier = train_linear_classifier_model(
    learning_rate=0.000005,
    steps=500,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

