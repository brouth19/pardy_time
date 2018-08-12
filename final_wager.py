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

    #previously included "incorrect_a","incorrect_b","incorrect_c"
    selected_features = jeopardy_dataframe[
    ["correct_a",
     "correct_b",
     "correct_c",
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
    processed_features["audacity_c"] = (
            jeopardy_dataframe["incorrect_c"] /
            jeopardy_dataframe["correct_c"])
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

# Determine correlation between variables in order to select best features.
correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["leader_win"]

corr = correlation_dataframe.corr()
display.display(corr)


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


def get_quantile_based_buckets(feature_values, num_buckets):
  quantiles = feature_values.quantile(
    [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
  return [quantiles[q] for q in quantiles.keys()]



def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """

    bucketized_correct_a = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("correct_a"),
        boundaries=get_quantile_based_buckets(training_examples["correct_a"], 4))
    bucketized_correct_b = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("correct_b"),
        boundaries=get_quantile_based_buckets(training_examples["correct_b"], 4))
    bucketized_correct_c = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("correct_c"),
        boundaries=get_quantile_based_buckets(training_examples["correct_c"], 4))
    bucketized_audacity_a = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("audacity_a"),
        boundaries=get_quantile_based_buckets(training_examples["audacity_a"], 4))
    bucketized_audacity_b = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("audacity_b"),
        boundaries=get_quantile_based_buckets(training_examples["audacity_b"], 4))
    bucketized_audacity_c = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("audacity_c"),
        boundaries=get_quantile_based_buckets(training_examples["audacity_c"], 4))
    bucketized_num_since_correct_a = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("num_since_correct_a"),
        boundaries=get_quantile_based_buckets(training_examples["num_since_correct_a"],4))
    bucketized_num_since_correct_b = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("num_since_correct_b"),
        boundaries=get_quantile_based_buckets(training_examples["num_since_correct_b"], 4))
    bucketized_num_since_correct_c = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("num_since_correct_c"),
        boundaries=get_quantile_based_buckets(training_examples["num_since_correct_c"], 4))
    bucketized_amount_before_final_a = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("amount_before_final_a"),
        boundaries=get_quantile_based_buckets(training_examples["amount_before_final_a"], 6))
    bucketized_amount_before_final_b= tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("amount_before_final_b"),
        boundaries=get_quantile_based_buckets(training_examples["amount_before_final_b"], 6))
    bucketized_amount_before_final_c = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("amount_before_final_c"),
        boundaries=get_quantile_based_buckets(training_examples["amount_before_final_c"], 6))
    """
    bucketized_amount_leader_wagered = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("amount_leader_wagered"),
        boundaries=get_quantile_based_buckets(training_examples["amount_leader_wagered"], 6))
    """
    feature_columns = set([
        bucketized_correct_a,
        bucketized_correct_b,
        bucketized_correct_c,
    bucketized_amount_before_final_a,
    bucketized_amount_before_final_b,
    bucketized_amount_before_final_c,
    bucketized_audacity_a,
    bucketized_audacity_b,
    bucketized_audacity_c,
    bucketized_num_since_correct_a,
    bucketized_num_since_correct_b,
    bucketized_num_since_correct_c])
    feature_columns.union([tf.feature_column.numeric_column("dd_a")])
    feature_columns.union([tf.feature_column.numeric_column("dd_b")])
    feature_columns.union([tf.feature_column.numeric_column("dd_c")])
    feature_columns.union([tf.feature_column.numeric_column("num_wins_row_champ")])
    feature_columns.union([tf.feature_column.numeric_column("amount_leader_wagered")])
    occupation_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='occupation_leader',
        vocabulary_file="unique_occupations.txt",
        vocabulary_size=86)
    feature_columns.union(occupation_feature_column)
    final_category_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='final_category',
        vocabulary_file="unique_final_jeopardy_categories.txt",
        vocabulary_size=181)
    feature_columns.union(final_category_feature_column)

    return feature_columns


def model_size(estimator):
    variables = estimator.get_variable_names()
    size = 0
    for variable in variables:
        if not any(x in variable
               for x in ['global_step',
                         'centered_bias_weight',
                         'bias_weight',
                         'Ftrl']
              ):
          size += np.count_nonzero(estimator.get_variable_value(variable))
    return size


def train_linear_classifier_model(
        learning_rate,
        regularization_strength,
        steps,
        batch_size,
        feature_columns,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model.

      In addition to training, this function also prints training progress information,
      as well as a plot of the training and validation loss over time.

      Args:
        learning_rate: A `float`, the learning rate.
        regularization_strength: A `float` that indicates the strength of the L1
           regularization. A value of `0.0` means no regularization.
        steps: A non-zero `int`, the total number of training steps. A training step
          consists of a forward and backward pass using a single batch.
        feature_columns: A `set` specifying the input feature columns to use.
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

    periods = 7
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                          l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
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
    learning_rate=0.1,
    regularization_strength=0.1,
    steps=300,
    batch_size=100,
    feature_columns=construct_feature_columns(),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
print("Model size:", model_size(linear_classifier))

