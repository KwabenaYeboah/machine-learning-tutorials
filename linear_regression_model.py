from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled=True

import pandas as pd
import tensorflow as tf
#load dataset
dftrain = pd.read_csv(r'machine-learning-projects\train.csv')
dfeval = pd.read_csv(r'machine-learning-projects\eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary_list = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary_list))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#Input Function
def make_input_fn(data_df, lable_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), lable_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train) # Train input Function
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False) # Evaluate input function

#Creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

#Training the model
linear_est.train(train_input_fn)
results = linear_est.evaluate(eval_input_fn) #Testing the model

#Making predictions
results = list(linear_est.predict(eval_input_fn))
print('\n\n')
print(dfeval.loc[3])
print(y_eval.loc[3])

print(f"Probability of surviving: {results[4]['probabilities'][1]}")


