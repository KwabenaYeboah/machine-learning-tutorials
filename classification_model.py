from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import logging
logging.getLogger('tensorflow').disabled = True

import pandas as pd
import tensorflow as tf

#tf.get_logger().setLevel('INFO')

#defining contants for parsing data
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

#load dataset
train = pd.read_csv(r'machine-learning-projects\iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(r'machine-learning-projects\iris_test.csv', names=CSV_COLUMN_NAMES, header=0)

#Pop species column from train and test
train_y = train.pop('Species')
test_y = test.pop('Species')  

#creating input function
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))  #Convert the inputs to Dataset

    # Shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Define the feature columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#Build the model
# Build a DNN with 2 hidden laysers with 30 and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively
    hidden_units=[30, 19],
    # The mode must choose between 3 classes
    n_classes=3)

#Train the model/classifier
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

#Evaluate the trained model
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model

#Input function for prediction
def input_fn(features, batch_size=256):
    # Convert the inputs to a dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Type numeric values as prompted.")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ': ')
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))

for pre_dict in predictions:
    class_id = pre_dict['class_ids'][0]
    probability = pre_dict['probabilities'][class_id]

    print('Predictions is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100*probability))
