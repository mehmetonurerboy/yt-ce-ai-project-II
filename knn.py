import pandas as pd
import numpy as np
import os
import math

def calculate_distance(element1,element2):
    distance = 0.0
    for indis in range(len(element1)):
        distance += (element1[indis] - element2[indis])**2
    return math.sqrt(distance)

def kNNCalculation(used_field_info, label_array, example_data, k_value=1):
    calculated_labels = []
    for test_element in range(len(example_data)):
        print(test_element)
        lengths = []
        for train_element in range(len(used_field_info)):
            lengths.append(calculate_distance(used_field_info[train_element],example_data[test_element]))

        length_sequence = np.argsort(lengths)

        label_value = 0.0
        count = 0
        indis = 0
        while indis < len(length_sequence) and count < k_value:
            if length_sequence[indis] in range(0,(k_value - 1)):
                label_value += label_array[length_sequence[indis]]
                count += 1
            indis += 1

        if label_value > (k_value / 2):
            label_value = 1.0
        else:
            label_value = 0.0

        print("Calculated label field : ")
        print(label_value)
        calculated_labels.append(label_value)

    return calculated_labels


abspath = os.path.abspath(__file__)
directory_name = os.path.dirname(abspath)
os.chdir(directory_name)

print(directory_name)

#prepareTestAndTrainDatasets(directory_name, 'creditcard.csv', 'test.csv')

print("train dataframe \n\n")
raw_train_data_values = pd.read_csv(directory_name + "/train_dataset.csv", header=0).values.tolist()
print(len(raw_train_data_values))

raw_test_data_values = pd.read_csv(directory_name + "/test_dataset.csv", header=0).values.tolist()

train_dataset_used_values = [[]]

for indis in range(len(raw_train_data_values)):
    train_dataset_used_values.append(raw_train_data_values[indis][1:-1])

test_label_array = []
example_data = [[]]
for indis in range(len(raw_test_data_values)):
    example_data.append(raw_test_data_values[indis][1:-1])
    test_label_array.append(raw_test_data_values[indis][len(raw_test_data_values[indis]) - 1])

train_dataset_used_values.remove(train_dataset_used_values[0])
example_data.remove(example_data[0])

knn_results = kNNCalculation(used_field_info=train_dataset_used_values, label_array=test_label_array, example_data=example_data)


#        predict
#      |----|----|
#   r  | TP | FN |
#   e  |----|----|
#   a  | FP | TN |
#   l  |----|----|


tp = 0
fp = 0
tn = 0
fn = 0

print(knn_results)

for indis in range(len(raw_test_data_values)):
    if knn_results[indis] == 0.0:
        if knn_results[indis] == 0.0:
            tp += 1
        else:
            fp += 1
    else:
        if knn_results[indis] == 1.0:
            tn += 1
        else:
            fn += 1

tp /= len(raw_test_data_values)
fp /= len(raw_test_data_values)
tn /= len(raw_test_data_values)
fn /= len(raw_test_data_values)

print("tp : ")
print(tp)
print("fn : ")
print(fn)
print("fp : ")
print(fp)
print("tn : ")
print(tn)

print("\n\n\n\n")

count = 0

for indis in range(len(knn_results)):
    if knn_results[indis] == 1.0:
        count += 1

print("count : ")
print(count)