import math
import os
import pandas as pd

def mean(array):
    mean_val = 0.0
    for indis in range(len(array)):
        mean_val += array[indis]
    mean_val /= len(array)

    return mean_val

def std(array):
    mean_val = mean(array)

    std_val = 0.0
    for indis in range(len(array)):
        std_val += (array[indis] - mean_val)**2
    std_val /= (len(array) - 1)
    std_val = math.sqrt(std_val)

    return std_val

def column_base_naive_bayes_probability(column_mean_value,column_std_value,given_value):
    return ( 1 / ( math.sqrt( 2 * math.pi ) * column_std_value ) ) * math.e**( (given_value - column_mean_value)**2 / ((-2) * (column_std_value)**2 ) )

def transpoze(matrix):
    result_matrix = [[]]
    for i in range(len(matrix[0])):
        temp_array = []
        for j in range(len(matrix)):
            temp_array.append(matrix[j][i])
        result_matrix.append(temp_array)
    result_matrix.remove(result_matrix[0])
    return result_matrix

def naiveBayesClassification(used_field_info, label_array, example_data_matrix):
    fraud_mean_array = []
    normal_mean_array = []
    fraud_std_array = []
    normal_std_array = []
    normal = [[]]
    fraud = [[]]
    fraud_data_count = 0
    normal_data_count = 0

    for indis in range(len(used_field_info[0])):
        temp = []
        for element in range(len(used_field_info)):
            temp.append(used_field_info[element][indis])

        if label_array[indis] == 0.0:
            normal.append(temp)
            normal_data_count += 1
        else:
            fraud.append(temp)
            fraud_data_count += 1

    normal.remove(normal[0])
    fraud.remove(fraud[0])

    for indis in range(len(used_field_info)):
        fraud_mean_array.append(mean(fraud[:][indis]))
        fraud_std_array.append(std(fraud[:][indis]))
        normal_mean_array.append(mean(normal[:][indis]))
        normal_std_array.append(std(normal[:][indis]))

    naive_label = []

    for indis in range(len(example_data_matrix[0])):
        fraud_probability = 1.0
        normal_probability = 1.0

        for element in range(len(example_data_matrix)):
            normal_probability *= column_base_naive_bayes_probability(normal_mean_array[element],normal_std_array[element],example_data_matrix[element][indis])
            fraud_probability *= column_base_naive_bayes_probability(fraud_mean_array[element],fraud_std_array[element],example_data_matrix[element][indis])

        fraud_probability *= (fraud_data_count / len(used_field_info[0]))
        normal_probability *= (normal_data_count / len(used_field_info[0]))

        print("fraud probability : ")
        print(fraud_probability)
        print("normal probability : ")
        print(normal_probability)

        if normal_probability == 0.0:
            naive_label.append(0.0)
        else:
            if fraud_probability > normal_probability:
                naive_label.append(1.0)
            else:
                naive_label.append(0.0)

    return naive_label


abspath = os.path.abspath(__file__)
directory_name = os.path.dirname(abspath)
os.chdir(directory_name)


print("train dataframe \n\n")
raw_train_data_values = pd.read_csv(directory_name + "/train_dataset.csv", header=0).values.tolist()
print(len(raw_train_data_values))

raw_test_data_values = pd.read_csv(directory_name + "/test_dataset.csv", header=0).values.tolist()

train_dataset_used_values = [[]]
train_dataset_label = []

for indis in range(len(raw_train_data_values)):
    train_dataset_used_values.append(raw_train_data_values[indis][1:-1])
    train_dataset_label.append(raw_train_data_values[indis][-1])

train_dataset_used_values.remove(train_dataset_used_values[0])

print(train_dataset_used_values[0])
print(train_dataset_label[0])

print("test datası : ")
print(raw_test_data_values[0][1:-1])

#kNNCalculation(used_field_info=train_dataset_used_values, label_array=train_dataset_label, example_data=raw_test_data_values[0][1:-1])

transpozed_matrix = transpoze(train_dataset_used_values)

test_dataset_used_values = [[]]
test_dataset_label = []

for indis in range(len(raw_test_data_values)):
    test_dataset_used_values.append(raw_test_data_values[indis][1:-1])
    test_dataset_label.append(raw_test_data_values[indis][-1])

test_dataset_used_values.remove(test_dataset_used_values[0])
example_datas = transpoze(test_dataset_used_values)

print(len(example_datas))
print(len(example_datas[0]))




print(len(transpozed_matrix))
print(len(transpozed_matrix[0]))


naive_bayes_results = naiveBayesClassification(transpozed_matrix, train_dataset_label, example_datas)



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
fn = 0

for indis in range(len(raw_test_data_values)):
    if naive_bayes_results[indis] == 0.0:
        if test_dataset_label[indis] == 0.0:
            tp += 1
        else:
            fp += 1
    else:
        if test_dataset_label[indis] == 1.0:
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

for indis in range(len(naive_bayes_results)):
    if naive_bayes_results[indis] == 1.0:
        count += 1

print("count : ")
print(count)