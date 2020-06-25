import os
import sys
import pandas as pd
import random


def prepareTestAndTrainDatasets(file_directory, file_name):
    main_df = pd.read_csv(file_directory + '/' + file_name, header=0)
    column_names = main_df.columns
    dataset_values = main_df.values.tolist()

    fraud_data = [[]]
    normal_data = [[]]

    for indis in range(len(dataset_values)):
        if dataset_values[indis][len(dataset_values[indis])-1] == float(0.00):
            normal_data.append(dataset_values[indis])
        else:
            fraud_data.append(dataset_values[indis])

    fraud_data.remove(fraud_data[0])
    normal_data.remove(normal_data[0])

    train_normal_data_count = int(len(normal_data) * 0.7)
    train_fraud_data_count = int(len(fraud_data) * 0.7)

    test_normal_data_count = len(normal_data) - train_normal_data_count
    test_fraud_data_count = len(fraud_data) - train_fraud_data_count

    test_normal_indices = []
    test_fraud_indices = []

    indis = 0

    while indis < test_normal_data_count:
        normal_indice = random.randint(0,len(normal_data)-1)
        if normal_indice not in test_normal_indices:
            test_normal_indices.append(normal_indice)
            indis += 1

    indis = 0

    while indis < test_fraud_data_count:
        fraud_indice = random.randint(0,len(fraud_data)-1)
        if fraud_indice not in test_fraud_indices:
            test_fraud_indices.append(fraud_indice)
            indis += 1

    train_dataset = [[]]
    test_dataset = [[]]

    fraud_indis = 0
    normal_indis = 0

    print("test_fraud_indices : ")
    print(test_fraud_indices)
    print("test_normal_indices : ")
    print(test_normal_indices)

    print("\n\n\nTEST RAW DATASET CREATION\n\n")
    for indis in range(test_fraud_data_count + test_normal_data_count):
        print(indis)
        if fraud_indis >= len(test_fraud_indices):
            print(normal_data[test_normal_indices[normal_indis]])
            test_dataset.append(normal_data[test_normal_indices[normal_indis]])
            normal_indis += 1
        elif normal_indis >= len(test_normal_indices):
            print(fraud_data[test_fraud_indices[fraud_indis]])
            test_dataset.append(fraud_data[test_fraud_indices[fraud_indis]])
            fraud_indis += 1
        else:
            random_float = random.random()

            if random_float <= float(0.1):
                print(fraud_data[test_fraud_indices[fraud_indis]])
                test_dataset.append(fraud_data[test_fraud_indices[fraud_indis]])
                fraud_indis += 1
            else:
                print(normal_data[test_normal_indices[normal_indis]])
                test_dataset.append(normal_data[test_normal_indices[normal_indis]])
                normal_indis += 1

    fraud_indis = 0
    normal_indis = 0
    test_normal_indices.sort()
    test_fraud_indices.sort()

    fraud_sequence = 0
    normal_sequence = 0

    print("\n\n\nTRAIN RAW DATASET CREATION\n\n")
    indis = 0
    while indis < (train_normal_data_count + train_fraud_data_count):
    #for indis in range(train_normal_data_count + train_fraud_data_count):
        print(indis)
        if fraud_sequence >= len(fraud_data):
            if normal_indis < len(test_normal_indices) and normal_sequence == test_normal_indices[normal_indis]:
                normal_indis += 1
            else:
                print(normal_data[normal_sequence])
                train_dataset.append(normal_data[normal_sequence])
                indis += 1
            normal_sequence += 1
        elif normal_sequence >= len(normal_data):
            if fraud_indis < len(test_fraud_indices) and fraud_sequence == test_fraud_indices[fraud_indis]:
                fraud_indis += 1
            else:
                print(fraud_data[fraud_sequence])
                train_dataset.append(fraud_data[fraud_sequence])
                indis += 1
            fraud_sequence += 1
        else:
            random_float = random.random()

            if random_float <= float(0.1):
                if fraud_sequence < test_fraud_indices[len(test_fraud_indices) - 1]:
                    while fraud_indis < len(test_fraud_indices) and fraud_sequence == test_fraud_indices[fraud_indis]:
                        fraud_sequence += 1
                        fraud_indis += 1

                print(fraud_data[fraud_sequence])
                train_dataset.append(fraud_data[fraud_sequence])
                fraud_sequence += 1
            else:
                if normal_sequence < test_normal_indices[len(test_normal_indices) - 1]:
                    while normal_indis < len(test_normal_indices) and normal_sequence == test_normal_indices[normal_indis]:
                        normal_sequence += 1
                        normal_indis += 1

                print(normal_data[normal_sequence])
                train_dataset.append(normal_data[normal_sequence])
                normal_sequence += 1

            indis += 1

    train_df = pd.DataFrame(data=train_dataset, columns=column_names)
    test_df = pd.DataFrame(data=test_dataset, columns=column_names)

    train_df.to_csv(directory_name + "/train_dataset.csv", header=0, index=None)
    test_df.to_csv(directory_name + "/test_dataset.csv", header=0, index=None)


abspath = os.path.abspath(__file__)
directory_name = os.path.dirname(abspath)
os.chdir(directory_name)

file_name = sys.argv[1]

prepareTestAndTrainDatasets(directory_name, file_name)