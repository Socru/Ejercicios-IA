import csv
import math
import random
import requests

# descargar datos
def download_data(url, filename):
    response = requests.get(url)
    with open(filename, 'w') as file:
        file.write(response.text)

# conjunto de datos Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
filename = 'iris.data'
download_data(url, filename)

# cargar datos
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:
                dataset.append(row)
    return dataset

dataset = load_csv(filename)

#valores flotantes y las etiquetas
for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i][:-1]] + [dataset[i][-1]]

# características y etiquetas
X = [row[:-1] for row in dataset]
y = [row[-1] for row in dataset]

# conjunto entrenamiento y prueba
def train_test_split(X, y, test_size=0.2):
    dataset = list(zip(X, y))
    random.shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_size))
    train_set, test_set = dataset[:split_index], dataset[split_index:]
    train_X, train_y = zip(*train_set)
    test_X, test_y = zip(*test_set)
    return list(train_X), list(test_X), list(train_y), list(test_y)

# distancia Euclidiana
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# vecinos más cercanos
def get_neighbors(training_X, training_y, test_instance, k):
    distances = []
    for i in range(len(training_X)):
        dist = euclidean_distance(test_instance, training_X[i])
        distances.append((training_X[i], training_y[i], dist))
    distances.sort(key=lambda x: x[2])
    neighbors = distances[:k]
    return neighbors

# predicción
def predict_classification(training_X, training_y, test_instance, k):
    neighbors = get_neighbors(training_X, training_y, test_instance, k)
    output_values = [row[1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Dividir entrenamiento y prueba
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# valor de k
k = 3

# predicciones conjunto de prueba y mostrar resultados
predictions = []
for test_instance in test_X:
    prediction = predict_classification(train_X, train_y, test_instance, k)
    predictions.append(prediction)
    print(f'Predicción: {prediction}, Real: {test_y[test_X.index(test_instance)]}')

