const fs = require('fs');

// Distancia Euclidiana entre dos puntos
function euclideanDistance(point1, point2) {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
        sum += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sum);
}

// Obtener las k muestras más cercanas a un punto dado
function getKNearestNeighbors(X_train, y_train, xq, k) {
    const distances = [];
    for (let i = 0; i < X_train.length; i++) {
        const dist = euclideanDistance(X_train[i], xq);
        distances.push([dist, y_train[i]]);
    }
    distances.sort((a, b) => a[0] - b[0]); // Ordenar por distancia
    return distances.slice(0, k); // Devolver las k muestras más cercanas
}

// Clasificar la clase de un punto dado utilizando el voto de la mayoría
function classify(neighbors) {
    if (neighbors.length === 0) {
        // Manejar el caso cuando no se encuentran vecinos cercanos
        return "No neighbors found";
    }
    
    const counts = {};
    for (const [, label] of neighbors) {
        counts[label] = (counts[label] || 0) + 1;
    }
    return Object.entries(counts).reduce((a, b) => (a[1] > b[1] ? a : b))[0];
}

// Calcular la precisión del modelo
function calculateAccuracy(y_test, predictions) {
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] === y_test[i]) {
            correct++;
        }
    }
    return correct / predictions.length;
}

// Cargar el conjunto de datos Iris desde un archivo CSV
function loadIrisDataset(filename) {
    const data = fs.readFileSync(filename, 'utf8');
    const lines = data.trim().split('\n');
    const X = [];
    const y = [];
    for (const line of lines) {
        const values = line.trim().split(',').map(parseFloat); // Eliminar espacios en blanco y luego dividir por coma
        X.push(values.slice(0, 4));
        y.push(parseInt(values[4]));
    }
    return { X, y };
}

// Dividir los datos en entrenamiento y prueba
const { X, y } = loadIrisDataset('iris.csv');
const splitIndex = Math.floor(0.7 * X.length);
const X_train = X.slice(0, splitIndex);
const y_train = y.slice(0, splitIndex);
const X_test = X.slice(splitIndex);
const y_test = y.slice(splitIndex);

// Hiperparámetros
const k = 3;

// Realizar predicciones
const predictions = [];
for (const xq of X_test) {
    const neighbors = getKNearestNeighbors(X_train, y_train, xq, k);
    const predictedClass = classify(neighbors);
    predictions.push(predictedClass);
}

// Calcular la precisión del modelo
const accuracy = calculateAccuracy(y_test, predictions);
console.log(`Accuracy: ${accuracy}`);

// Predecir una nueva observación
const xqNew = [5.1, 3.5, 1.4, 0.2]; // Esta es una observación de prueba
const neighborsNew = getKNearestNeighbors(X_train, y_train, xqNew, k);
const predictedClassNew = classify(neighborsNew);
console.log(`Predicted class for xq: ${predictedClassNew}`);

// Imprimir una muestra para verificar si se cargó correctamente
console.log("Sample from dataset:");
console.log(`Features: ${X[0]}, Label: ${y[0]}`);
