# Introduccion
Repositorio de Claudio Agüero para los trabajos en clase del proyecto de Rede Neuronales Q3 2025.

# Estructura
A cada ejercicio del proyecto le pertenece un archivo `.py`. Cada uno de estos esta enumerado al principio con el ejercicio al que le corresponde. Los archivos `.py` que no tienen un numero al principio son modulos con funciones utilizadas en los ejercicios 4 y 5.

# Correr los modelos 4 y 5
Para correr los modelos de Mnist y Fashion Mnist simplemente ejecutar su archivo de ejercicio.
## Argumentos
Los siguientes argumentos pueden ser agregados a la hora de ejecutar los ejercicios 4 y 5:

### `mode`:
Default: `test`

(type: `str`) Que hacer con el modelo, entrenarlo o probarlo.
Opciones:
- `test`
- `train`

### `epochs`:
Default: 10

(type: `int`) Cantidad de Epocas para entrenar el modelo.

### `batch-size`:
Default: 64

(type: `int`) Tamaño de los mini-batches para entrenar el modelo.

### `learning_rate`:
Default: 0.001

(type: `float`) Tasa de entrenamiento para los optimizadores.

### `mode`:
Default: `Adam`

(type: `str`) Optimizador a utilizar a la hora de entrenar.
Opciones:
- `SGD`
- `SGD+Momentum`
- `Adam`
- `RMSprop`
