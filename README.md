# fcnn

Лабораторная работа №1. </br>
Полносвязная нейронная сеть с функциями активации tanh(y) на скрытом слое и softmax(y) на выходном слое. </br>
Решается задача классификации изображений, на которых изображены цифры, на 10 классов (0, 1, 2, ..., 9) </br> 
Перед запуском программы необходимо скачать датасет с сайта http://yann.lecun.com/exdb/mnist/  и распаковать его в папку с проектом

## Данные 

MNIST dataset (28\*28)
* Train set: 60 000
* Test set: 10 000

## Необходимые параметры
*	learningRate - скорость обучения
*	epochCount - количество эпох
*	hiddenSize - количество нейронов на скрытом слое

## Запуск

Приложен проект Visual Studio. Скомпилировать программу можно в самой студии.</br>
Запуск: </br>
     ``` fcnn.exe [learning_rate] [epoch] [hidden layer size]``` </br>
В случае недостатка аргументов программа завершается и выводит подсказку по запуску.

## Результаты

| Learning rate | Epochs | Hidden layer | Accuracy, (%) |
|-|-|-|-|
|  0.1 | 10 | 20 | 87.29 |
| 0.01 | 10 | 20 | 93.05 |
| 0.01 | 10 | 50 | 94.91 |
| 0.01 | 10 | 80 | 95.84 |
