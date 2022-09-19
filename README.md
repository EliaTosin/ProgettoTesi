# ProgettoTesi

## Descrizione
In questo progetto mi sono approcciato alla reti neurali, sviluppando un classificatore che riconosce i loghi automobilistici. Partendo con il dataset è disponible su <a href="https://www.kaggle.com/datasets/volkandl/car-brand-logos">Kaggle</a>, una nota piattaforma che propone dataset di qualsiasi tipo. Questo nel particolare include foto raccolte da diversi motori di ricerca di 8 differenti marchi.<br>Come linguaggio è stato usato Python in accoppiata con la libreria <a href="https://pytorch.org/">Pytorch</a> per l'utilizzo delle reti neurali e di <a href="https://docs.opencv.org/4.x/d1/dfb/intro.html">OpenCV</a> per il processamento di video. Come modelli usati ho utilizzato inizialmente una LeNet 5 per poi passare ad una ResNet 18, utilizzando sempre augmentation a causa del numero limitato di elementi nel dataset.

## Risultati ottenuti
Inizio subito mostrando la matrice di confusione del modello ResNet e un grafico a barre che rappresenta le accuratezze per ogni classe.
![image](https://user-images.githubusercontent.com/72021066/191021369-8be96b54-8835-4075-8111-418500cf31cf.png)

Volendo un'esempio grafico delle predizioni, ecco 24 immagini dal test dataset.
![image](https://user-images.githubusercontent.com/72021066/191021839-747345ba-54f6-4c4e-abfa-4ceb398886e2.png)
