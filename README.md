# ProgettoTesi

## Descrizione
In questo progetto mi sono approcciato alla reti neurali, sviluppando un classificatore che riconosce i loghi automobilistici. Partendo con il dataset è disponible su <a href="https://www.kaggle.com/datasets/volkandl/car-brand-logos">Kaggle</a>, una nota piattaforma che propone dataset di qualsiasi tipo. Questo nel particolare include foto raccolte da diversi motori di ricerca di 8 differenti marchi.<br>Come linguaggio è stato usato Python in accoppiata con la libreria <a href="https://pytorch.org/">Pytorch</a> per l'utilizzo delle reti neurali e di <a href="https://docs.opencv.org/4.x/d1/dfb/intro.html">OpenCV</a> per il processamento di video. Come modelli usati ho utilizzato inizialmente una LeNet 5 per poi passare ad una ResNet 18, utilizzando sempre augmentation a causa del numero limitato di elementi nel dataset.

## Risultati ottenuti
### Specifiche generali del modello
Inizio subito mostrando la matrice di confusione del modello ResNet e un grafico a barre che rappresenta le accuratezze per ogni classe.
![image](https://user-images.githubusercontent.com/72021066/191021369-8be96b54-8835-4075-8111-418500cf31cf.png)

### Esempio di predizioni
Volendo un'esempio grafico delle predizioni, ecco 24 immagini dal test dataset.
![image](https://user-images.githubusercontent.com/72021066/191022160-b3b41985-83d9-4baf-b5c7-8d01fe69814e.png)

### Predizioni da video
Spingendomi oltre, ho realizzato dei video in cui vado ad inquadrare dei loghi che ho trovato per farli processare dalla rete neurale. Le riprese sono a puro scopo di testare il modello, non voglio minimamente esporre i proprietari dei veicoli tramite l'inquadratura della targa.

#### Logo opel
https://user-images.githubusercontent.com/72021066/191024476-c74b6d73-316f-4e79-a2c9-5603777c9630.mp4

#### Logo hyundai
https://user-images.githubusercontent.com/72021066/191025400-2a0942cb-5962-4651-9d37-a0326e2056b2.mp4

#### Logo wolkswagen e opel
https://user-images.githubusercontent.com/72021066/191024906-647cc224-e268-4154-82d5-8bfbb206fce9.mp4




