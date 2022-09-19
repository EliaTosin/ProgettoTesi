# ProgettoTesi

## Descrizione
In questo progetto mi sono approcciato alla reti neurali, sviluppando un classificatore che riconosce i loghi automobilistici. Partendo con il dataset è disponible su <a href="https://www.kaggle.com/datasets/volkandl/car-brand-logos">Kaggle</a>, una nota piattaforma che propone dataset di qualsiasi tipo. Questo nel particolare include foto raccolte da diversi motori di ricerca di 8 differenti marchi.<br>Come linguaggio è stato usato Python in accoppiata con la libreria <a href="https://pytorch.org/">Pytorch</a> per l'utilizzo delle reti neurali e di <a href="https://docs.opencv.org/4.x/d1/dfb/intro.html">OpenCV</a> per il processamento di video. Come modelli usati ho utilizzato inizialmente una LeNet 5 per poi passare ad una ResNet 18, utilizzando sempre augmentation a causa del numero limitato di elementi nel dataset.<br>Per consultare come è stato allenato il modello, le augmentation usate, i test svolti e altro è tutto documentato in questo repository.

## Risultati ottenuti
### Specifiche generali del modello
Inizio subito mostrando la matrice di confusione del modello ResNet e un grafico a barre che rappresenta le accuratezze per ogni classe.
![image](https://user-images.githubusercontent.com/72021066/191037398-4763ef21-d095-44a8-b850-bd871178aa21.png)


### Esempio di predizioni
Volendo un'esempio grafico delle predizioni, ecco 24 immagini dal test dataset.
![image](https://user-images.githubusercontent.com/72021066/191022160-b3b41985-83d9-4baf-b5c7-8d01fe69814e.png)<br>
In alternativa è possibile consultare una demo live pubblicata su <a href="https://huggingface.co/spaces/EliaT/ClassificatoreTesi">Huggingface</a>.

### Predizioni da video
Spingendomi oltre, ho realizzato dei video in cui vado ad inquadrare dei loghi che ho trovato per farli processare dalla rete neurale. Le riprese sono a puro scopo di testare il modello, non voglio minimamente esporre i proprietari dei veicoli tramite l'inquadratura della targa.

#### Logo opel


https://user-images.githubusercontent.com/72021066/191032061-34d6386b-5995-4707-9bea-dfe982fa5678.mp4


#### Logo hyundai


https://user-images.githubusercontent.com/72021066/191031600-1d73702e-329a-487e-86f7-7be88f3cc214.mp4


#### Logo wolkswagen e opel


https://user-images.githubusercontent.com/72021066/191032082-7d7a9a22-b3a6-4ace-b360-7c5c5cf7240b.mp4

#### Logo toyota


https://user-images.githubusercontent.com/72021066/191034460-38cdbbfa-5b35-4e5b-a780-5ad39e9fdeaa.mp4


## Conclusioni
Come possiamo notare il modello funziona abbastanza bene nei primi tre video, identificando i loghi.<br>Nel terzo il modello soffre per i riflessi dovuti alla lucentezza della verniciatura (che causano quindi indecisione) e nel quarto video con il logo che presenta dei colori inusuali (andando quindi a sbagliare la predizione).<br><br>Non contento di questi risultati, conclusa la tesi e le sue deadline, ho continuato a portare avanti questo progetto nel branch <a href="https://github.com/EliaTosin/ProgettoTesi/tree/improved">improved</a> di questo repository.
