# ProgettoTesi

## Descrizione
Questo progetto è la continuazione del mio progetto di tesi, che ho ampiamente descritto nel <a href="https://github.com/EliaTosin/ProgettoTesi/blob/main/README.md">readme</a> presente nel branch main di questo repository.<br><br>Come modifiche la prima svolta è stata usando un modello più complesso (resnet 50), adattando l'output layer al numero delle classi e tenendo i layer convolutivi preaddestrati sul dataset ImageNet. Poi è stata usata una nuova libreria per le augmentation chiamata <a href="https://imgaug.readthedocs.io/en/latest/">Imgaug</a>, che ha permesso di rielaborare i colori delle immagini. Infine è stato cambiato lo scheduler (CosineAnnealingLR) che in accoppiata con ben 100 epoche di training ha raggiunto una precisione media del 95%.

## Risultati ottenuti
### Specifiche generali del modello
Inizio subito mostrando la matrice di confusione del modello ResNet e un grafico a barre che rappresenta le accuratezze per ogni classe.
![image](https://user-images.githubusercontent.com/72021066/191050277-667d6374-98a5-4171-91da-bf800a14e736.png)


### Esempio di predizioni
Volendo un'esempio grafico delle predizioni, ecco 24 immagini dal test dataset.
![image](https://user-images.githubusercontent.com/72021066/191050605-b27367e8-2590-4fc8-acc6-d6cbcec29ee0.png)
In alternativa è possibile consultare una demo live pubblicata su <a href="https://huggingface.co/spaces/EliaT/NewCarClassifier">Huggingface</a>.

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
Come possiamo notare il modello funziona abbastanza bene nei primi tre video, identificando i loghi.<br>Nel terzo il modello soffre per i riflessi dovuti alla lucentezza della verniciatura (che causano quindi indecisione) e nel quarto video con il logo che presenta dei colori inusuali (andando quindi a sbagliare la predizione).<br><br>Non contento di questi problemi, conclusa la tesi e le sue deadline, ho continuato a portare avanti questo progetto nel branch <a href="https://github.com/EliaTosin/ProgettoTesi/tree/improved">improved</a> di questo repository.
