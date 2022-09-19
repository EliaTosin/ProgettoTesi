# ProgettoTesi

## Descrizione
Questa è la continuazione del mio progetto di tesi, che ho ampiamente descritto nel <a href="https://github.com/EliaTosin/ProgettoTesi/blob/main/README.md">readme</a> presente nel branch main di questo repository.<br><br>Come modifiche la prima svolta è stata usando un modello più complesso (resnet 50), adattando l'output layer al numero delle classi e tenendo i layer convolutivi preaddestrati sul dataset ImageNet. Poi è stata usata una nuova libreria per le augmentation chiamata <a href="https://imgaug.readthedocs.io/en/latest/">Imgaug</a>, che ha permesso di rielaborare i colori delle immagini. Infine è stato cambiato lo scheduler (CosineAnnealingLR) che in accoppiata con ben 100 epoche di training ha raggiunto una precisione media del 95%.

## Risultati ottenuti
### Specifiche generali del modello
Inizio subito mostrando la matrice di confusione del modello ResNet e un grafico a barre che rappresenta le accuratezze per ogni classe.
![image](https://user-images.githubusercontent.com/72021066/191050277-667d6374-98a5-4171-91da-bf800a14e736.png)


### Esempio di predizioni
Volendo un'esempio grafico delle predizioni, ecco 24 immagini dal test dataset.
![image](https://user-images.githubusercontent.com/72021066/191050605-b27367e8-2590-4fc8-acc6-d6cbcec29ee0.png)
In alternativa è possibile consultare una demo live pubblicata su <a href="https://huggingface.co/spaces/EliaT/NewCarClassifier">Huggingface</a>.

### Predizioni da video
Riutilizzando i video che ho usato in precedenza, vediamo ora come performa il nuovo modello. Le riprese sono a puro scopo di testare il modello, non voglio minimamente esporre i proprietari dei veicoli tramite l'inquadratura della targa.

#### Logo opel


https://user-images.githubusercontent.com/72021066/191096668-e8fa146a-3f16-415d-b6ac-c1e794827dab.mp4


#### Logo hyundai


https://user-images.githubusercontent.com/72021066/191096685-862fd215-f57f-40ff-8954-7d8bbc9ca9f2.mp4


#### Logo wolkswagen e opel


https://user-images.githubusercontent.com/72021066/191096696-8683c13d-3718-4f27-8d92-b452e5b34cc1.mp4


#### Logo toyota


https://user-images.githubusercontent.com/72021066/191096709-d5c958b9-af83-412d-82f8-b8538ff43946.mp4

## Conclusioni
Come possiamo notare ora il modello funziona molto bene con i video e nei casi delle immagini va a sbagliare quando il logo non è ben visibile (caso dei cerchioni qui). Sicuramente possiamo evidenziare come problema che il modello quando non ha un logo visibile ma qualche oggetto che viene scambiato per quello (targa, qualcosa sul terreno), va a predire con molta confidenza su un logo sbagliato. Questo è dovuto al fatto che non ci sia una classe 'sconosciuta' su cui il modello sia stato allenato, in cui se no saprebbe probabilmente identificare se sia presente un logo nell'immagine processata.
