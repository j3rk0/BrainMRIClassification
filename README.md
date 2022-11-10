## Un approccio basato sul text mining per la classificazione interbretabile di immaggini MRI

questa repository contiene gli script python per addestrare una pipeline di classificazione di immagini mri di tumori al cervello.
è possibile reperire il dataset al seguente indirizzo: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=7953679

L'approccio è basato sull metodo bag of visual words con normalizzazione tf-idf unito con l'estrazione di features riguardo la dimensione e la posizione del tumore e le caratteristiche della slice analizzata.

l'immagine viene preprocessata attraverso vari step e successivamente vengono estratti dei descrittori tramite RootSift.
la clusterizzazione dei keypoints è effettuata tramite un kmeans con 50 cluster.


è utilizzato come classificatore un boosting di random forest tramite xgboost.
