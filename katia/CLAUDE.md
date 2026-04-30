# CLAUDE.md — Istruzioni per Claude Code

## Scope
- Non modificare o toccare file fuori da /katia/, quelli possono essere solo letti
- Non eseguire git add, git commit, git push, git pull — questi li gestisce l'utente

## Progetto: MICrONS — Neuron Type Classifier
Progetto universitario (Bocconi, corso di neuroscience) su MICrONs dataset:
1 mm³ di corteccia visiva di topo, con dati funzionali (calcium imaging 2P, time series
di risposta a stimoli visivi) e anatomici (connettività sinaptica, posizione, layer, cell type).

### Broad Research question
"Using functional data only, build a classifier to guess the excitatory neuron type.
Can one see different layers? Is it possible to differentiate different excitatory types
in layers 5 and 6?"

### Approccio pianificato
- Label target: layer (e potenzialmetne cell_type)
- Focus iniziale: neuroni in V1 (area più ricca di dati)
- VINCOLO CRITICO: usare solo functional data (NO structural/anatomical)

### Dati disponibili (via microns-datacleaner)
- Functional data: time series per neurone (calcium trace + spike trace), time series degli stimuli visivi, behavioural data
- Paper rilevante: https://www.nature.com/articles/s41586-025-08829-y
- Dataset HuggingFace: https://huggingface.co/datasets/NeuroBLab/MICrONS/tree/main

### Package principale
microns-datacleaner — documentazione: https://microns-milano-colab.github.io/MICrONS-datacleaner/microns_datacleaner.html

## File nel repo
- `basic_tutorial.ipynb` e `tutorial_microns.ipynb`: notebook dei professori,
  LEGGI per capire il package ma NON modificare né toccare

## Notebook — istruzioni
- Prima di aggiungere celle, leggi TUTTE quelle esistenti per capire dove siamo arrivati
- Aggiungi commenti markdown descrittivi prima di ogni sezione di codice
- Ogni cella deve fare una cosa sola (no monoblocchi)
- Se un'operazione può essere pesante (es. scaricare dati, query grandi),
  aggiungi un warning in markdown e chiedi conferma prima di eseguire

## Stile codice
- Python con pandas, numpy, matplotlib, seaborn
- Commenti in inglese (standard scientifico)
- Titoli delle sezioni markdown in inglese