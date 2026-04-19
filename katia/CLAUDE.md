# CLAUDE.md — Istruzioni per Claude Code

## Scope
- Lavora SOLO nella cartella /katia/ e nelle sue sottocartelle
- Non leggere, modificare o toccare file fuori da /katia/
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
- Label target: cell_type (e potenzialmente layer — da verificare se ridondante o informativo)
- Focus iniziale: neuroni in V1 (area più ricca di dati)
- Estensione possibile: testare su altre aree del cervello
- Classificatore aggiuntivo: label = area + layer + cell_type combinati
- VINCOLO CRITICO: usare solo functional data (NO structural/anatomical come posizione)

### Dati disponibili (via microns-datacleaner)
- Functional data: time series per neurone (calcium trace + spike trace)
- Embeddings pre-calcolati: readout_info/foundation_model.pkl su HuggingFace
  (embedding 1024-dim, ultimo layer modello foundation — suggerito dal prof)
  → Questo risolve il problema di usare time series come input al classificatore
- Paper rilevante: https://www.nature.com/articles/s41586-025-08829-y
- Dataset HuggingFace: https://huggingface.co/datasets/NeuroBLab/MICrONS/tree/main

### Package principale
microns-datacleaner — documentazione: https://microns-milano-colab.github.io/MICrONS-datacleaner/microns_datacleaner.html

## File nel repo
- `basic_tutorial.ipynb` e `tutorial_microns.ipynb`: notebook dei professori,
  LEGGI per capire il package ma NON modificare né toccare
- `exploration.ipynb`: notebook principale di lavoro — leggi le celle esistenti
  prima di aggiungere qualsiasi cosa

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