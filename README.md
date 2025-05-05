# Solution RAG avec modèles LLM légers

Ce projet implémente un système RAG (Retrieval Augmented Generation) utilisant des modèles LLM  via Ollama pour traiter et interroger vos documents.



## Prérequis

- Python 3.8+ avec pip
- editeur de code (vscode/crsor) et installer l'extension jupyter notebook
- [Ollama](https://ollama.com/download) installé sur votre système
- Au moins 10 Go d'espace disque libre pour les modèles LLM
- installer les dépendances necessaires(voir fichier requirements)
-telecharger les llms




## Utilisation

1. Exécutez le script principal avec chainlit :

```bash
python app.py
```

Ce script va :
- Redémarrer Ollama si nécessaire
- Configurer le prompt, le modèle d'embedding, le modèle pour le  re-ranking et le LLM
- Charger les documents depuis le dossier `./processus_branchements` et les stocker dans le dossier `./markdown_branchements`
- Indexer les fichiers puis les charger
- Découper les documents en morceaux puis créer la base vectorielle
- Créer un index vectoriel dans Faiss
- créer le retriever
- Exécuter une requête de test
- stimuler des interactions avec le  llm

## Structure du projet

- `index.ipynb` : Script principal qui implémente le système RAG
- `./processus_branchement/` : Dossier contenant les documents à indexer (markdown, texte)
- `./faiss_index/` : Dossier où l'index vectoriel est stocké
- `./markdown_branchements/` : Dossier contenant les fichiers markdown

