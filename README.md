# ASSISTANT_IA

## Aperçu

Cette application est un système RAG (Retrieval-Augmented Generation) conçu pour fournir des réponses précises et contextuelles sur les procédures de branchement électrique d'ENEO Cameroun. Elle utilise des documents Markdown comme source de connaissances, indexés via un magasin vectoriel FAISS, et génère des réponses à l'aide d'un modèle de langage local (Mistral 7B via Ollama). L'interface utilisateur est basée sur Chainlit, et les fichiers sources (PDF, DOC, DOCX, TXT) sont servis via une route statique pour un accès facile.
Fonctionnalités principales

- Réponses basées sur des documents : Fournit des réponses uniquement à partir des documents fournis, avec des citations cliquables vers les fichiers sources.
- Interface utilisateur intuitive : Interface web interactive via Chainlit.
- Accès aux documents sources : Liste des sources disponible dans source_links.md et via la commande list sources.
- Optimisation CPU : Conçu pour fonctionner sur des serveurs sans GPU.
- Cache en mémoire : Réponses mises en cache pour améliorer les performances.
- reformulations des requêtes utilisateur: reformulation des requêtes utilisateur pour une meiulleure compréhension afin d'optimiser la récupération des documents pertinents.

## Prérequis

- Python : Version 3.9 ou supérieure.
- Ollama : Installé et configuré avec le modèle de votre choix.
- Dépendances Python : Listées dans requirements.txt.
- Dossier markdown_branchements contenant les fichiers Markdown des documents.
- Dossier processus_branchement contenant les fichiers sources (PDF, DOC, DOCX, TXT) organisés dans des sous-dossiers si nécessaire.
- Au moins 10 Go d'espace disque libre pour les modèles LLM

## Installation

Cloner le dépôt :
git clone https://github.com/https://github.com/Anaaurelle237/Assistant_IA.git
cd Assistant_IA


## Créer un environnement virtuel :
python -m venv venv
Sur Windows : venv\Scripts\activate


## Installer les dépendances :
pip install -r requirements.txt


## Configurer Ollama :

Installez Ollama et les modèles que vous souhaitez


## Configurer les dossiers :

Placez vos fichiers Markdown dans markdown_branchements.
Placez vos fichiers sources (PDF, DOC, DOCX, TXT) dans processus_branchement, avec une structure de sous-dossiers si nécessaire.
Exemple de structure :
projet/
├── markdown_branchements/
│   ├── procedure.md
│   ├── branchement.md
├── processus_branchement/
│   ├── sous_dossier/
│   │   ├── procedure.pdf
│   ├── autre_sous_dossier/
│   │   ├── branchement.docx
├── app.py
├── requirements.txt


## Configurer l'URL du serveur :

Modifiez le fichier app.py pour définir CONFIG["base_url"] avec l'URL de votre serveur (par exemple, http://localhost:8000 pour un développement local ou http://<votre-domaine>:<port> pour un serveur distant).


## Utilisation

Lancer l'application :
chainlit run app.py
L'application démarre par défaut sur http://localhost:8000. Si vous utilisez un serveur distant, vérifiez l'URL configurée.

## Interagir avec l'application :

Poser une question : Entrez une question . Les réponses incluent des liens cliquables vers les documents sources. Si le lien n'est pas cliquable, copier et coller dans un nouvel onglet pour afficher le contenu du document. 
Vider le cache : Tapez clear cache pour réinitialiser le cache des réponses.



Pour personnaliser, modifiez le fichier app.py avant de lancer l'application.

## Dépannage

- Les liens redirigent vers l'interface Chainlit : Vérifiez que CONFIG["base_url"] correspond à l'URL publique de votre serveur.
Assurez-vous que le dossier processus_branchement est accessible et que les fichiers sources existent.


- Temps de réponse lent : Vérifiez que l'index FAISS (faiss_index.pkl) est chargé correctement (logs à l'initialisation).
Réduisez top_k_retrieve ou augmentez chunk_size dans CONFIG pour optimiser. Supprimer aussi la fonction de reformulation de questions.


- Contribuer: Clonez le dépôt.
Créez une branche pour vos modifications :git checkout -b ma-fonctionnalite

- Soumettez une pull request avec une description claire de vos changements.

## Licence
Ce projet est sous licence Apache License 2.0. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
