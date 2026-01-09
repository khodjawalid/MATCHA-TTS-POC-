# Matcha-TTS (implémentation pédagogique)

Ce dépôt propose une ossature claire et modulaire pour étudier Matcha-TTS ("Matcha-TTS: A fast TTS architecture with conditional flow matching"), centrée sur l'acoustique (mél-spectrogrammes) et destinée à des étudiants de master. Le vocoder (HiFi-GAN) est supposé externe.

## Objectifs du projet
- Fournir une base de code PyTorch propre pour l'entraînement et l'inférence d'un modèle de synthèse vocale de type Matcha-TTS sur LJSpeech.
- Illustrer l'usage du conditional flow matching (OT-CFM) pour générer des mél-spectrogrammes rapidement.
- Séparer clairement les modules (encodage texte, prédiction de durées, décodeur UNet/attention, flow, entraînement, inférence) avec masques True=valide et longueurs préservées.

## Aperçu de l'architecture Matcha-TTS
- **Encodage texte** : embeddings + blocs Transformer, vocabulaire fixe (IPA) et normalisation des phonèmes.
- **Prédicteur de durées** : convolution léger pour estimer la durée par token et dilater temporellement le texte (monotonicité préservée).
- **Décodeur** : UNet 1D conditionné avec attention; downsample par `avg_pool1d`, upsample par `interpolate(nearest)`, longueurs strictement préservées, masques alignés True=valide.
- **Flow matching conditionnel (OT-CFM)** : trajectoire linéaire bruit→mél, cible de vitesse constante; sampler Euler pour l'inférence rapide.
- **Vocoder externe** : à brancher (HiFi-GAN, non inclus) pour convertir les mél-spectrogrammes en audio.

## Pipeline d'entraînement
1. Prétraitement : phonémisation (espeak-ng) + vocabulaire statique; extraction/normalisation des mél-spectrogrammes LJSpeech.
2. Encodage texte : `TextEncoder` (Transformers) produit des embeddings masqués (True=valide).
3. Prédiction des durées : `DurationPredictor` donne des longueurs par token; upsampling texte → temps.
4. Décodeur + OT-CFM : UNet 1D conditionné prédit le champ de vitesse; perte OT-CFM masquée; prior + durée complètent les pertes.
5. Optimisation : AdamW, AMP optionnel (torch.amp), clipping gradient, scheduler; checkpoints périodiques.
6. Modes debug/validation :
	- **Overfit** sur le premier batch (rejeu cache) via `training.overfit_one_batch` ou CLI `--overfit`.
	- **Validation qualitative** : génération de mél .pt sur phrases fixes dans `outputs/val_mels/` toutes les `validation.every_steps`.

## Pipeline d'inférence
1. Texte → phonèmes (vocab fixe) → encodage.
2. Prédiction de durées → expansion temporelle.
3. Sampler Euler OT-CFM : bruit → mél en `inference.n_steps` pas, température réglable.
4. Vocoder externe pour audio (non fourni).

## Structure du repertoire implementation
Vue d'ensemble :

```
matcha_tts/
├── configs/config.yaml        # Hyperparamètres centralisés
├── data/                      # Prétraitement texte/audio, dataset LJSpeech
├── models/                    # Encodage texte, durée, décodeur, modèle complet
├── flow/ot_cfm.py             # Logique de conditional flow matching
├── training/                  # Boucle d'entraînement, pertes, scheduler
├── inference/sampler.py       # Sampler pour l'inférence CFM
├── utils/                     # Audio, masques, logging
├── train.py                   # Script d'entraînement
├── infer.py                   # Script d'inférence
 └── tests/                    # Tests unitaires/smoke
└── requirements.txt
```

### Prérequis
- Python 3.10+
- PyTorch 2.1+
- LJSpeech téléchargé et disponible à l'emplacement configuré dans `configs/config.yaml`.

### Premiers pas
```bash
pip install -r requirements.txt
# Adapter configs/config.yaml si besoin (chemins LJSpeech, workers, validation)
python -m matcha_tts.train --config matcha_tts/configs/config.yaml --dataset_path ./data/LJSpeech-1.1 --num_workers 0

# Overfit debug (réutilise le 1er batch, pas de validation)
python -m matcha_tts.train --config matcha_tts/configs/config.yaml --dataset_path ./data/LJSpeech-1.1 --overfit --overfit_steps 50 --no_val --num_workers 0

# Inférence (mél uniquement, sans vocoder)
python -m matcha_tts.infer --config matcha_tts/configs/config.yaml --text "Hello world" --output outputs/mel.pt
```

### Validation qualitative
- Activable via `validation.enabled` et `validation.every_steps` dans la config ou `--val_every` / `--no_val` en CLI.
- Sauvegardes légères `.pt` dans `outputs/val_mels/step_{global_step}_{idx}.pt` contenant `mel`, `mel_mask`, `text`, `step`.

### Notes de debug / environnements serveurs
- Masques : True = frame/phonème valide; UNet strictement longueur-préservant.
- `num_workers=0` par défaut pour limiter les soucis multiprocess en Jupyter/serveur; ajustable via CLI.
- AMP utilise `torch.amp` pour éviter les warnings `torch.cuda.amp` tout en restant compatible.

### Notes pédagogiques
- Le code est volontairement minimaliste et largement commenté en français pour faciliter la lecture.
- Les modules sont séparés pour illustrer chaque étape de la chaîne TTS.
- Le vocoder n'est pas inclus : on se concentre sur l'acoustique (mél-spectrogrammes).

## Structure du répertoire d'évaluation 
Ce dépôt contient l’ensemble des scripts et commandes utilisés pour évaluer
**Matcha-TTS** en termes de **Real-Time Factor (RTF)** et de
**Word Error Rate (WER)**, sur **CPU et GPU**.

L’objectif est de proposer un **pipeline d’évaluation simple, reproductible et
structuré**, permettant de comparer les performances de Matcha-TTS selon les
mêmes métriques que celles rapportées dans l’article ICASSP 2024.

Les fichiers volumineux (audios générés, checkpoints, sorties intermédiaires)
ne sont pas versionnés. Seuls les scripts, les commandes et les résultats
synthétiques finaux sont conservés dans le dépôt.

---

### Architecture du dépôt

matcha-eval/
├─ README.md
├─ .gitignore
├─ scripts/
│ ├─ rtf_cpu.py
│ ├─ rtf_gpu.py
│ ├─ wer.py
│ ├─ synth_batch.py
│ └─ utils.py
├─ commands/
│ ├─ install_wsl.md
│ ├─ run_cpu.md
│ └─ run_gpu.md
├─ configs/
│ └─ eval_texts.txt
└─ results/
├─ summary.csv
└─ README.md


---

## Description des répertoires

### `scripts/`
Ce répertoire contient l’ensemble des scripts Python nécessaires à la génération
et à l’évaluation des signaux audio :
- `synth_batch.py` : génération batch des audios à partir de textes d’entrée
- `rtf_cpu.py` : calcul du Real-Time Factor sur CPU
- `rtf_gpu.py` : calcul du Real-Time Factor sur GPU
- `wer.py` : calcul du Word Error Rate à l’aide d’un système de reconnaissance vocale
- `utils.py` : fonctions utilitaires partagées (durée audio, lecture/écriture, traitement du texte)

---

### `commands/`
Ce répertoire regroupe les **commandes exactes** permettant de reproduire
l’intégralité des expériences :
- `install_wsl.md` : installation des dépendances système, de l’environnement conda
  et de Matcha-TTS
- `run_cpu.md` : exécution complète du pipeline d’évaluation sur CPU
- `run_gpu.md` : exécution complète du pipeline d’évaluation sur GPU

Ces fichiers garantissent une reproductibilité totale des expériences.

---

### `configs/`
Contient les données d’entrée et paramètres d’évaluation :
- `eval_texts.txt` : ensemble des phrases utilisées pour l’évaluation
  (une phrase par ligne)

---

### `results/`
Contient uniquement les résultats finaux synthétisés :
- `summary.csv` : tableau récapitulatif des métriques RTF et WER
- `README.md` : description des colonnes et interprétation des résultats

Les fichiers intermédiaires (audios, logs, sorties temporaires) ne sont pas stockés.

---

## Protocole d’évaluation

1. Génération des signaux audio à partir des textes d’évaluation avec Matcha-TTS
2. Mesure du temps de synthèse et calcul du **Real-Time Factor (RTF)**
3. Transcription automatique des audios générés à l’aide d’un système ASR
4. Calcul du **Word Error Rate (WER)** entre le texte de référence et la transcription
5. Agrégation des résultats dans un tableau synthétique unique

Les évaluations CPU et GPU sont réalisées séparément pour plus de clarté.

---

## Reproductibilité

L’ensemble des expériences peut être reproduit en suivant les instructions
présentes dans le répertoire `commands/`.

Le dépôt est conçu pour permettre le scénario suivant :

cloner → installer → exécuter → obtenir les métriques


sans intervention manuelle supplémentaire.

---

## Remarques

- Ce dépôt contient uniquement les scripts d’évaluation.
- Matcha-TTS est utilisé comme dépendance externe.
- Une correction de compatibilité liée aux versions récentes de PyTorch est
  appliquée lors de l’installation.
- La méthodologie d’évaluation suit les métriques présentées dans l’article
  original sur Matcha-TTS.
