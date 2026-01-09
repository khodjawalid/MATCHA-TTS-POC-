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

## Structure du dépôt
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

## Prérequis
- Python 3.10+
- PyTorch 2.1+
- LJSpeech téléchargé et disponible à l'emplacement configuré dans `configs/config.yaml`.

## Premiers pas
```bash
pip install -r requirements.txt
# Adapter configs/config.yaml si besoin (chemins LJSpeech, workers, validation)
python -m matcha_tts.train --config matcha_tts/configs/config.yaml --dataset_path ./data/LJSpeech-1.1 --num_workers 0

# Overfit debug (réutilise le 1er batch, pas de validation)
python -m matcha_tts.train --config matcha_tts/configs/config.yaml --dataset_path ./data/LJSpeech-1.1 --overfit --overfit_steps 50 --no_val --num_workers 0

# Inférence (mél uniquement, sans vocoder)
python -m matcha_tts.infer --config matcha_tts/configs/config.yaml --text "Hello world" --output outputs/mel.pt
```

## Validation qualitative
- Activable via `validation.enabled` et `validation.every_steps` dans la config ou `--val_every` / `--no_val` en CLI.
- Sauvegardes légères `.pt` dans `outputs/val_mels/step_{global_step}_{idx}.pt` contenant `mel`, `mel_mask`, `text`, `step`.

## Notes de debug / environnements serveurs
- Masques : True = frame/phonème valide; UNet strictement longueur-préservant.
- `num_workers=0` par défaut pour limiter les soucis multiprocess en Jupyter/serveur; ajustable via CLI.
- AMP utilise `torch.amp` pour éviter les warnings `torch.cuda.amp` tout en restant compatible.

## Notes pédagogiques
- Le code est volontairement minimaliste et largement commenté en français pour faciliter la lecture.
- Les modules sont séparés pour illustrer chaque étape de la chaîne TTS.
- Le vocoder n'est pas inclus : on se concentre sur l'acoustique (mél-spectrogrammes).
