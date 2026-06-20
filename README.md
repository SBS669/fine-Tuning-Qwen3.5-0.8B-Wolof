# Fine-Tuning-Qwen3.5-0.8B-Wolof

Ce projet utilise le modèle Qwen3.5-0.8B qui est un modèle ultra-léger et puissant pour le fine-tuning .

L’objectif est d'avoir un modèle performant ne nécessitant pas beaucoup de puissance de calcul pour l'entraînement et la production.

---

## Objectif



* Prétraiter un dataset
* Entraîner le modèle en version quantifiée (1000 lignes de données)

---

## Bibliothèques Python principales

* Python 3
* torch (PyTorch) — Framework deep learning de base
* transformers (Hugging Face) — Chargement du modèle, tokenizer, génération de texte
* peft — Fine-tuning efficace via LoRA (Low-Rank Adaptation)
* trl — Entraînement supervisé avec SFTTrainer / SFTConfig
* bitsandbytes — Quantification 4-bit (NF4) pour réduire la mémoire GPU
* accelerate — Distribution et accélération de l'entraînement
* datasets (Hugging Face) — Chargement du dataset Wolof
* scikit-learn — Métriques d'évaluation (accuracy_score, f1_score)

---

## Dataset


vonewman/alpaca-sharegpt-wolof — dataset conversationnel en langue Wolof (Hugging Face)

---

## Techniques de fine-tuning


QLoRA = Quantification 4-bit (BitsAndBytes) + LoRA adapters sur les couches q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

---

## Environnement d'exécution

Google Colab 
GPU T4 

---

## Portabilité & Perspective

* Le modèle léger fine-tuné en Wolof peut s'intégrer facilement dans des petits dispositifs (smartphones, Raspberry Pi, edge devices) pour traiter le Wolof localement, sans nécessiter de connexion cloud.
* Entraîner le modèle avec plus de puissance de calcul pour obtenir un modèle prêt en production
