# MediTrust — Project Status

**Dernière mise à jour :** 22/11/2026, session 4 (Kiro)

## Phase actuelle
**PHASE 2 — BASELINES CNN ET VIT (E1)**

## Statut
**PRÉPARATION TERMINÉE — EN ATTENTE DU PILOTE GPU (Anasse) ET DE SA REVUE.** Aucun entraînement complet, aucune recherche de lr, aucune évaluation sur le test.

## Dépôt
- URL : https://github.com/An4ss3/MediTrust (public), branche `main`.
- Identité Git configurée pour ce dépôt seulement : `Anasse Harki <202443320+An4ss3@users.noreply.github.com>`.
- Colab clone le dépôt sans identifiants et ne pousse jamais.

## Environnement
| Élément | Local, Python système | Local, `.venv` (vérification Phase 2, ignoré par Git) | Colab |
|---|---|---|---|
| Python | 3.13.9 | 3.13.9 | relevé par le notebook 02 |
| numpy | 2.3.4 | 2.5.5 | idem |
| torch | non installé | 2.15.0+cpu | préinstallé, non réinstallé |
| timm | — | 1.0.31 | installé par `requirements.txt` |
| onnx / onnxruntime | — | 1.23.2 / 1.32.0 | installés par `requirements.txt` |
| scikit-learn | 1.7.2 | dernière version | préinstallé |
| pytest | 9.2.0 | dernière version | installé si absent |

`requirements.txt` : bornes inférieures (numpy, matplotlib, pytest, torch, timm, scikit-learn, onnx, onnxruntime). Les versions exactes de chaque run sont enregistrées dans son JSON (`env`).

## Travail réalisé (avec preuve)
- Sessions 1–2 : spécifications v0.1 et v0.2.
- Session 3 : Phase 1 implémentée (commits `627e424`, `85700f3`).
- **22/11/2026 : Phase 1 validée sur Colab par Anasse et approuvée** (14 tests réussis, effectifs officiels, hash `IID` et `DIR05` identiques aux fichiers versionnés).
- Session 4 (Phase 2, préparation) :
  - `src/models.py` : `MediTrustModel` ([B,3,64,64] dans [0,1] → resize bilinéaire 224 → normalisation ImageNet → backbone timm → 9 logits). ResNet-18 `resnet18.tv_in1k` (11 181 129 paramètres avec la tête à 9 classes) et DeiT-Tiny `deit_tiny_patch16_224.fb_in1k` (5 526 153). Recalcul explicite de l'attention (T1), export ONNX.
  - `src/evaluate.py` : prédiction, statistiques suffisantes et `metrics_from_stats` (accuracy, macro-F1, NLL, Brier, ECE, AUROC par histogramme avec borne d'erreur, rappel par classe, matrice de confusion), AUROC exacte (scikit-learn), registre de gel `freeze()` et verrou `evaluate_test()`.
  - `src/train.py` : `fit()` (SGD momentum 0,9, weight decay 1e-4, cosinus constant par époque, batch 128, AMP fp16 sur GPU, clipping 1,0, dernier checkpoint), reprise automatique, temps par époque, mémoire GPU maximale, JSON par run ; `estimate_budget()` (niveaux N0–N2).
  - Poids pré-entraînés téléchargés et vérifiés en local : sortie (16, 9) finie pour les deux modèles ; export ONNX : R18 42,6 Mo, écart max 7,4e-6 ; DEIT 21,2 Mo, écart max 5,3e-6 ; argmax identiques.
  - Tests : **34 réussis** dans `.venv` (dont les tests sur le vrai dataset) ; **16 réussis, 4 ignorés** avec le Python système sans torch.
  - `notebooks/02_baselines.ipynb` : modes `PILOT` (défaut), `LR_SEARCH`, `FULL_E1` (verrouillé par `CONFIRM_FULL_E1` et `FROZEN_LR`). Le split test est retiré de la mémoire après chargement. Exécution de contrôle sur CPU avec données réduites (`MEDITRUST_SMOKE=1`) : les 3 modes passent ; l'interruption puis la reprise fonctionnent ; le garde-fou `FULL_E1` bloque. Les chiffres de temps de cette exécution CPU n'ont aucune valeur de budget.

## Tâche en cours
Pilote GPU sur Colab (Anasse) : `notebooks/02_baselines.ipynb`, `MODE = "PILOT"`.

## Prochaine tâche
1. Anasse : exécuter le pilote, puis envoyer `results/P2-pilot.json` (et la sortie des sections 7–8) pour revue.
2. Après revue : choisir le niveau de repli et l'approuver ; lancer `LR_SEARCH` ; geler les lr ; puis seulement `FULL_E1`.

## Décisions en attente
- **D15** : confirmation par l'enseignant que T4 est couvert sans CLIP/VLM (bloque la poursuite de la Phase 2 au-delà de la préparation).
- **D14** : jour exact du rendu en décembre 2026.
- **D10** : niveau de repli, à proposer à partir du pilote mesuré.
- **Correction du budget E4 (à approuver)** : le décompte détaillé des passes d'attaque dans `estimate_budget` donne pour E4 ≈ 37,6 U (20 entraînement FGSM-RS, 1,1 suivi PGD, 6,9 évaluation validation pour D6, 8,8 évaluation test PGD/FGSM, 0,8 courbes FGSM E1) au lieu de ≈ 25 U au protocole §9, plus ≈ 55,6 U de contingence si le PGD renforcé de D6 est déclenché. Total obligatoire N0 ≈ 192 U (protocole : ≈ 180). Aucune décision scientifique ne change ; seul le tableau §9 du protocole serait mis à jour, après approbation.
- **D6** : accepté provisoirement ; seuils de 2 et 5 points à confirmer avant la Phase 4.

## Blocages connus
- Pilote GPU non exécuté (impossible depuis Kiro).
- D15 non confirmée.

## Budget de calcul
Provisoire : ≈ 192 U au niveau N0 (décompte détaillé ci-dessus). Les heures seront calculées par le notebook 02 à partir des temps mesurés. Niveau de repli : non choisi.

## Dernière expérience validée
Aucune expérience (E1–E7) lancée. Dernier run validé : `P1-data` (Phase 1).

## Décisions de projet actées
- Spécifications v0.2 approuvées le 22/11/2026, v0.3 pour la clarification de D6.
- Phase 1 approuvée le 22/11/2026.
- Dépôt public `An4ss3/MediTrust`.
- Dataset : PathMNIST+ 64×64, splits officiels, test réservé à l'évaluation finale.
- Six thèmes obligatoires : T1, T4, T9, T10, T11, T12. T13 optionnel.
- Partitions de référence : `IID_k4_s0` (sha256 `609240177941c0ed…`) et `DIR05_k4_s0` (`e56860455cfc27fb…`).
- Le run pilote à lr = 0,01 (`LR-{R18,DEIT}-lr0.01-s0`, 2 époques, seed 0) est aussi le premier run de la recherche du lr (aucun calcul perdu).
- Prototype de recherche : aucune revendication clinique, de confidentialité formelle ni de déploiement edge réel.

## Registre de gel (protocole, §7)
Chaque entrée est ajoutée avant l'évaluation test correspondante : date, commit, run_id, éléments gelés. Elle est recopiée dans `results/freeze.json` (via `src.evaluate.freeze`).

| Date | Expérience | Éléments gelés | Commit |
|---|---|---|---|
| — | — | aucune entrée | — |

## Journal des évaluations sur le test
Aucune évaluation de modèle sur le test. `results/test_log.csv` n'existe pas encore.

## Déviations au protocole
Aucune.

## Risques techniques suivis
- L'export ONNX utilise l'exportateur TorchScript (`dynamo=False`), déclaré obsolète par PyTorch 2.15 mais fonctionnel. À revérifier en Phase 6 ; repli : exportateur `torch.export`.
- Données d'entraînement placées sur le GPU (uint8, ≈ 1,1 Go) pour éviter le goulot CPU de Colab ; la mémoire GPU maximale sera mesurée par le pilote.

## Procédure de reprise de session
1. Lire ce fichier.
2. Consulter `ROADMAP.md` (première case non cochée de la phase courante).
3. Relire les sections pertinentes du cahier des charges et du protocole.
4. Reprendre la prochaine tâche sans refaire le travail validé.
5. En fin de session, mettre à jour ce fichier : date, travail réalisé avec preuve, tâche suivante, décisions, gels, déviations, blocages.
