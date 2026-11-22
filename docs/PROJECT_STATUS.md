# MediTrust — Project Status

**Dernière mise à jour :** 22/11/2026, session 6 (Kiro)

## Phase actuelle
**PHASE 2 — BASELINES CNN ET VIT (E1) : RECHERCHE DU LEARNING RATE**

## Statut
**PILOTE GPU TERMINÉ ET VALIDÉ ; BUDGET N0 APPROUVÉ ; `LR_SEARCH` PRÊT, EN ATTENTE D'APPROBATION POUR EXÉCUTION.** Aucun run E1, aucune évaluation sur le test.

## Dépôt
- URL : https://github.com/An4ss3/MediTrust (public), branche `main`.
- Identité Git configurée pour ce dépôt seulement : `Anasse Harki <202443320+An4ss3@users.noreply.github.com>`.
- Colab clone le dépôt sans identifiants et ne pousse jamais.

## Environnement
| Élément | Local, Python système | Local, `.venv` (vérification Phase 2, ignoré par Git) | Colab |
|---|---|---|---|
| Python | 3.13.9 | 3.13.9 | 3.13.16 (pilote) |
| numpy | 2.3.4 | 2.5.5 | relevée dans les JSON de runs |
| torch | non installé | 2.15.0+cpu | 2.14.1+cu130 (préinstallé), GPU Tesla T4 |
| timm | — | 1.0.31 | 1.0.31 |
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
- Session 5 : correction de la reprise sur GPU (commit `63a25f6`, voir « Incidents »).
- **22/11/2026 : pilote GPU exécuté par Anasse sur Colab (T4, commit `63a25f6`, `MODE = "PILOT"`, `SMOKE = false`) et revu par Anasse et ChatGPT.** Valeurs rapportées depuis `P2-pilot.json` (fichier sur Drive, non versionné, non inspecté par Kiro) :

  | Modèle | Run | Temps/époque | Débit train | Validation | Pic GPU | Checkpoint |
  |---|---|---|---|---|---|---|
  | R18 | `LR-R18-lr0.01-s0` | 101,17 s | 889,57 img/s | 4,24 s | 3 285 Mo | 85,41 Mo |
  | DEIT | `LR-DEIT-lr0.01-s0` | 119,15 s | 755,34 img/s | 4,51 s | 3 302 Mo | 42,27 Mo |

  Contrôles d'architecture, exports ONNX et recalcul de l'attention réussis sur Colab. Ces mesures reproduisent exactement, avec `estimate_budget`, les projections de 7,50 h (M = R18) et 8,28 h (M = DEIT), marge de 30 % incluse.
- Session 6 : cellule `LR_SEARCH` du notebook 02 complétée (indique pour chaque run s'il est réutilisé ou entraîné, vérifie 2 époques terminées, enregistre `P2-lr-search.json` sur Drive). Vérifiée par une exécution à blanc avec `fit()` simulé (aucun entraînement local) : runs à 0,01 réutilisés, règle macro-F1 puis NLL correcte.

## Tâche en cours
Approbation de l'exécution de `LR_SEARCH` (notebook 02).

## Prochaine tâche
1. Anasse (après approbation) : exécuter `MODE = "LR_SEARCH"` sur Colab (4 runs restants : lr 0,003 et 0,03 × R18 et DEIT, 2 époques, seed 0 ; ≈ 30–35 min de GPU estimées d'après le pilote), puis envoyer `results/P2-lr-search.json`.
2. Revue des lr proposés, puis gel des lr par architecture.
3. `FULL_E1` seulement après la résolution de D15 et le gel des lr.

## Décisions en attente
- **D15** : confirmation par l'enseignant que T4 est couvert sans CLIP/VLM. Bloque les runs définitifs E1 ; la recherche du lr peut se faire avant.
- **D14** : jour exact du rendu en décembre 2026.
- **Correction du tableau §9 du protocole** : les projections approuvées (7,50 h / 8,28 h) utilisent déjà le décompte détaillé (≈ 192 U, dont E4 ≈ 37,6 U ; contingence PGD renforcé ≈ 55,6 U, ≈ 1,2 h pour M = R18). Le tableau §9 indique encore ≈ 180 U et E4 ≈ 25 U : à aligner si Anasse le confirme.
- **D6** : accepté provisoirement ; seuils de 2 et 5 points à confirmer avant la Phase 4.

## Blocages connus
- D15 non confirmée (bloque `FULL_E1`).

## Budget de calcul
**Niveau N0 approuvé le 22/11/2026** (10 époques, 3 seeds, 4 clients, protocole adversarial complet). Projection mesurée : 7,50 h si M = R18, 8,28 h si M = DEIT (marge 30 % incluse), soit ≈ 3 à 4 sessions Colab de 2,5 h. Provisoire : à recalculer après E1 avec l'architecture M retenue et à surveiller au fil des expériences.

## Dernière expérience validée
Aucune expérience (E1–E7) lancée. Dernier run validé : `P2-pilot` (pilote GPU, 22/11/2026), runs `LR-R18-lr0.01-s0` et `LR-DEIT-lr0.01-s0` (2 époques chacun, sur Drive).

## Décisions de projet actées
- Spécifications v0.2 approuvées le 22/11/2026, v0.3 pour la clarification de D6.
- Phase 1 approuvée le 22/11/2026.
- Pilote GPU validé et **budget N0 approuvé** le 22/11/2026 (aucune réduction d'époques, de seeds ni de résolution).
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

## Incidents
- 22/11/2026, pilote GPU (`LR-R18-lr0.01-s0`) : la reprise après l'interruption volontaire de l'époque 1 a échoué (`TypeError: RNG state must be a torch.ByteTensor`). Cause : le checkpoint était chargé avec `map_location=<cuda>`, ce qui déplaçait aussi l'état RNG CPU sur le GPU. Correction : chargement sur CPU, l'état RNG CUDA étant désormais aussi sauvegardé et restauré. Test de régression ajouté (`test_resume_rng_state_stays_on_cpu`, et `test_interrupt_resume_on_gpu` exécuté sur Colab). Le checkpoint de l'époque 1, déjà écrit, est intact et réutilisable. Méthodologie inchangée.

## Risques techniques suivis
- L'export ONNX utilise l'exportateur TorchScript (`dynamo=False`), déclaré obsolète par PyTorch 2.15 mais fonctionnel. À revérifier en Phase 6 ; repli : exportateur `torch.export`.
- Données d'entraînement placées sur le GPU (uint8, ≈ 1,1 Go) pour éviter le goulot CPU de Colab ; la mémoire GPU maximale sera mesurée par le pilote.

## Procédure de reprise de session
1. Lire ce fichier.
2. Consulter `ROADMAP.md` (première case non cochée de la phase courante).
3. Relire les sections pertinentes du cahier des charges et du protocole.
4. Reprendre la prochaine tâche sans refaire le travail validé.
5. En fin de session, mettre à jour ce fichier : date, travail réalisé avec preuve, tâche suivante, décisions, gels, déviations, blocages.
