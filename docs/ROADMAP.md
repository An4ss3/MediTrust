# MediTrust — Roadmap

**Version :** 0.2 (corrections de revue du 21/11/2026 ; en attente d'approbation)
**Règles :**
- Une phase n'est cochée « terminée » qu'avec une preuve vérifiable (fichier, sortie de test, JSON de run).
- Chaque transition de phase demande l'approbation d'Anasse.
- Les durées sont exprimées en sessions de travail, car la date de rendu n'est pas encore connue (D14).
- **Discipline du test** (protocole, §7) : dans chaque phase, les décisions sont gelées dans le registre de gel avant l'évaluation test correspondante. Aucun résultat test ne règle une phase ultérieure.

Ordre ajusté par rapport à la proposition initiale : l'adversarial (Phase 4) passe **avant** la calibration et l'explicabilité (Phase 5), car la température et le seuil doivent être ajustés sur le modèle final (cahier des charges, §11.3).

| Phase | Intitulé | Expériences | Thèmes | Difficulté | Sessions estimées |
|---|---|---|---|---|---|
| 0 | Recherche et validation des exigences | — | — | Faible | 1 |
| 1 | Environnement et exploration de PathMNIST | — | — | Faible | 1–2 |
| 2 | Baselines CNN et ViT | E1 | T4 (et test Q, K, V de T1) | Moyenne | 2–3 |
| 3 | Apprentissage fédéré | E2 | T10 | Moyenne à élevée | 2–3 |
| 4 | Robustesse adversariale | E4 | T11 | Élevée | 1–2 |
| 5 | Calibration, explicabilité et attention | E3, E6 | T9, T1 | Moyenne | 2 |
| 6 | Optimisation Edge | E5 | T12 | Moyenne à élevée | 1–2 |
| 7 | Application Gradio et évaluation finale | Tableau de bord | Tous | Faible à moyenne | 1–2 |
| 8 | Apprentissage continu (optionnel, sur go) | E7 | T13 | Moyenne | 1 |
| 9 | Rapport final et présentation | — | Tous | Moyenne | 2 |

---

## Phase 0 — Recherche et validation des exigences

**Objectif :** disposer de spécifications approuvées.
**Dépendances :** aucune.
**Livrables :** les 4 documents de `docs/`.
**Validation :** approbation explicite des décisions D1–D15 par Anasse (et la revue ChatGPT).

- [x] Inspection du répertoire (vide au démarrage, 21/11/2026)
- [x] Recherche documentaire et vérification des sources (dataset, modèles, méthodes, références)
- [x] Rédaction de `CAHIER_DES_CHARGES.md`, `PROTOCOLE_EXPERIMENTAL.md`, `ROADMAP.md`, `PROJECT_STATUS.md`
- [x] Première revue (Anasse et ChatGPT) : architecture acceptée sur le principe ; 5 corrections demandées et intégrées en v0.2 (évaluation fédérée, D6, discipline du test, budget, T4)
- [ ] Approbation finale des décisions D1–D15
- [ ] Confirmation par l'enseignant de la couverture de T4 sans CLIP/VLM (D15). Elle peut être obtenue en parallèle de la Phase 1, mais doit l'être avant la Phase 2

## Phase 1 — Environnement et exploration de PathMNIST

**Objectif :** une base de code minimale, testée, qui charge les données et produit les partitions.
**Dépendances :** Phase 0 approuvée ; décision D12 (visibilité du dépôt) ; approbation de `git init` et de la création du dépôt.
**Livrables :** `requirements.txt`, `.gitignore`, README initial, `src/data.py`, `tests/`, `notebooks/01_data.ipynb`, `results/partitions/`.

- [ ] Vérifier les versions disponibles sur Colab (Python, torch, CUDA) et figer `requirements.txt`
- [ ] Créer un environnement virtuel local (installation de torch CPU, soumise à approbation car volumineuse)
- [ ] `src/data.py` : téléchargement ou cache Drive de `pathmnist_64.npz`, contrôle MD5, vérification des effectifs, dataset en mémoire, augmentations tensorielles
- [ ] Statistiques : effectifs par classe et par split, planche d'images par classe
- [ ] Partitions `IID` et `DIR05` (train et val), heatmaps, fichiers d'indices versionnés
- [ ] Tests pytest : partitions disjointes et couvrantes, formes, bornes [0, 1]
- [ ] Notebook 01 exécuté sur Colab depuis une session vierge

**Validation :** effectifs 89 996 / 10 004 / 7 180 et MD5 confirmés dans la sortie du notebook ; `pytest` vert ; partitions commitées.

## Phase 2 — Baselines CNN et ViT (E1)

**Objectif :** comparer `R18` et `DEIT` en centralisé et choisir le modèle de pipeline M.
**Dépendances :** Phase 1 ; D15 confirmée.
**Livrables :** `src/models.py`, `src/train.py`, `src/evaluate.py` (`metrics_from_stats`, verrou du test, registre de gel), `notebooks/02_baselines.ipynb`, `results/E1/`.

- [ ] `MediTrustModel` (upsampling, normalisation, backbone timm) ; test du recalcul de l'attention Q, K, V contre timm
- [ ] Boucle d'entraînement avec checkpoint et reprise ; test d'interruption volontaire
- [ ] `evaluate_test` qui refuse tout run absent de `results/freeze.json` ; test pytest de ce refus
- [ ] Export ONNX de test des deux modèles (dé-risquage précoce de la Phase 6)
- [ ] Pilote : chronométrage (t_U par architecture), mémoire, choix du lr sur la validation
- [ ] Projection du budget mesuré, choix du niveau de repli N0–N3 (protocole, §9) et gel avant E1 ; N3 seulement avec approbation
- [ ] 6 runs E1 (2 modèles × 3 seeds)
- [ ] Contrôle de plausibilité CR3 sur la validation ; application de la règle D3 ; gel (lr, N, checkpoints, M) **avant** l'évaluation test
- [ ] Évaluation test et tableau E1 (comparaison au benchmark à titre descriptif)

**Validation :** 6 JSON de runs ; registre de gel E1 daté et antérieur à la première ligne E1 de `test_log.csv` ; CR3 satisfait sur validation.
**Point d'approbation :** confirmation du modèle de pipeline M.

## Phase 3 — Apprentissage fédéré (E2)

**Objectif :** mesurer le coût de la décentralisation.
**Dépendances :** Phase 2 (M, lr).
**Livrables :** `src/federated.py`, `notebooks/03_federated.ipynb`, `results/E2/`.

- [ ] FedAvg pondéré, simulation séquentielle, évaluation fédérée par statistiques suffisantes (protocole, §1.3), checkpoint par round
- [ ] Tests : agrégation FedAvg exacte ; `metrics_from_stats` exacte pour les métriques additives, la macro-F1 et l'ECE, et dans la borne pour AUROC et τ ; contre-exemple montrant que la moyenne pondérée des macro-F1 locales diffère de la macro-F1 globale
- [ ] Runs `E2-IID` et `E2-DIR05` (3 seeds chacun), `E2-LOCAL`, `E2-ABL`, selon le niveau de repli gelé
- [ ] Courbes de convergence, métriques par client, volume communiqué
- [ ] Gel des checkpoints E2, puis évaluation test et tableau E2

**Validation :** tous les JSON E2 ; tests verts ; tableau comparatif avec IC.

## Phase 4 — Robustesse adversariale (E4)

**Objectif :** mesurer la vulnérabilité et durcir M_FL.
**Dépendances :** Phase 3 (M_FL, protocole FL).
**Livrables :** `src/robustness.py`, `notebooks/04_robustness.ipynb`, `results/E4/`.

- [ ] FGSM, PGD et FGSM-RS dans l'espace 64×64 [0, 1] ; tests des bornes (‖δ‖∞ ≤ ε, image dans [0, 1])
- [ ] Entraînement `E4-FLAT-DIR05-s0` avec suivi PGD sur la validation
- [ ] Évaluations sur la validation complète (propre, FGSM, PGD-10 à 2, 4 et 8/255) pour M_FL et M_FL-AT ; contrôles d'alerte FGSM/PGD, et PGD renforcé (50 itérations × 5 départs) en cas d'alerte
- [ ] Application de la règle D6 (propre et PGD, validation uniquement) et gel de M* **avant** toute évaluation test d'E4
- [ ] Évaluation test des 4 comparaisons sur la grille ε, contrôles de cohérence

**Validation :** courbes accuracy/ε ; contrôles de cohérence documentés ; M* gelé avant la première ligne E4 de `test_log.csv`.
**Point d'approbation :** confirmation de M*.

## Phase 5 — Calibration, explicabilité et attention (E3, E6)

**Objectif :** rendre la confiance exploitable et évaluer les explications.
**Dépendances :** Phase 4 (tous les checkpoints figés).
**Livrables :** fonctions de calibration dans `src/evaluate.py`, `src/explain.py`, `notebooks/05_calibration_xai.ipynb`, `results/E3/`, `results/E6/`, `calibration.json`.

- [ ] Temperature scaling (grille de T, protocole E3), ECE, NLL, Brier, reliability diagrams, risk-coverage et AURC pour toutes les variantes, sur la validation
- [ ] Seuil τ (D7) sur la validation de M*
- [ ] Grad-CAM, attention par tête, rollout, entropie
- [ ] Gel de T, τ et des paramètres E6 (protocole, §7.2), puis évaluations test d'E3 et d'E6
- [ ] Test de suppression et sanity check de randomisation
- [ ] Planches qualitatives et rédaction des limites interprétatives

**Validation :** tableaux E3 (validation et test, avant et après) ; tableau E6 chiffré ; T, τ et paramètres E6 gelés avant leurs évaluations test.

## Phase 6 — Optimisation Edge (E5)

**Objectif :** comparer FP32 et INT8 à conditions égales.
**Dépendances :** Phase 5 (M*, T).
**Livrables :** `src/edge.py`, `notebooks/06_edge.ipynb`, `results/E5/`, ONNX FP32 et INT8 sur Drive.

- [ ] Export ONNX de M* et test de parité (CR7)
- [ ] Quantification statique et dynamique ; matrice de compatibilité 2 × 2 sur les modèles E1
- [ ] Banc de latence sur CPU Colab et PC local, avec drapeaux CPU journalisés
- [ ] Recalibration de T pour l'INT8 sur la validation et effet sur τ
- [ ] Gel de la méthode de quantification, de la méthode de calibration et de T_INT8, puis évaluation test

**Validation :** tableau E5 complet sur les deux plateformes ; parité vérifiée ; gel antérieur à l'évaluation test.

## Phase 7 — Application Gradio et évaluation finale

**Objectif :** démonstrateur fonctionnel et tableau de bord QR0.
**Dépendances :** Phases 5 et 6.
**Livrables :** `app/app.py`, tableau de bord final, README complet.

- [ ] Application conforme au cahier des charges, §12.3 (FP32/INT8, confiance calibrée, revue experte, explication, latence, avertissement)
- [ ] Vérification manuelle sur des images de test après évaluation finale ; capture d'écran
- [ ] Tableau de bord final construit uniquement à partir des JSON
- [ ] README : installation, commandes, versions, temps d'exécution, attributions
- [ ] Décision go/no-go pour T13 (approbation)

**Validation :** CR9 ; tableau de bord dans `results/` ; README relu.

## Phase 8 — Apprentissage continu (optionnel, E7)

**Objectif :** mesurer l'oubli et l'effet d'EWC.
**Dépendances :** go explicite en fin de Phase 7.
**Livrables :** fonctions EWC (dans `src/train.py`), `notebooks/07_continual.ipynb`, `results/E7/`.

- [ ] Fisher diagonale, perte EWC
- [ ] Runs naïf et EWC (grille λ sur la validation)
- [ ] Tableau oubli/plasticité

**Validation :** JSON E7 et tableau.

## Phase 9 — Rapport final et présentation

**Objectif :** communiquer des résultats vérifiés.
**Dépendances :** Phase 7 (et 8 si retenue).
**Livrables :** rapport, diapositives, démonstration.

- [ ] Rapport structuré selon la progression demandée par le module : problématique → concepts → méthode → expérience → analyse → conclusion
- [ ] Chaque chiffre est relié à un JSON de run ; hypothèses et résultats séparés
- [ ] Limites et perspectives (FedProx/FedBN, DP, AutoAttack, résolution supérieure, CLIP)
- [ ] Revérification des références
- [ ] Répétition de la démonstration depuis une session vierge

**Validation :** relecture croisée ; démonstration reproduite.
