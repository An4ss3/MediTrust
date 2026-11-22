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
- [x] Approbation finale de la v0.2 (22/11/2026) : dépôt public, rendu en décembre 2026 (jour à confirmer), D6 accepté provisoirement avec la clarification « alerte, pas de rejet automatique » (v0.3)
- [ ] Confirmation par l'enseignant de la couverture de T4 sans CLIP/VLM (D15). Elle peut être obtenue en parallèle de la Phase 1, mais doit l'être avant la Phase 2

## Phase 1 — Environnement et exploration de PathMNIST

**Statut : TERMINÉE ET APPROUVÉE le 22/11/2026** (validée en local, puis sur Colab par Anasse : 14 tests réussis, partitions identiques aux fichiers versionnés).
**Objectif :** une base de code minimale, testée, qui charge les données et produit les partitions.
**Dépendances :** Phase 0 approuvée ; décision D12 (visibilité du dépôt) ; approbation de `git init` et de la création du dépôt.
**Livrables :** `requirements.txt`, `.gitignore`, README initial, `src/data.py`, `tests/`, `notebooks/01_data.ipynb`, `results/partitions/`.

- [x] Dépôt public `https://github.com/An4ss3/MediTrust` créé et poussé (commit `627e424`) ; un clone anonyme passe les tests
- [x] `requirements.txt` minimal (bornes inférieures). Versions exactes de Colab non vérifiées depuis le PC : à relever lors de l'exécution du notebook 01
- [x] Pas d'environnement PyTorch local (non nécessaire en Phase 1) ; seul `pytest` a été installé en local
- [x] `src/data.py` : téléchargement avec contrôle MD5, vérification des formes, des effectifs et des étiquettes, conversion en tenseur, augmentations tensorielles
- [x] Statistiques : effectifs par classe et par split, planche d'images par classe (`results/figures/`)
- [x] Partitions `IID` et `DIR05` (train et val), heatmaps, fichiers d'indices versionnés (`results/partitions/`)
- [x] Tests pytest : 13 réussis en local (dont les tests sur le vrai dataset) ; 1 test (prétraitement PyTorch) à exécuter sur Colab
- [x] Cellules du notebook 01 exécutées en local (hors chemin Colab)
- [x] Notebook 01 exécuté sur Colab depuis une session vierge (Anasse, 22/11/2026) : 14 tests réussis, hash `IID` et `DIR05` identiques

**Validation :** effectifs 89 996 / 10 004 / 7 180 et MD5 confirmés en local et sur Colab ; `pytest` vert en local et sur Colab ; partitions commitées et reproduites.

## Phase 2 — Baselines CNN et ViT (E1)

**Statut : implémentation et tests locaux terminés (22/11/2026) ; pilote GPU à exécuter par Anasse ; aucun entraînement complet lancé.**
**Objectif :** comparer `R18` et `DEIT` en centralisé et choisir le modèle de pipeline M.
**Dépendances :** Phase 1 ; D15 confirmée (en attente : la préparation continue sans CLIP, la Phase 2 complète reste bloquée).
**Livrables :** `src/models.py`, `src/train.py`, `src/evaluate.py` (`metrics_from_stats`, verrou du test, registre de gel), `notebooks/02_baselines.ipynb`, `results/E1/`.

- [x] `MediTrustModel` (upsampling, normalisation, backbone timm) ; test du recalcul de l'attention Q, K, V contre timm (écart max < 1e-5, attention fusionnée ou non)
- [x] Boucle d'entraînement avec checkpoint et reprise ; test d'interruption volontaire (reprise identique à un run ininterrompu, sur CPU)
- [x] `evaluate_test` qui refuse tout run absent de `results/freeze.json` ; test pytest de ce refus
- [x] Export ONNX de test des deux modèles : parité ONNX Runtime/PyTorch < 1e-5 en local, sur tenseurs synthétiques
- [x] Métriques par statistiques suffisantes (`metrics_from_stats`) ; tests d'égalité fédéré/regroupé et contre-exemple macro-F1
- [x] Notebook 02 (modes `PILOT` par défaut, `LR_SEARCH`, `FULL_E1` verrouillé) ; exécution de contrôle sur CPU avec données réduites
- [ ] Pilote GPU sur Colab (Anasse) : chronométrage (t_U par architecture), mémoire, taille des checkpoints, temps de validation, reprise sur GPU
- [ ] Revue du pilote ; puis recherche du lr sur la validation (mode `LR_SEARCH`)
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
