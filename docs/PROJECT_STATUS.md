# MediTrust — Project Status

**Dernière mise à jour :** 22/11/2026, session 3 (Kiro)

## Phase actuelle
**PHASE 1 — ENVIRONNEMENT ET DONNÉES**

## Statut
**IMPLÉMENTÉE ET VALIDÉE EN LOCAL — EN ATTENTE DE L'EXÉCUTION COLAB ET DE LA REVUE** (Phase 2 non commencée)

## Dépôt
- URL : https://github.com/An4ss3/MediTrust (public)
- Branche `main`. Commit de la Phase 1 : `627e424`.
- Identité Git configurée **pour ce dépôt seulement** : `Anasse Harki <202443320+An4ss3@users.noreply.github.com>`, pour ne pas publier l'adresse Gmail.
- Colab clone le dépôt sans identifiants et ne pousse jamais.

## Environnement
| Élément | Local (Windows 11) | Colab |
|---|---|---|
| Python | 3.13.9 | à relever (notebook 01, cellule 1) |
| numpy | 2.3.4 | à relever |
| matplotlib | 3.10.8 | à relever |
| pytest | 9.2.0 (installé en session 3) | installé par `requirements.txt` si absent |
| torch | non installé (non nécessaire en Phase 1) | préinstallé, version à relever |
| git | 2.50.0 ; GitHub CLI absent ; authentification via Git Credential Manager (compte `An4ss3`) | — |

`requirements.txt` ne contient que des bornes inférieures (numpy, matplotlib, pytest, torch) pour ne pas réinstaller torch sur Colab. Les versions exactes seront relevées dans les JSON de runs à partir de la Phase 2.

## Travail réalisé (avec preuve)
- Sessions 1–2 : spécifications v0.1 et v0.2.
- Session 3 (22/11/2026) :
  - v0.2 approuvée. D6 clarifié en v0.3 : une relation FGSM/PGD inattendue est une alerte qui déclenche un PGD renforcé (50 itérations × 5 départs), sans rejet automatique.
  - Fichiers créés : `.gitignore`, `.gitattributes`, `README.md`, `requirements.txt`, `src/data.py`, `tests/conftest.py`, `tests/test_data.py`, `notebooks/01_data.ipynb`, `results/partitions/{IID,DIR05}_k4_s0.npz`, `results/figures/phase1_*.png`, `results/phase1_data_summary.json`.
  - Dataset téléchargé en local (`data/pathmnist_64.npz`, 1,07 Go, ignoré par Git) : MD5 `55aa9c1e0525abe5a6b9d8343a507616` conforme.
  - Intégrité vérifiée : train (89 996, 64, 64, 3), val (10 004, …), test (7 180, …), uint8, pixels dans [0, 255], 9 classes présentes dans chaque split.
  - Effectifs par classe : train [9366, 9509, 10360, 10401, 8006, 12182, 7886, 9401, 12885] ; val [1041, 1057, 1152, 1156, 890, 1354, 877, 1045, 1432] ; test [1338, 847, 339, 634, 1035, 592, 741, 421, 1233]. Train et val ont les mêmes proportions ; le test a un mélange de classes différent (adipose 18,6 % contre 10,4 %, debris 4,7 % contre 11,5 %).
  - Partitions (K = 4, graine 0), disjointes et exhaustives sur train et val :
    - `IID` : train [22500, 22496, 22500, 22500], val [2499, 2501, 2505, 2499], sha256 `609240177941c0ed…`
    - `DIR05` (α = 0,5, 1 tirage) : train [32289, 12034, 21126, 24547], val [3590, 1338, 2348, 2728], sha256 `e56860455cfc27fb…`. Fort déséquilibre (par ex. client 1 : 8 images adipose ; client 2 : 5 images debris).
  - Tests : `pytest` local → **13 réussis, 1 ignoré** (prétraitement PyTorch, torch absent). Rejoués avec succès sur un clone anonyme du dépôt public.
  - Notebook 01 : toutes les cellules de code exécutées en local (chemin non Colab), partitions régénérées identiques aux fichiers versionnés.

## Tâche en cours
Revue de la Phase 1 par Anasse (et ChatGPT).

## Prochaine tâche
1. Anasse : exécuter `notebooks/01_data.ipynb` sur Colab (« Run all », runtime CPU), vérifier « identical to versioned file: True » et « 14 passed », relever les versions affichées.
2. Après revue et approbation : Phase 2 (modèles, pilote, E1). Prérequis : D15.

## Décisions en attente
- **D15** : confirmation par l'enseignant que T4 est couvert sans CLIP/VLM (bloque la Phase 2).
- **D14** : jour exact du rendu en décembre 2026.
- **D6** : accepté provisoirement ; seuils de 2 et 5 points à confirmer avant la Phase 4.
- **D10** : seuils du budget et niveau de repli, choisis après le pilote de la Phase 2.

## Blocages connus
- Exécution Colab non faite (impossible depuis Kiro).
- Test du prétraitement PyTorch (`to_tensor`, `augment`) non exécuté en local.
- D15 non confirmée.

## Budget de calcul
Provisoire, non mesuré : environ 180 U, soit 6 à 12 h GPU selon l'hypothèse de 2 à 4 min/U. La valeur sera remplacée par la mesure du pilote (Phase 2). Niveau de repli : non choisi.

## Dernière expérience validée
Aucune expérience (E1–E7) lancée. Run de données `P1-data` : `results/phase1_data_summary.json` (local, commit 627e424).

## Décisions de projet actées
- Spécifications v0.2 approuvées le 22/11/2026, v0.3 pour la clarification de D6.
- Dépôt public `An4ss3/MediTrust`.
- Dataset : PathMNIST+ 64×64, splits officiels, test réservé à l'évaluation finale.
- Six thèmes obligatoires : T1, T4, T9, T10, T11, T12. T13 optionnel.
- Partitions de référence : `IID_k4_s0` et `DIR05_k4_s0` (hash ci-dessus), fixées pour toutes les expériences.
- Prototype de recherche : aucune revendication clinique, de confidentialité formelle ni de déploiement edge réel.

## Registre de gel (protocole, §7)
Chaque entrée est ajoutée avant l'évaluation test correspondante : date, commit, run_id, éléments gelés. Elle est recopiée dans `results/freeze.json`.

| Date | Expérience | Éléments gelés | Commit |
|---|---|---|---|
| — | — | aucune entrée | — |

## Journal des évaluations sur le test
Aucune évaluation de modèle sur le test. En Phase 1, seuls les effectifs par classe du test ont été lus (statistique descriptive, sans décision associée).

## Déviations au protocole
Aucune.

## Procédure de reprise de session
1. Lire ce fichier.
2. Consulter `ROADMAP.md` (première case non cochée de la phase courante).
3. Relire les sections pertinentes du cahier des charges et du protocole.
4. Reprendre la prochaine tâche sans refaire le travail validé.
5. En fin de session, mettre à jour ce fichier : date, travail réalisé avec preuve, tâche suivante, décisions, gels, déviations, blocages.
