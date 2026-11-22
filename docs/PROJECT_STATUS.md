# MediTrust — Project Status

**Dernière mise à jour :** 21/11/2026, session 2 (Kiro)

## Phase actuelle
**PHASE 0 — REQUIREMENTS VALIDATION**

## Statut
**AWAITING REVIEW AND APPROVAL** (v0.2 : architecture acceptée sur le principe, approbation finale en attente)

## Travail réalisé (avec preuve)
- Session 1 :
  - Répertoire `C:\Users\harki\Downloads\MediTrust` inspecté : vide au démarrage. Python local 3.13.9, PyTorch non installé.
  - Recherche vérifiée en ligne : métadonnées officielles de PathMNIST (`info.py`), MedMNIST+ (`on_medmnist_plus.md`, Zenodo 10519652), source NCT-CRC-HE (Zenodo 1214456), benchmarks officiels, Doerrich et al. 2025, fiche timm DeiT-Tiny, documentation de quantification ONNX Runtime, FAQ Colab.
  - Les 27 identifiants arXiv cités ont été vérifiés via l'API arXiv, et le DOI PLOS Medicine (Kather 2019) via Crossref.
  - Rédaction des 4 documents de `docs/` (v0.1).
- Session 2 : intégration des 5 corrections de revue (v0.2), avec cohérence vérifiée entre les 4 documents :
  1. **Évaluation fédérée** : distinction entre métriques additives et non linéaires ; statistiques suffisantes (protocole, §1.3).
  2. **D6** : sélection de M* sur validation propre et PGD-10 sur validation, avec contrôles d'éligibilité.
  3. **Discipline du test** : pré-enregistrement, registre de gel, `evaluate_test` verrouillé, règle pour les défauts de code, CR3 déplacé sur la validation (protocole, §7).
  4. **Budget provisoire** : niveaux de repli N0–N3 choisis après le pilote et gelés avant E1 (protocole, §9).
  5. **T4** : analyse de la fiche officielle et décision D15 (cahier des charges, §8.2).
- Aucun code, aucun téléchargement de données, aucun dépôt Git créé.

## Tâche en cours
Approbation finale de la v0.2 par Anasse (et ChatGPT).

## Prochaine tâche (après approbation)
Phase 1, première tâche : vérifier les versions sur Colab, figer `requirements.txt`, créer `.gitignore` et `src/data.py` (chargement de `pathmnist_64.npz` avec contrôle MD5 et des effectifs), puis le notebook 01 d'exploration.

## Décisions en attente
D1–D15 (voir `CAHIER_DES_CHARGES.md`, §22). Points à trancher en priorité :
- **D6** : seuils de 2 points (perte de macro-F1 propre) et de 5 points (gain d'accuracy PGD-10 à 4/255).
- **D10** : cible de ≤ 15 h GPU et seuils des niveaux N1 (15 h), N2 (25 h) et N3 (40 h).
- **D7** : couverture cible de 90 %.
- **D12** : dépôt public ou privé.
- **D14** : date de rendu.
- **D15** : confirmation par l'enseignant que T4 est couvert sans CLIP/VLM. CLIP n'est pas ajouté sans approbation explicite.

## Blocages connus
- Approbation des spécifications (bloque la Phase 1).
- Autorisation requise avant `git init`, création du dépôt GitHub et installation de PyTorch en local.
- D15 doit être confirmée avant la Phase 2.
- Date de rendu inconnue (planification en sessions uniquement).

## Budget de calcul
Provisoire, non mesuré : environ 180 U, soit 6 à 12 h GPU selon l'hypothèse de 2 à 4 min/U. La valeur sera remplacée par la mesure du pilote (Phase 2). Niveau de repli : non choisi.

## Dernière expérience validée
Aucune.

## Décisions de projet actées
Aucune décision technique n'est encore formellement approuvée. Ont été acceptés sur le principe lors de la revue du 21/11/2026 : l'architecture globale, les six thèmes, PathMNIST+ 64×64, ResNet-18 contre DeiT-Tiny, FedAvg à 4 clients simulés et le pipeline expérimental. Faits fixés par le cadrage initial :
- Dataset principal : PathMNIST (MedMNIST v2/v3), splits officiels, test réservé à l'évaluation finale.
- Six thèmes obligatoires : T1, T4, T9, T10, T11, T12. T13 optionnel.
- Prototype de recherche : aucune revendication clinique, de confidentialité formelle ni de déploiement edge réel.

## Registre de gel (protocole, §7)
Chaque entrée est ajoutée avant l'évaluation test correspondante : date, commit, run_id, éléments gelés. Elle est recopiée dans `results/freeze.json`.

| Date | Expérience | Éléments gelés | Commit |
|---|---|---|---|
| — | — | aucune entrée | — |

## Journal des évaluations sur le test
Aucune évaluation. Le journal détaillé sera tenu dans `results/test_log.csv`.

## Déviations au protocole
Aucune.

## Procédure de reprise de session
1. Lire ce fichier.
2. Consulter `ROADMAP.md` (première case non cochée de la phase courante).
3. Relire les sections pertinentes du cahier des charges et du protocole.
4. Reprendre la prochaine tâche sans refaire le travail validé.
5. En fin de session, mettre à jour ce fichier : date, travail réalisé avec preuve, tâche suivante, décisions, gels, déviations, blocages.
