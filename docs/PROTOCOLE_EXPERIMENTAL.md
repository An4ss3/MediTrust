# MediTrust — Protocole expérimental

**Version :** 0.3 (v0.2 approuvée le 22/11/2026 ; clarification de D6 en E4)
**Date :** 21 novembre 2026
**Référence :** `CAHIER_DES_CHARGES.md` (objectifs OS1–OS9, questions QR0–QR7, décisions D1–D15)

Les valeurs marquées « initial » sont des points de départ. Elles seront confirmées par le pilote de la Phase 2 sur la **validation uniquement**, puis gelées dans `PROJECT_STATUS.md` avant les runs définitifs. Les hypothèses ci-dessous sont des attentes à tester, pas des résultats. Toutes les durées GPU sont provisoires jusqu'au pilote (§9).

---

## 1. Données et partitions

### 1.1 Splits officiels

| Split | Effectif | Origine | Usage autorisé |
|---|---|---|---|
| train | 89 996 | NCT-CRC-HE-100K (90 %) | Apprentissage, calibration de la quantification statique, Fisher (EWC) |
| val | 10 004 | NCT-CRC-HE-100K (10 %, mêmes lames que le train) | Choix du lr, sélection d'architecture (D3), choix du modèle final (D6 : propre et PGD), température T, seuil τ, λ EWC, méthode de calibration INT8, suivi PGD |
| test | 7 180 | CRC-VAL-HE-7K (50 patients distincts) | Évaluation finale uniquement, après gel des décisions (§7) ; jamais utilisé pour régler une expérience |

Fichier : `pathmnist_64.npz` (Zenodo 10519652, MD5 `55aa9c1e0525abe5a6b9d8343a507616`). Le MD5 et les effectifs sont vérifiés à chaque chargement.

### 1.2 Partitions fédérées (K = 4)

| ID | Construction | Graine |
|---|---|---|
| `IID` | Pour chaque classe, indices mélangés puis répartis en 4 parts égales | 0 |
| `DIR05` | Pour chaque classe c, tirage p_c ~ Dirichlet(0,5 · 1₄), puis répartition des indices de c selon p_c (Hsu et al., 2019). Le tirage est refait si un client a moins de 2 000 images de train, et le nombre de tirages est journalisé | 0 |

- Une même matrice de proportions p_{k,c} sert à répartir le **train** et la **validation** officiels. Chaque client possède ainsi un train local et une validation locale de même distribution, et l'union des validations locales est exactement la validation officielle.
- Le **test** n'est jamais partitionné.
- Les indices sont sauvegardés dans `results/partitions/{IID,DIR05}_k4_s0.npz` (versionnés). Une figure (heatmap clients × classes) et un tableau des effectifs sont produits.
- Les partitions sont identiques pour tous les seeds d'entraînement, afin d'isoler la variance due à l'entraînement.
- Tests automatiques : les parts sont disjointes, leur union couvre tout le split, et les effectifs sont conformes.

### 1.3 Évaluation fédérée sur la validation

Notation : le client k détient n_k images de validation, et n = Σ_k n_k = 10 004.

**Métriques additives.** Ce sont des moyennes de pertes ou d'indicateurs par échantillon : accuracy, NLL, Brier, accuracy sous attaque FGSM/PGD. Pour elles, m_global = Σ_k n_k · m_k / n **exactement**. La moyenne pondérée des valeurs locales est donc valide.

**Métriques non linéaires.** Macro-F1, ECE, AUROC et quantiles de confiance ne sont **pas** égaux, en général, à la moyenne pondérée de leurs valeurs locales.
- Le F1 d'une classe est un rapport de comptes. Il est indéfini chez un client qui n'a pas cette classe, situation fréquente en DIR05.
- L'ECE et l'AUROC dépendent de la distribution conjointe des scores de tous les clients.

**Règle.** Chaque client renvoie des **statistiques suffisantes additives**. Le serveur les somme, puis calcule la métrique à partir des sommes.

| Métrique | Statistique renvoyée par chaque client | Calcul serveur | Exactitude |
|---|---|---|---|
| accuracy, NLL, Brier, accuracy sous attaque | somme sur ses exemples, n_k | somme globale / n | exacte |
| macro-F1, rappel et précision par classe | matrice de confusion 9×9 (comptes) | F1 par classe sur la matrice sommée, moyenne sur les 9 classes | exacte |
| ECE (15 bins) | par bin : nombre d'exemples, somme des confiances, nombre de corrects | ECE à partir des bins sommés | exacte |
| NLL(T) pour le temperature scaling | somme des NLL pour chaque T d'une grille fixe | argmin sur la grille | exacte sur la grille |
| AUROC macro OvR | pour chaque classe c, histogrammes à 1 000 bins sur [0, 1] du score p_c, séparément pour les exemples positifs et négatifs | AUROC par classe par comptage des paires entre bins (paires d'un même bin comptées 1/2), puis moyenne | approchée ; borne d'erreur (½ × proportion de paires positif/négatif dans un même bin) calculée et rapportée |
| quantile τ | histogramme à 10 000 bins des confiances calibrées | quantile sur l'histogramme cumulé | approchée à 1e-4 près |

- Une seule fonction `metrics_from_stats` sert au centralisé (K = 1) et au fédéré. Aucun chemin de code séparé n'existe donc pour les deux cas.
- Test pytest sur des prédictions aléatoires réparties sur 4 clients :
  - métriques exactes : valeur agrégée égale à la valeur calculée sur les prédictions regroupées, à 1e-9 près ;
  - AUROC et τ : écart dans la borne annoncée ;
  - contre-exemple vérifiant que la moyenne pondérée des macro-F1 locales diffère de la macro-F1 globale.
- L'AUROC de validation n'intervient dans aucune décision. Sur le **test**, qui n'est pas partitionné et reste détenu par l'évaluateur, toutes les métriques sont calculées exactement à partir des prédictions.
- Métriques par client (E2, dispersion inter-clients) : accuracy locale, et macro-F1 locale calculée sur les seules classes présentes chez le client. Ce choix est signalé dans les tableaux.
- Limite : les statistiques renvoyées révèlent des informations agrégées sur chaque client, par exemple sa distribution de classes via la matrice de confusion. Aucune garantie formelle de confidentialité n'est revendiquée.

## 2. Prétraitement et augmentation

| Étape | Train | Val / Test / Application |
|---|---|---|
| Conversion | uint8 → float32 dans [0, 1] | identique |
| Augmentation | flip H (p = 0,5), flip V (p = 0,5), rotation k·90° (k uniforme) | aucune |
| Dans le modèle (`MediTrustModel`) | upsampling bilinéaire 64 → 224, normalisation ImageNet | identique |

Pas d'augmentation couleur, de label smoothing ni de mixup (D13). Les données sont conservées en uint8 en mémoire (1,1 Go pour le train), et l'augmentation est appliquée sur des tenseurs, ce qui évite le goulot CPU de Colab.

## 3. Modèles

| ID | Backbone timm | Paramètres | Méthode XAI | Quantification |
|---|---|---|---|---|
| `R18` | `resnet18.tv_in1k` | ≈ 11,7 M | Grad-CAM (dernier bloc `layer4`) | statique QDQ S8S8 per-channel |
| `DEIT` | `deit_tiny_patch16_224.fb_in1k` | 5,7 M | attention brute par tête, rollout, entropie | dynamique (MatMul/Gemm) |

La tête de classification est remplacée par une couche linéaire à 9 sorties. Le fine-tuning est complet. Pour T1, l'export ONNX et l'extraction des poids d'attention, l'attention fusionnée de timm est désactivée.

## 4. Configuration d'entraînement commune

| Paramètre | Valeur (initiale) | Remarque |
|---|---|---|
| Optimiseur | SGD, momentum 0,9, weight decay 1e-4 | Identique en centralisé et en local FL. L'état du momentum est réinitialisé à chaque round FL |
| lr | choisi parmi {0,003 ; 0,01 ; 0,03} par le pilote (§4.1), par architecture | |
| Schedule | cosinus, constant pendant une époque (centralisé) ou un round (FL) | Schedules identiques quand R = époques et E_local = 1 |
| Époques / rounds | 10 | Confirmé par le pilote (durée) |
| Batch | 128 | |
| Précision | AMP fp16 | Le T4 ne gère pas bf16 |
| Clipping du gradient | norme globale 1,0 | |
| Checkpoint retenu | dernier (fin du schedule), sans early stopping | Même règle pour tous ; évite de surexploiter une validation au plafond |
| Checkpoint de reprise | à chaque époque ou round : poids, optimiseur, numéro d'époque ou de round, états RNG, historique | |

### 4.1 Pilote (Phase 2, seed 0)
- Chronométrage d'une époque pour chaque architecture (t_U en min/U, médiane de 2 époques), et mesure de la mémoire GPU maximale. Le type de GPU obtenu est journalisé.
- Choix du lr : 2 époques par valeur et par architecture, critère = macro-F1 de validation (NLL en cas d'égalité à 0,1 point près). Coût : environ 12 unités.
- Test d'interruption et de reprise.
- Le lr retenu pour l'architecture M est réutilisé tel quel en FL, pour la comparabilité.
- **Projection du budget** avec les t_U mesurés et choix du niveau de repli N0–N3 (§9). Ce choix est gelé avant les runs E1.

## 5. Expériences

Unité de coût : **U = une passe d'entraînement complète sur les 89 996 images à 224** pour l'un des deux modèles. Hypothèse provisoire, non mesurée : 2 à 4 min par U sur GPU T4. Elle sera remplacée par la valeur mesurée au pilote. Les coûts en U ci-dessous valent pour le niveau N0 (§9).

---

### E1 — CNN vs ViT (T4)

| Rubrique | Contenu |
|---|---|
| Question | QR1 |
| Hypothèse H1 | DeiT-Tiny et ResNet-18 ont une performance de test comparable (écart de macro-F1 < 2 points), conformément à Doerrich et al. (2025). Le classement n'est pas présumé. |
| Modèles | `R18`, `DEIT`, pré-entraînés, centralisé |
| Données | train (entraînement), val (lr, sélection D3), test (évaluation finale après gel de D3) |
| Configuration | §4, 10 époques au niveau N0 (6 aux niveaux N2/N3, §9), seeds {0, 1, 2}, lr issu du pilote |
| Baseline | `R18` (référence officielle MedMNIST) |
| Métriques | accuracy, macro-F1, AUROC macro one-vs-rest, matrice de confusion, rappel par classe ; paramètres ; taille ONNX FP32 (Mo) ; latence CPU batch 1 (mesurée avec le banc E5, colonne complétée en Phase 6) ; durée par époque ; mémoire GPU maximale |
| Analyse | moyenne ± écart-type sur 3 seeds ; IC bootstrap à 95 % (1 000 rééchantillonnages appariés du test) de Δ macro-F1 ; écart validation/test par modèle |
| Décision produite | modèle de pipeline M selon D3, consignée avant toute évaluation test |
| Artefacts | `results/E1/*.json`, courbes d'apprentissage, matrices de confusion, tableau E1 ; checkpoints Drive `E1-{R18,DEIT}-s{0,1,2}` |
| Critères de réussite | 6 runs complets ; contrôle de plausibilité CR3 réalisé **sur validation avant tout accès au test** (accuracy de validation de R18 ≥ 0,90, seuil fixé a priori, rappel non nul pour chaque classe, loss convergente) ; comparaison au benchmark officiel (test 0,907–0,909) rapportée à titre descriptif seulement ; décision D3 gelée |
| Calcul | 2 modèles × 3 seeds × 10 U = 60 U, plus pilote 12 U |

---

### E2 — Centralisé vs fédéré (T10)

| Rubrique | Contenu |
|---|---|
| Question | QR2 |
| Hypothèses | H2a : FedAvg-IID atteint une macro-F1 de test proche du centralisé (écart < 1 point). H2b : FedAvg-DIR05 est inférieur au centralisé et converge plus lentement. H2c : FedAvg-DIR05 dépasse la moyenne des modèles locaux isolés. H2d : l'écart de performance entre clients est plus grand en DIR05 qu'en IID. |
| Modèle | architecture M (issue d'E1), même initialisation pré-entraînée |
| Données | partitions `IID` et `DIR05` (train et val locales), test global |
| Conditions | voir tableau ci-dessous |
| Métriques | métriques E1 sur le test ; courbe macro-F1 de validation par round (évaluation fédérée, §1.3) ; métriques par client sur sa validation locale (min, max, écart-type ; définitions au §1.3) ; volume communiqué = 2 × K × taille(M) × R |
| Analyse | Δ par rapport au centralisé (moyenne ± écart-type, IC bootstrap) ; courbes de convergence ; dispersion inter-clients |
| Artefacts | `results/E2/*.json`, heatmaps de partition, courbes, tableau E2 ; checkpoints Drive |
| Critères de réussite | toutes les conditions exécutées ; agrégation FedAvg validée par un test unitaire (moyenne pondérée exacte des poids) ; tests de `metrics_from_stats` (§1.3) verts : exactitude pour les métriques additives et la macro-F1 par matrice de confusion, borne respectée pour AUROC et τ |

| ID condition | Description | Seeds | Coût |
|---|---|---|---|
| `E2-CENT` | Centralisé : checkpoints E1 de M **réutilisés** (même configuration) | 0, 1, 2 | 0 U |
| `E2-IID` | FedAvg, IID, R = 10, E_local = 1 | 0, 1, 2 | 30 U |
| `E2-DIR05` | FedAvg, DIR05, R = 10, E_local = 1 (**scénario de référence**, M_FL) | 0, 1, 2 | 30 U |
| `E2-LOCAL` | 4 modèles locaux, chacun sur son train DIR05, 10 époques | 0 | ≈ 10 U au total |
| `E2-ABL` | FedAvg, DIR05, R = 5, E_local = 2 (calcul égal, deux fois moins de communication) | 0 | 10 U |

Contrôle d'équité : même initialisation, même optimiseur, même lr et même schedule. Même nombre total de passages par image (10, ou le nombre fixé par le niveau de repli). Même règle de checkpoint (dernier round). Même évaluation. `E2-ABL` est une ablation descriptive : son résultat ne modifie pas la configuration de référence (D4), déjà gelée.

---

### E3 — Calibration et revue experte (T9)

| Rubrique | Contenu |
|---|---|
| Question | QR3 |
| Hypothèses | H3a : le temperature scaling réduit ECE et NLL sur la validation, sans changer l'accuracy (l'argmax est invariant). H3b : la réduction est moindre sur le test, à cause du décalage patients et centres (Ovadia et al., 2019). H3c : M_FL-AT a une ECE avant calibration plus faible que M_FL (Grabinski et al., 2022). |
| Modèles (post-hoc, **aucun entraînement**) | E1 `R18` et `DEIT` (seed 0) ; E2 `CENT`, `IID`, `DIR05` (seed 0) ; E4 `M_FL-AT`. Les versions INT8 sont calibrées avec le même code, dans E5 (Phase 6) |
| Méthode | T scalaire choisi par minimisation de la NLL de validation sur une grille fixe de 400 valeurs log-espacées dans [0,25 ; 8] (pas relatif ≈ 0,9 %). La NLL étant additive, la même procédure est exacte en fédéré : chaque client renvoie la somme de ses NLL pour chaque T (§1.3). Les modèles centralisés utilisent la même fonction avec K = 1 |
| Seuil | τ = quantile 10 % des confiances calibrées (max softmax) sur la validation de M* (D7), calculé sur l'histogramme agrégé (§1.3) |
| Métriques | ECE (15 bins de largeur égale, top-label), NLL, Brier multiclasse, accuracy ; reliability diagrams ; courbe risk-coverage et AURC (Geifman & El-Yaniv, 2017) ; pour τ : couverture, accuracy sélective, erreur des cas acceptés et signalés, sur validation et test |
| Artefacts | `results/E3/*.json`, reliability diagrams avant et après, courbes risk-coverage, `calibration.json` (T, τ) pour M* |
| Critères de réussite | T et τ calculés sur la validation seulement ; tableau avant/après sur validation et test pour toutes les variantes |
| Calcul | inférence seulement, < 1 U |

---

### E4 — Robustesse adversariale (T11)

| Rubrique | Contenu |
|---|---|
| Question | QR4 |
| Hypothèses | H4a : l'accuracy de M_FL chute fortement quand ε augmente, plus sous PGD que sous FGSM. H4b : M_FL-AT améliore l'accuracy robuste (FGSM et PGD) à ε ≤ 4/255, au prix d'une baisse d'accuracy propre. H4c : une relation inattendue entre FGSM et PGD sur M_FL-AT est un avertissement possible de masquage de gradient, à investiguer par une attaque renforcée, sans conclusion automatique. |
| Modèles | M_FL (= `E2-DIR05`, seed 0) ; M_FL-AT (nouvel entraînement) ; contexte : `R18` et `DEIT` d'E1 (courbe FGSM seulement) |
| Menace | boîte blanche, L∞, non ciblée, sur l'image 64×64 dans [0, 1], clipping dans [0, 1], gradients propagés à travers l'upsampling et la normalisation |
| Entraînement M_FL-AT | FedAvg DIR05, R = 10, E_local = 1, même lr et même initialisation que M_FL. Chaque client remplace chaque batch par son exemple FGSM-RS : δ₀ ~ U(−ε, ε), pas α = 1,25·ε, ε_train = 4/255 (Wong et al., 2020). Suivi PGD-10 à ε_train sur 1 000 images de validation à chaque round. En cas d'effondrement (catastrophic overfitting), le meilleur round selon PGD sur validation est retenu et le fait est documenté. |
| Attaques d'évaluation | FGSM (sans départ aléatoire) ; PGD-10, pas ε/4, départ aléatoire, 1 restart ; ε ∈ {0, 1, 2, 4, 8}/255 ; contrôle à 16/255 |
| Comparaisons demandées | (1) standard / propre, (2) standard / adversarial, (3) AT / propre, (4) AT / adversarial |
| Métriques | accuracy et macro-F1 propres et sous attaque par ε ; confiance moyenne sur les exemples adversariaux ; ECE propre (reprise dans E3) |
| Contrôles de cohérence (Carlini et al., 2019) | accuracy PGD ≤ accuracy FGSM ; décroissance monotone avec ε ; accuracy proche de 0 à 16/255 pour le modèle standard. Un contrôle échoué est investigué avant toute conclusion |
| Évaluation sur validation pour D6 | sur la validation officielle complète, par évaluation fédérée exacte (§1.3), pour M_FL et M_FL-AT : macro-F1 propre ; accuracy FGSM à 4/255 ; accuracy PGD-10 à 2, 4 et 8/255 (graine d'attaque 0) |
| Décision produite | M* selon D6. Contrôles d'alerte sur la validation : PGD ≤ FGSM à 4/255 et accuracy PGD décroissante sur {2, 4, 8}/255. Une alerte déclenche un PGD renforcé (50 itérations, 5 départs aléatoires, pire cas par image) sur les deux modèles, sans rejet automatique ; seule une alerte persistante après PGD renforcé rend M_FL-AT non fiable pour D6 (coût supplémentaire éventuel ≈ 25 U, pris sur la marge de 30 %). M* = M_FL-AT si aucune alerte non résolue, si la baisse de macro-F1 propre est ≤ 2 points et si le gain d'accuracy PGD-10 à 4/255 est ≥ 5 points. Sinon M* = M_FL. Gel dans le registre **avant** toute évaluation test d'E4 |
| Artefacts | `results/E4/*.json`, courbes accuracy/ε (FGSM, PGD), exemples adversariaux visualisés, checkpoint Drive `E4-FLAT-DIR05-s0` |
| Critères de réussite | 4 comparaisons rapportées sur toute la grille ; contrôles de cohérence documentés ; aucune formulation de robustesse au-delà des attaques testées |
| Calcul | entraînement ≈ 2 × 10 U = 20 U ; évaluation PGD sur la validation (2 modèles) ≈ 2,5 U ; évaluation PGD sur le test < 2 U |

---

### E5 — Optimisation Edge (T12)

| Rubrique | Contenu |
|---|---|
| Question | QR5 |
| Hypothèses | H5a : l'INT8 réduit la taille du fichier d'environ 3–4× (poids 32 → 8 bits, hors surcoûts). H5b : la perte de macro-F1 de test est < 1 point pour `R18` en statique. H5c : le gain de latence dépend de l'architecture et du CPU, et peut être nul sans VNNI. H5d : l'accord des prédictions FP32/INT8 est > 99 %. |
| Modèles (post-hoc) | M* (principal) ; matrice de compatibilité : `R18` et `DEIT` d'E1 (seed 0) × {statique, dynamique} |
| Procédure | export ONNX (opset récent, Resize inclus, batch dynamique) → prétraitement ONNX Runtime (`quant_pre_process`) → quantification. Statique : 512 images du **train** stratifiées (graine 0), MinMax ; si la perte en validation dépasse 1 point, essai de Percentile, choix sur la validation. Dynamique : poids INT8 |
| Parité | ONNX FP32 vs PyTorch sur la validation : accord argmax 100 %, écart max des logits < 1e-3 |
| Conditions de mesure | même machine, même version d'ONNX Runtime, CPUExecutionProvider ; batch 1 ; `intra_op_num_threads` ∈ {1, défaut} ; 50 itérations de chauffe puis 500 mesures ; médiane et p95 ; journalisation du modèle de CPU et des drapeaux (AVX2, AVX-512, VNNI) ; deux plateformes : runtime CPU Colab et PC local |
| Métriques | taille (Mo) ; latence p50/p95 ; accuracy, macro-F1, AUROC sur le test ; ECE avec T_FP32 puis avec T_INT8 réajusté sur la validation ; accord FP32/INT8 ; couverture au seuil τ |
| Artefacts | `results/E5/*.json`, tableau de compatibilité, `final_fp32.onnx`, `final_int8.onnx` (Drive) |
| Critères de réussite | CR7 ; mesures FP32 et INT8 dans des conditions identiques ; échecs de quantification documentés le cas échéant |
| Calcul | CPU uniquement |

---

### E6 — Explicabilité et attention (T9, T1)

| Rubrique | Contenu |
|---|---|
| Question | QR6 |
| Hypothèses | H6a : supprimer les 25 % de pixels les plus pertinents fait baisser la probabilité de la classe prédite davantage que supprimer 25 % de pixels aléatoires, pour les deux méthodes. H6b : les cartes d'un modèle aux poids randomisés sont faiblement corrélées (Spearman) à celles du modèle entraîné. H6c : l'entropie d'attention diminue avec la profondeur (attention plus concentrée). Hypothèse exploratoire, sans direction attendue : l'entraînement adversarial modifie les cartes et les scores. |
| Modèles (post-hoc) | `R18` (Grad-CAM) et `DEIT` (attention brute, rollout) d'E1, seed 0 ; variantes du pipeline M_FL et M_FL-AT avec la méthode de leur architecture |
| T1, partie mécanisme | sur un bloc de DEIT : recalcul explicite softmax(QKᵀ/√d_h)V à partir des projections qkv, comparé à la sortie timm (écart max < 1e-5, test pytest) ; cartes CLS→patchs pour les 3 têtes × 12 couches ; entropie moyenne par tête et par couche sur 900 images de test |
| Rollout | moyenne des têtes, ajout de l'identité (0,5·A + 0,5·I), renormalisation des lignes, produit sur les couches, ligne CLS → grille 14×14 → upsampling 64×64. Paramètres fixés a priori, sans réglage |
| Grad-CAM | sortie de `layer4` (7×7) pour la classe prédite, ReLU, upsampling 64×64 |
| Test de suppression | 900 images de test (100 par classe, graine 0). Remplacement de la fraction f ∈ {10, 25, 50} % des pixels les plus pertinents par la couleur moyenne de l'image, comparé à la même fraction aléatoire (5 tirages). Métrique : baisse moyenne de la probabilité calibrée de la classe initialement prédite, et accuracy après suppression |
| Sanity check | randomisation complète des poids (Adebayo et al., 2018) ; corrélation de Spearman entre les cartes avant et après sur 100 images |
| Qualitatif | planche fixe de 2 images par classe (18) × méthodes × variantes, incluant des erreurs et des cas signalés pour revue |
| Artefacts | `results/E6/*.json`, planches de figures, courbes d'entropie |
| Critères de réussite | CR8 ; limites interprétatives rédigées (attention ≠ causalité, rollout non spécifique à la classe, artefacts du masquage) ; aucune affirmation de validité médicale |
| Calcul | < 2 U (le sanity check et la suppression ne demandent que de l'inférence) |

---

### E7 — Apprentissage continu (T13, optionnel, D9)

Exécutée seulement après un go explicite à la fin de la Phase 7.

| Rubrique | Contenu |
|---|---|
| Question | QR7 |
| Hypothèse H7 | Le fine-tuning naïf sur l'établissement B dégrade la performance sur A (oubli). EWC réduit cet oubli au prix d'une moindre performance sur B. |
| Scénario | transfert séquentiel du modèle entre deux établissements sans fédération : tâche A = client 1 de DIR05, tâche B = client 2 de DIR05 (distributions de classes différentes) |
| Modèle de départ | modèle local du client 1 issu d'`E2-LOCAL` (**réutilisé**) |
| Conditions | naïf : fine-tuning sur B, 3 époques ; EWC (Kirkpatrick et al., 2017) : Fisher diagonale sur le train A, λ ∈ {1, 10, 100, 1000} × échelle choisie sur la validation (moyenne des validations locales A et B) |
| Métriques | accuracy et macro-F1 sur la validation locale A avant et après (oubli = différence) ; idem sur B (plasticité) ; macro-F1 sur le test global |
| Calcul | ≈ 5 U |

---

## 6. Tableau de bord final (QR0)

Sur le **test**, pour les variantes `E2-CENT`, `E2-IID`, `E2-DIR05` (M_FL), `M_FL-AT` et M* en INT8 :

| Dimension | Indicateur |
|---|---|
| Performance | accuracy, macro-F1, AUROC |
| Décentralisation | images transférées (0 en FL), volume de poids communiqué |
| Fiabilité | ECE et NLL après TS, AURC, couverture et accuracy sélective à τ |
| Robustesse | accuracy PGD-10 à 2/255 et 4/255 |
| Interprétabilité | baisse de probabilité au test de suppression à 25 %, Spearman du sanity check |
| Efficacité | taille (Mo), latence p50 sur CPU local |

Ce tableau répond à QR0. Il est construit uniquement à partir des JSON de runs.

## 7. Discipline du test et prévention des fuites

### 7.1 Règles

1. Seuls les splits officiels sont utilisés. Il n'y a aucun échange entre train, val et test.
2. **Pré-enregistrement.** Après approbation, les configurations d'E1 à E7 sont celles de ce protocole. Aucun résultat de test ne sert à régler une expérience, qu'elle soit en cours ou ultérieure : hyperparamètres, seuils, choix de modèle, conditions, grilles. Une modification d'une expérience ultérieure n'est admise que si elle est justifiée par la validation, par le coût mesuré ou par un défaut de code. Elle est consignée comme déviation, avec sa date, dans `PROJECT_STATUS.md` **avant** ses runs.
3. **Registre de gel.** Avant chaque évaluation test, les éléments listés au §7.2 sont enregistrés dans le registre de gel de `PROJECT_STATUS.md` (date, commit, run_id) et dans `results/freeze.json`.
4. **Verrou du test.** Le test n'est lu que par `evaluate_test(run_id)`. Cette fonction :
   - refuse de s'exécuter si le run_id n'est pas présent dans `results/freeze.json` ;
   - enregistre chaque appel (date, run, commit) dans `results/test_log.csv`.
5. Toutes les décisions sont prises sur la validation : lr, niveau de budget, D3, D6, T, τ, méthode de calibration INT8, λ EWC, round retenu en cas de catastrophic overfitting.
6. **Défaut de code révélé après une évaluation test** (par exemple un désalignement d'étiquettes ou un prétraitement erroné) : seul le défaut est corrigé, et aucun hyperparamètre ni choix de conception ne change. Le résultat initial reste dans `test_log.csv`. Toutes les évaluations affectées sont refaites et les deux résultats sont rapportés comme déviation.
7. La normalisation utilise des constantes ImageNet, sans statistiques calculées sur les données.
8. La calibration de la quantification statique et la Fisher d'EWC utilisent le train.
9. Les partitions FL ne portent que sur train et val.
10. Limite connue et non corrigeable : train et val partagent les mêmes lames (split aléatoire au niveau patch). La validation est donc optimiste. Cette limite est assumée et analysée (écart validation/test).
11. Les images d'exemple de l'application ne sont utilisées qu'après l'évaluation finale.

### 7.2 Éléments à geler avant chaque évaluation test

| Évaluation test | Éléments gelés au préalable | Phase |
|---|---|---|
| E1 | lr par architecture, niveau de repli N, nombre d'époques, 6 checkpoints E1 (dernière époque), résultat du contrôle CR3 sur validation, choix de M (D3) | 2 |
| E2 | M, partitions (hash), configurations FL, checkpoints de toutes les conditions | 3 |
| E4 | checkpoint M_FL-AT (avec le round retenu s'il y a eu catastrophic overfitting), paramètres d'attaque, grille ε, mesures de validation et choix de M* (D6) | 4 |
| E3 | T de chaque variante, τ (D7), bins de l'ECE | 5 |
| E6 | couche Grad-CAM, paramètres du rollout, fractions de suppression, échantillon de 900 images (graine), graine de randomisation | 5 |
| E5 | méthode de quantification, méthode de calibration (MinMax ou Percentile), T_INT8, paramètres du banc de latence | 6 |
| E7 (si go) | λ, nombre d'époques | 8 |
| Tableau de bord final | tous les gels précédents ; aucun nouveau calcul de décision | 7 |

## 8. Reproductibilité

- **Graines** : `random`, `numpy`, `torch` (CPU et CUDA), générateur des batchs et des augmentations ; graines d'entraînement {0, 1, 2} ; graine de partition 0.
- **Déterminisme** : best-effort (`cudnn.benchmark = False`). Des différences mineures entre exécutions GPU restent possibles, ce qui justifie l'usage de plusieurs seeds.
- **JSON de run** (`results/<E>/<run_id>.json`) : run_id, expérience, question de recherche, date, commit Git, configuration complète, graine, versions (Python, torch, timm, CUDA, onnxruntime, medmnist), GPU ou CPU, durée, mémoire maximale, MD5 du dataset, hash de la partition, métriques, chemins Drive des artefacts.
- **Identifiants** : `E1-R18-s0`, `E1-DEIT-s1`, `E2-IID-s2`, `E2-DIR05-s0`, `E2-LOCAL-DIR05-c3`, `E2-ABL-DIR05-e2r5`, `E4-FLAT-DIR05-s0`, `E5-<modèle>-<méthode>-<plateforme>`, `E6-<modèle>`, `E7-<naive|ewc>-<λ>`.
- **Notebooks** : un par phase. Chacun affiche les versions et le matériel en tête, appelle `src/` et ne contient aucune logique dupliquée.

## 9. Budget de calcul (provisoire jusqu'au pilote)

Les heures ci-dessous ne sont **pas des mesures**. Elles appliquent l'hypothèse de 2 à 4 min/U au niveau N0. Le type de GPU du Colab gratuit n'est pas garanti et ses limites d'usage ne sont pas publiées.

| Poste (niveau N0) | Unités U | Heures GPU (hypothèse 2–4 min/U) | Réutilisation |
|---|---|---|---|
| Pilote (lr, chronométrage) | 12 | 0,4–0,8 | — |
| E1 | 60 | 2–4 | — |
| E2 | 80 | 2,7–5,3 | `E2-CENT` réutilise E1 |
| E3 | < 1 | < 0,1 | tous les checkpoints |
| E4 | ≈ 25 | 0,8–1,7 | M_FL d'E2 |
| E5 | 0 (CPU) | — | M*, E1 |
| E6 | < 2 | < 0,2 | E1, E2, E4 |
| **Total obligatoire** | **≈ 180** | **≈ 6–12 (hypothèse)** | |
| E7 (optionnel) | ≈ 5 | 0,2–0,4 | `E2-LOCAL` |

### 9.1 Procédure après le pilote

1. Mesurer t_U pour chaque architecture.
2. Calculer la projection P = Σ (U × t_U de l'architecture concernée) × 1,3. Le facteur 1,3 est une marge pour les reprises et les évaluations.
3. Choisir le niveau de repli selon P :

| Niveau | Condition (proposée, D10) | Mesures | Unités ≈ |
|---|---|---|---|
| N0 | P ≤ 15 h | plan complet | 180 |
| N1 | 15 h < P ≤ 25 h | suppression d'`E2-ABL` ; `E2-IID` réduit à 1 seed | 150 |
| N2 | 25 h < P ≤ 40 h | N1, plus 6 époques ou rounds au lieu de 10 pour **tous** les entraînements (E1, E2, E4), ce qui préserve l'équité | 100 |
| N3 | P > 40 h | N2, plus entrée à 112×112 pour les deux modèles (interpolation des position embeddings de DeiT de 14×14 à 7×7). Modifie D1 : **approbation obligatoire** et nouveau chronométrage | à mesurer |

4. Geler le niveau choisi avant les runs E1. Il s'applique à toutes les expériences.
5. Le nombre d'époques et la résolution ne changent plus ensuite, car cela romprait la comparabilité avec E1 et `E2-CENT`.
6. Après E1, la projection est recalculée avec l'architecture M. Seules les suppressions de conditions de N1 peuvent encore être appliquées, avec consignation.
7. Les interruptions Colab sont absorbées par la reprise sur checkpoint (sessions de ≤ 2–3 h) et ne justifient pas un changement de configuration.

## 10. Analyse statistique

- Comparaisons principales (E1, E2) : moyenne ± écart-type sur 3 seeds, et IC à 95 % par bootstrap apparié sur le test (1 000 rééchantillonnages) pour les écarts de macro-F1 et d'accuracy.
- E4 et E5 : un seed. Les IC bootstrap portent sur la variabilité du test seulement, pas sur celle de l'entraînement, et ce point est signalé.
- On ne rapporte pas de « gagnant » lorsque l'IC de l'écart contient 0.

## 11. Correspondance objectifs ↔ expériences ↔ thèmes

| Objectif | Question | Expérience | Thème |
|---|---|---|---|
| OS1 | QR1 | E1 | T4 |
| OS2 | QR6 | E6 (partie mécanisme) | T1 |
| OS3 | QR2 | E2 | T10 |
| OS4 | QR3 | E3 | T9 |
| OS5 | QR6 | E6 | T9, T1 |
| OS6 | QR4 | E4 | T11 |
| OS7 | QR5 | E5 | T12 |
| OS8 | QR0 | Tableau de bord et application | Tous |
| OS9 | QR7 | E7 | T13 |
