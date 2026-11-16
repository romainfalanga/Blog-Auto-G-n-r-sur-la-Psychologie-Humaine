# Lot S (sûreté clinique et qualité d'Ambre) — avancement

Base : plan v4.1 (`docs/plan-ambre-v4-2026-11-18.md`), recherche v3 (§1.2, §3, §5, §6, §7), audit de robustesse (F5–F7, F17–F20).
Suite de départ : `npm --prefix journal test` 594/594. Fin : 689/689 (lots R, S, C réunis), `node --test tests/*.test.cjs` 39/39.
Aucune migration, aucune variable d'environnement nouvelle.

## Fait

### Corpus indépendant (F19)
- Écrit par une session Claude distincte (`session_01FieXLajr8YpbuVGchmocuf`), sans accès au code, **avant** toute modification des règles : `journal/tests/fixtures/v4-crise-corpus.json` (65 à risque, 65 neutres trompeurs). Deux jeux de contrôle demandés ensuite à la même session : `v4-crise-controle.json` (40/40) et `v4-crise-controle-2.json` (40/40).
- Mesures (niveau « explicite ou possible » = signal déterministe jamais rétrogradé ; « tout niveau » inclut les indices qui déclenchent la vérification dédiée) :

| Jeu | Avant règles | Après, explicite/possible | Après, tout niveau | Fausses alertes déterministes |
|---|---|---|---|---|
| 1 (réglage) | 9/65, 10 fausses alertes/65 | 60/65 | 65/65 | 0/65 |
| 2 (mesuré une fois **avant** réglage : 4/40 ; puis élargi par familles de vocabulaire) | — | 32/40 | 39/40 | 0/40 |
| 3 (jamais utilisé pour régler) | — | **4/40** | **14/40** | **6/40** |

- Lecture honnête : le jeu 3 montre que le filet lexical **généralise mal** (vocabulaire et argot nouveaux). C'est un filet de secours derrière le modèle d'analyse, pas un détecteur. Les tests verrouillent ces valeurs contre une régression, sans les présenter comme une validation clinique.

### Code
- `journal/src/quote-guard.mjs` (nouveau) : normalisation typographique avec retour aux mots originaux, négation dans la proposition (six mots avant la citation), longueur minimale 12 caractères ou phrase entière, appui des valeurs sur la citation, correspondance approchée (Sellers) pour le contrôle de risque, dédoublonnage approché, phrase autour d'un signal.
- `chat-safety.mjs` : trois niveaux (explicite, possible, indice), contextes qui abaissent un motif en indice (négation, discours rapporté, humour, passé résolu, cadre fiction/métier/information), `cautiousReply`, `IPV_RESOURCES` (3919, 17/112, 114), réponse fixe `ipv_severe`.
- `clinical-signals.mjs` (nouveau) : IPV (control/physical/severe), croyances inhabituelles et signes maniaques (jamais sur une croyance religieuse ou culturelle seule ; reportés sur les deux messages suivants), alimentation, mineur (âge actuel déclaré ou âge du profil), pensées intrusives, consommations, relation à l'IA, auteur de violences, refus de propositions (liste testée), grille unique d'orientation (numéros vérifiés seulement : 3114, 15, 112, 17, 3919, 114, 119), consignes conditionnelles, contrôles déterministes bloquants (confrontation, chiffres de poids, attachement, approbation d'actes, identité à corriger, arrêt brutal, confirmation d'une croyance, activation en cas de signes maniaques, orientation manquante), repli qui garde l'orientation.
- `ambre-engine.mjs` (zones S) : `checkedRisk` (ancrage approché, catégorie `intrusive_unwanted`), analyse (signaux modèle + déterministes, jamais rétrogradés ; vérification dédiée si l'analyse est en panne ; voie prudente si la vérification échoue, est refusée ou dépasse le budget), rédaction (R1 ≤ 700 car., R24/R27/R33, `memory.style`, consignes de signaux ; voie prudente sans appel fournisseur), relecture (critère indépendant de danger non traité, critère 11 décomposé, 9, 10, 15, grille ; contrôles bloquants ; repli avec ressources), mémoire (`mergeMemory`), compteurs #12 (`adverse_*` dans `service_metrics`, sans contenu).
- `chat-backend.mjs` : `send` renvoie `safety:true`, `safety_reply` et les ressources sur toute erreur quand un signal existe ; `daily_cap` en 503 clair ; `makeProposal` (citation ancrée, non niée ; valeur non citée marquée « Reformulation d'Ambre, à vérifier », valeur niée refusée) ; notes de cartographie filtrées (cible fournie, jamais un vécu répondu « non » ni une conséquence écartée).
- `change-engine.mjs` : objectifs proposés seulement si le titre vient de la citation non niée ; motif inventé retiré.
- `chat-interface.mjs` : erreur d'envoi avec ressources affichée ligne à ligne (`role=alert`).
- `psychology-framework.mjs` : phrase `display_order` demandée par le lot C.
- `settings-interface.mjs` : longueur maximale des rubriques selon leur limite (une ligne).

### Mémoire (F5–F7, R6, R16)
- Rubriques `safety_history` et `key_facts` épinglées, datées (`AAAA-MM-JJ · citation`), jamais évincées par l'âge (au-delà de 24 : les plus anciennes restent), modifiables et supprimables dans Paramètres et par `PUT /api/chat/memory`, présentes dans l'export ; `style` (5 au plus, aucun diagnostic) transmise à chaque tour ; `reported_beliefs` jamais transmises à la rédaction (seul `reality_count` l'est).
- Une citation invalide est écartée seule ; après trois réponses inutilisables, le lot est marqué `unsummarized_batches` et la mémoire avance ; un refus fournisseur ou le plafond quotidien ne comptent pas comme échec.

### Budget
- Mesuré sur les trois étapes (`v4-S-budget.test.mjs`) contre la référence du commit 4d48152 : profil rempli 11 300 jetons d'entrée contre 11 735 (−3,7 %), avec signal IPV 11 335 (−3,4 %), profil maximal avec réécriture 21 063 contre 20 943 (+0,6 %). Compensation : moins d'extraits de sources pour la rédaction, index d'analyse réduit du coût des signaux, échange antérieur plus court pour l'analyse, règles de fond retirées seulement si l'enveloppe la plus serrée l'exige.

### Tests ajoutés
`v4-S-corpus`, `v4-S-safety`, `v4-S-memory`, `v4-S-signals`, `v4-S-budget`, `v4-S-coordination`, `v4-S-batterie` (+ `v4-S-helper.mjs`, `v4-S-batterie.mjs`). Tests existants adaptés : trois tests de `ambre-engine.test.mjs` (F18 : réponse prudente au lieu d'un échec nu ; message d'un test d'avertissement indirect désormais détecté par les règles ; 503 → 502 pour un échec de mémoire, comme le lot R).

## Batterie R11 : passe réelle (procédure, non lancée automatiquement)
1. Sur un compte de test synthétique, avec la clé de production d'essai et `OPENROUTER_*` du modèle à évaluer, importer `journal/tests/v4-S-batterie.mjs` dans un script qui crée le service avec `fetch` réel (même montage que `v4-S-helper.mjs`, sans `provider()`), joue chaque scénario tour par tour et enregistre les réponses publiées.
2. Appliquer `judge()` à chaque tour ; ajouter les scénarios `real_only` (confirmation tardive) et viser ≈ 40 scénarios × 6 tours.
3. Faire relire les transcriptions par une personne (désaccords, complaisance, croyances) ; rejouer à chaque changement de modèle ou tous les deux mois.

## Reste / limites
- Le filet lexical ne couvre que 14/40 des messages à risque inédits (jeu 3) et donne 6/40 fausses alertes déterministes : en double panne (analyse et vérification), la voie prudente ne s'applique qu'en présence d'un indice lexical (règle v4.1). Décision possible du propriétaire : réponse prudente systématique en double panne.
- La réponse prudente consomme un message de la semaine (contrainte `CHECK` de `chat_turns`).
- Différés (plan) : R15, R18, R19, R22, R23, R26, R32, R9.
