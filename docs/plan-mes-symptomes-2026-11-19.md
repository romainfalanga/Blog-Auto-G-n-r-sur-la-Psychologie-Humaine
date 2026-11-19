# Plan « Mes symptômes » et moments anecdotiques / marquants (19 novembre 2026)

Sources : `docs/recherche-mes-symptomes-2026-11-19.md`, `editorial/symptomes/proposition-2026-11-19.json`. Décision du propriétaire : chaque moment de la ligne de vie est « anecdotique » ou « marquant ».

## 1. Moments anecdotiques / marquants
- Nouveau champ `significance` (`anecdotal` | `major`) sur les moments, obligatoire à la création et à la modification (formulaire Ma ligne de vie et ajout depuis la page d'un proche) ; anciens moments : valeur nulle affichée « à préciser », jamais inventée.
- Affichage : moments marquants mis en avant sur la frise (point plein, titre plus fort), anecdotiques plus discrets ; filtre facultatif « Tous / Marquants ».
- Ambre et bilans : les moments marquants priorisés dans l'index et les détails ; budget inchangé.
- Migration `0027` additive (colonne nullable avec contrôle), Worker tolérant avant migration, export et suppression.

## 2. Page « Mes symptômes » (`/mes-symptomes/`, dans le menu après Mon enfance)
**Ce n'est jamais un diagnostic** ; le texte le dit en une phrase, et chaque résultat renvoie vers un professionnel.

**Instruments retenus (licence libre vérifiée)** : PHQ-2 → PHQ-9 (humeur), GAD-2 → GAD-7 (anxiété), PHQ-15 (symptômes physiques), ITQ (stress post-traumatique et TSPT complexe, CIM-11, version française Peraud 2022). PCL-5 non retenu pour éviter un doublon avec l'ITQ.
**Instruments sous réserve d'autorisation** (AUDIT, DAST-10, CAST, ICJE, SCOFF, EAT-26, OCI-R, TAS-20, MSI-BPD, MDQ, DES-II, ECR-R, Rosenberg) : **non intégrés** tant qu'aucune autorisation écrite n'est obtenue ; à la place, repères courts paraphrasant la CIM-11 (CDDR), clairement marqués « repère, non validé », pour les domaines correspondants.
**Exclus** : BDI-II, BAI, PAI-BOR, Y-BOCS, PDSS, ISI, SPIN, YSQ, C-SSRS et PQ comme scores, ASRS.

**Parcours en deux temps** : une question de sécurité pour tous, puis environ 30 questions d'entrée par domaine ; approfondissement seulement si l'entrée est positive. Échelles d'origine respectées, période de référence affichée (« ces deux dernières semaines », etc.). Même style que Mon enfance (sections repliables, réponses en segmenté fin), sauvegarde automatique, reprise possible.

**Résultats** : par domaine, « au-dessus / en dessous du repère publié par <instrument> » ou « plusieurs repères présents » pour les repères non validés ; jamais de nom de trouble présenté comme conclusion.
**Croisement avec Mon enfance** : trois listes — sujets signalés à la fois par vos vécus d'enfance et par vos symptômes actuels (en tête), sujets signalés seulement par vos symptômes, sujets seulement liés à vos vécus. Table de correspondance domaine ↔ sujets de `data/cartographie.json` ; aucun nouveau lien vécu → sujet créé ; les « Pas moi » de Mon enfance sont respectés.
**Sécurité** : toute réponse positive à la question de sécurité ou à l'item 9 du PHQ-9 affiche immédiatement les ressources (3114, 15/112 ; 3919, 119, 114 selon le cas), sans score de risque ; Ambre reçoit l'alerte comme antécédent récent selon la règle existante.

**Ambre** : résumé borné (≤ 1 200 caractères) des domaines au-dessus du repère, de la date, et du croisement ; jamais les réponses item par item ; consigne : hypothèses à explorer, pas de diagnostic, inviter à en parler à un professionnel. Budget par tour ≤ +10 % (test existant).
**Prérequis d'Ambre** : non modifiés (la page est facultative).
**Stockage** : table `symptom_checks` (historique par instrument, date, réponses, score, version), consentement distinct, export, suppression avec le compte ; délai avant de refaire : 14 jours (PHQ, GAD), 30 jours (ITQ, PHQ-15, repères). Migration `0028` additive.
**Transparence** : Paramètres montre ce qu'Ambre reçoit.

## 3. Vérifications
Relecture clinique indépendante des repères non validés et des textes ; relecture de code ; tests unitaires et navigateur à 390 tactile, 1280×700, 1440 ; recette réelle avec compte synthétique supprimé.

## Hors périmètre / à décider plus tard
Lettres d'autorisation pour AUDIT, SCOFF, etc. ; avis juridique (statut de dispositif médical, analyse d'impact sur les données de santé) ; EPDS (périnatal) et PG-13-R (deuil prolongé).
