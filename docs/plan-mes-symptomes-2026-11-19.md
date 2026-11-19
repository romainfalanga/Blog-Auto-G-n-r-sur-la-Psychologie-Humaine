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


---

# Version 1.1 après relecture critique indépendante (fait foi)

Les points ci-dessous remplacent ce qui les contredit plus haut.

## Moments
- Migration unique `0027_life_event_significance.sql` ; `0028_symptom_checks.sql` (réponses, consentement, alertes de sécurité).
- `significance` = « marquant dans mon histoire » (anecdotique / marquant) ; `current_importance` (1–5, « pèse aujourd'hui ») est conservé, sans conversion automatique.
- Obligatoire à la création dans l'interface ; une modification sans le champ garde la valeur enregistrée ; les propositions d'Ambre (`life_event_create` / `update`) ne déduisent jamais la valeur : la personne la choisit en validant.
- Tri d'Ambre : moments cités explicitement, puis marquants, puis ordre actuel ; un moment avec `exposure` n'est jamais relégué parce qu'il est noté anecdotique.
- Même affichage sur la frise des proches ; le filtre « Marquants » montre à part les moments « à préciser ».

## Instruments
- PHQ-2 → PHQ-9 et GAD-2 → GAD-7 avec entrée à **≥ 2** (Levis 2020 ; Kroenke 2007), repère PHQ-9 / GAD-7 ≥ 10 ; approfondissement possible sur demande même si l'entrée est négative ; paliers affichés (5–9 compris).
- PHQ-15 : période et délai de 28 jours. ITQ : question d'événement et de date propre à l'ITQ, « typiquement » pour les items d'organisation de soi, TSPT et TSPT-C exclusifs ; jamais « diagnostic probable ».
- **PCL-5 exclu** (retiré du JSON et du stockage).
- Archives des pages de licence (phqscreeners, Trauma Measures Global, PDF Cn2r) avec empreinte ; traductions françaises officielles, aucun item retouché ; demande de confirmation au Cn2r notée (non bloquante).
- Le serveur recalcule les scores à partir des réponses et de la version de l'instrument ; générateur `scripts/build_symptomes.py --check`.

## Sûreté
- Page réservée aux adultes : la date de naissance de Mon compte (déjà obligatoire pour Ambre) doit donner ≥ 18 ans ; sinon orientation (119, Fil Santé Jeunes) et rien n'est enregistré.
- Écran de sécurité (question universelle, item 9 du PHQ-9) : ressources immédiates, boutons « Continuer », « Parler à Ambre », « Arrêter », sans délai ni score de risque.
- Sous chaque repère positif : « Un repère au-dessus du seuil signale qu'un échange avec un professionnel est utile ; beaucoup de personnes dans ce cas n'ont pas de trouble après un entretien. » Sous chaque résultat négatif : « Cela n'exclut pas une difficulté ; si quelque chose vous inquiète, parlez-en à un professionnel. »
- Temps 1 : humeur, anxiété, sécurité, puis liste de modules à cocher ; les modules sensibles (psychose, dissociation, personnalité, sexualité, substances) ne sont proposés que si la personne les choisit.
- Délais : questions « vie entière » et d'exposition posées une fois ; repères sur 12 mois ou 2 ans : 90 jours.
- Repères non validés réécrits selon la relecture (humeur_duree, humeur_perinatal, deuil_6mois en deux questions, adaptation_evenement, panique_inquietude, agoraphobie scindée avec résultat propre, questions de gêne « Non / Un peu / Nettement ») ; les 20 domaines restants sont rédigés puis relus par un relecteur clinique distinct avant intégration ; un domaine non relu n'est pas publié.

## Croisement avec Mon enfance
- Correspondances par règle explicite (instrument ou repère → sujets), pas par domaine entier ; retirés : prémenstruel, rumination, inférences négatives (PHQ-9), névrosisme (GAD-7), fibromyalgie, migraine (PHQ-15), tous les `sch_*`.
- Seuls les sujets avec lien empirique (`associated_count` > 0).
- Quatre états : signalé, non signalé, non exploré, refusé ; tri par ordre des modules puis alphabétique ; aucun nombre de vécus affiché.
- Liste 1 intitulée « Présent dans vos deux questionnaires », avec : « Beaucoup de personnes ayant ces vécus n'ont pas ces difficultés, et inversement ; ce rapprochement ne montre pas une cause. »

## Ambre
- Bloc symptômes ≤ 1 200 caractères pris sur la part de la cartographie ; test prouvant le plafond +10 % (mesures actuelles : rédaction +9,1 %, bilan +8,9 %).
- Interdits : nommer un trouble, présenter l'enfance comme cause, s'appuyer sur un résultat de plus de 90 jours sans le dater.
- Question de sécurité d'Ambre au plus une fois par 14 jours, avant toute exploration.

## Lots
- **0** (contenu, sans code) : licences archivées, 20 domaines restants rédigés, réécritures, puis relecture clinique distincte.
- **A** : migrations 0027/0028, API, moteur pur partagé, sécurité, export, suppression, tests.
- **B** : interface (page, frise, ajout depuis un proche) après gel du contrat d'API.
- **C** : Ambre et Paramètres.

## Risques notés (décision du propriétaire : non bloquants)
Statut de dispositif médical et analyse d'impact sur les données de santé : signalés, non bloquants selon la consigne du propriétaire.
