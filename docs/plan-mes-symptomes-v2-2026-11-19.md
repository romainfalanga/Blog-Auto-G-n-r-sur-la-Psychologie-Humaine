# Plan « Mes symptômes » v2 et vitesse des pages (19 novembre 2026)

Remplace `docs/plan-mes-symptomes-2026-11-19.md` (v1/v1.1, refusée par le propriétaire). Analyse de départ : rapport d'architecture du 19/11 (Mon enfance de bout en bout) et audit de performance (section 5).

## 0. Demande du propriétaire
- Aucun bouton d'accord à l'entrée.
- Tous les symptômes de l'ensemble des troubles et sujets du site, chacun demandé : Non / Oui / Je ne sais pas.
- Exactement l'interface de Mon enfance : sous-page « Mes symptômes » et onglet « Ce qui en ressort ».
- Hermétique : Mon enfance (vécus → sujets possibles) et Mes symptômes (symptômes → sujets possibles) produisent chacun leur « Ce qui en ressort », même présentation.
- Réponses réellement stockées en base ; Ambre a le détail des deux pages et formule des hypothèses sur ce qui est probable, sans renvoyer systématiquement vers un professionnel.
- Pages plus rapides.

## 1. Contenu : banque de symptômes
- Source : `editorial/symptomes/v2/` (`themes.json`, `liens-<domaine>.json`) → `scripts/build_symptomes.py` réécrit sur le modèle de `build_cartographie.py` → `data/symptomes.json` (même schéma que `data/cartographie.json` : `groups` = thèmes, `factors` = symptômes, `links`, `sources`, `ranking`, `coverage`) et copie compacte Worker.
- Sujets : les `outcomes` de `data/cartographie.json`, référencés par identifiant (jamais copiés) ; contrôle d'existence au build. Les 4 sujets « santé physique » (diabète, maladies cardiovasculaires, migraine, surpoids) et les sujets contextuels sont hors périmètre des liens.
- Volume visé : environ 220 à 260 symptômes distincts en 10 à 14 thèmes, couvrant chaque sujet à symptômes (diagnostics, groupes de symptômes, traits, tendances, schémas, attachements). Plafonds de questions relevés pour ce jeu.
- Formulation : français simple, à la première personne implicite (« Avoir du mal à s'endormir… »), description courte et exemples comme Mon enfance ; paraphrase originale (CDDR CC BY-NC-ND : pas de traduction suivie ; DSM non réutilisé) ; sources citées.
- Poids des liens : `critere` (caractéristique essentielle CIM-11 / définition du sujet), `moderee` (caractéristique associée documentée), `limitee` (symptôme non spécifique).
- Méthode : un rédacteur par domaine (7 en parallèle), puis un agent de dédoublonnage (fusion des symptômes partagés, `overlap_group`), puis relecture indépendante (faits, poids, lisibilité, licence). Couverture restante consignée.
- Sécurité : les symptômes « pensées de mort ou de suicide » et « se faire du mal volontairement » existent ; un « Oui » affiche sous la question une ligne discrète avec le 3114 (pas d'écran bloquant, pas de bouton d'accord).

## 2. Page et interface
- `/mes-symptomes/` devient une page Hugo statique comme Mon enfance, même layout rendu générique (`data-dataset`), mêmes fichiers JS/CSS paramétrés (titres des onglets, chemin d'API, clé de brouillon, jeu de données).
- Onglets « Mes symptômes » / « Ce qui en ressort », thèmes repliables, « n / N renseignés », sauvegarde automatique, brouillon local avant connexion, enregistrement en base une fois connecté (comme Mon enfance), retours « Je me reconnais » / « Pas moi », notes.
- Classement des sujets : nombre de symptômes distincts liés répondus « Oui » (liens `critere` et `moderee`, `limitee` compté à part), départagé par la part des caractéristiques essentielles présentes ; réglage `ranking` par jeu de données, appliqué au moteur navigateur et au moteur Worker (test de parité étendu). Mon enfance garde exactement son classement.
- Fonction Edge : chargement initial pour `/mes-symptomes` comme pour Mon enfance.
- Suppression de l'ancienne page Worker (questionnaires, accord, écrans de sécurité, croisement), des routes et du proxy `/mes-symptomes*` vers le Worker.

## 3. Stockage
- Migration `0029_symptom_maps.sql` : table `symptom_maps`, copie de `cartographies` (answers, feedback, notes, ambre_notes, consent, locked_at, version, updated_at), triggers `care_revision_*` comme `cartographies`.
- Backend : `createCartographyBackend` rendu paramétrable (table, données, route) ; routes `GET|PUT|DELETE /api/symptomes` ; export et suppression de compte.
- Migration `0030_drop_symptom_v1.sql` : suppression des 4 tables de la v1 (production : 1 ligne d'accord, 0 résultat) et du code associé.

## 4. Ambre et bilans
- Ambre reçoit les deux pages en détail : toutes les réponses « Oui » et « Je ne sais pas » par thème, les « Pas moi », les notes, et les deux classements « Ce qui en ressort » (sujets avec leurs symptômes ou vécus liés).
- Consignes : croiser les deux pages et la conversation ; formuler des hypothèses sur les troubles ou schémas probables, en les nommant comme hypothèses, avec ce qui les appuie et ce qui manque ; ne pas renvoyer systématiquement vers un professionnel ; sécurité suicide inchangée.
- Budget : décision du propriétaire (« accès à l'ensemble des informations ») — le plafond des sources est relevé pour ces deux blocs ; le test de budget est mis à jour avec cette décision explicite et un nouveau plafond mesuré.
- Paramètres : la ligne « Mes symptômes » montre ce qu'Ambre reçoit.
- Prérequis d'Ambre inchangés (Mes symptômes reste facultatif).

## 5. Vitesse des pages (audit du 19/11)
Mesures : Ambre connectée LCP 3,3–4,5 s (`/api/me` puis `/api/chat` en cascade) ; pages privées 1,3–1,5 s ; Paramètres 2,8 s ; /personnages/ 2,4 s avant interactivité sur mobile (index de 1,26 Mo dans le HTML) ; Mon enfance 356 Ko de données dans le HTML et Edge qui attend ~1 s ; démarrage à froid du Worker (examples.mjs 5,6 Mo) ; plantage de l'onglet Ambre dû à `SpeechRecognition.available()` (voice-input.mjs:114).
Corrections, lot P (Worker et pages publiques hors Mon enfance) :
1. Placement intelligent du Worker près de D1 (`placement: smart` dans les métadonnées de déploiement et `journal/wrangler.jsonc`) ; lectures regroupées (`Promise.all` / `DB.batch`) dans `/api/me`, `chat.state`, `initialPageData`.
2. Ambre pré-rendue (`INITIAL_PAGES.chat`).
3. `voice-input.mjs` : `available()` seulement au premier usage du micro, avec détection de fonctionnalité.
4. Script et CSS de l'application servis en fichiers à empreinte `public, max-age=31536000, immutable` (aucune donnée personnelle) ; page privée `no-store` avec seulement l'amorçage et les données initiales.
5. /personnages/ : index de recherche en fichier à empreinte, chargé à la première frappe.
6. Paramètres : `/api/chat/insight` chargé à l'ouverture de sa section.
7. `examples.mjs` chargé paresseusement.
8. `/js/*` en cache immuable ; logo réduit (WebP ~3 Ko).
Lot S2 (Mon enfance et Mes symptômes) : données en JSON statique à empreinte, Edge qui n'attend plus `initial-state` plus de ~300 ms (ou chargement des données privées par le navigateur).
Mesures avant/après avec le même script (`scratchpad/perf/`).

## 6. Lots et vérifications
- Lot S1 (contenu) : 7 rédacteurs, dédoublonnage, relecture indépendante, build `--check`.
- Lot S2 (moteur et page) : layout/JS/CSS génériques, classement par jeu de données, Edge, migration 0029, backend, retrait de la v1, migration 0030.
- Lot S3 (Ambre) : contexte, consignes, budget, Paramètres.
- Lot P (vitesse) : corrections de l'audit.
- Relecture de code indépendante par lot ; tests unitaires et navigateur (390 tactile, 1280, 1440) ; mesures avant/après ; déploiement (migrations tracées, Worker complet, fusion) ; recette réelle avec compte synthétique supprimé.

---

# Version 2.1 après relecture critique (fait foi sur ce qui précède)

## Contenu (lots S0, S1)
- Période de référence unique : « ces 12 derniers mois », affichée en tête de la page et rappelée dans les thèmes ; une précision libre reste possible par question (notes).
- Les libellés, descriptions et URL des sujets sont matérialisés par le build depuis `data/cartographie.json` ; `--check` échoue s'ils divergent. Le build et le module Worker s'ajoutent à la commande de `netlify.toml`.
- Lot S0 avant les rédacteurs : `editorial/symptomes/v2/themes.json` (10–14 thèmes de 15–25 questions) et environ 40 symptômes transdiagnostiques canoniques (sommeil, fatigue, appétit, concentration, irritabilité, idées de mort, etc.) que les rédacteurs réutilisent par identifiant.
- Format imposé `liens-<domaine>.json` :
  `{"domain":…,"symptoms":[{"id","canonical","theme","label","description","examples":[≤2],"safety":null|"suicide"|"automutilation","overlap_group","links":[{"outcome","strength":"critere|moderee|limitee","basis","summary","sources":[…]}]}],"sources":[{"id","type","citation","url","licence"}]}` ; dédoublonnage dans `fusion.json` `{"merge":[{"keep","drop":[…],"why"}]}`, appliqué mécaniquement (union des liens, poids le plus fort, arrêt si conflit non résolu).
- Contrôles du build : identifiants uniques ; sujets existants, ni contextuels ni « santé physique » ; sources résolues ; 3 à 12 liens `critere` par sujet lié ; symptômes de sécurité présents ; aide ≤ 120 caractères, ≤ 2 exemples ; taille des fichiers produits.

## Classement (version `essential_share_desc`, propre au jeu « symptômes »)
- Pour chaque sujet : C = familles (`overlap_group`) liées en `critere` ; c = familles `critere` répondues Oui ; m = familles `moderee` répondues Oui. Score S = ⌊1000·c/(C+1)⌋ ; ordre S, puis c, puis m, puis identifiant. Affiché si c ≥ 2 (ou c ≥ 1 quand C ≤ 2). `limitee` et « Je ne sais pas » ne classent pas, affichés « à préciser » dans le détail. Explication affichée : « part des signes caractéristiques présents, prudente quand un sujet a peu de signes ».
- Appliqué dans `static/js/cartographie.js` et `computeCartography` ; test de parité ; test de non-régression de Mon enfance sur instantané (classement, sections, DOM, brouillon).

## Interface
- Objet `ui` par jeu de données : textes, identifiants DOM, API, clé de brouillon ; ressources, exclusivité et compteur ACE désactivables.
- Lignes rendues à l'ouverture d'un thème ; compteur global « n / N » ; reprise sur le premier thème incomplet.
- Données servies en JSON statique à empreinte, mis en cache, au lieu d'être intégrées au HTML `no-store` ; résumés de liens chargés à la demande ; module Worker < 500 Ko. (Même traitement à étudier pour Mon enfance dans le lot P.)
- Edge : chemin `/mes-symptomes` et `page=symptomes` ; Worker : retrait de `pageTitles['/mes-symptomes']`, de la page v1 et des redirections `/mes-symptomes*` de `netlify.toml`.

## Stockage
- `0029_symptom_maps.sql` : `symptom_maps` (answers, feedback, notes, consent, locked_at, version, updated_at, `safety_at`), sans `ambre_notes` (les notes d'Ambre restent sur Mon enfance) ; triggers `care_revision_*` avec un périmètre `symptoms` ; `CARE_SCOPES` étendu (partageable avec le psychologue comme Mon enfance, sur choix de la personne).
- Backend paramétré (`table`, `body_limit` 120 000, `note_max`) ; export et suppression de compte.
- Pas d'accord explicite : choix du propriétaire, consigné.
- Ordre : déployer le Worker sans v1 et vérifier, puis appliquer `0030_drop_symptom_v1.sql`. Retrait de `symptomsForAmbre`, `SYMPTOMS_RULES`, `symptomsCost`/`symptomsShare`, `symptoms_safety`. Le fichier 0028 reste dans l'historique.

## Ambre
- Blocs fixes hors détails : `symptom_map` ≤ 3 500 caractères (n/N, Oui et Je ne sais pas par thème, Pas moi, reconnus, 10 premiers sujets avec c/C et m) et Mon enfance ≤ 3 000 ; index par thème et par sujet classé (pas par symptôme) pour le détail ; résumé de 600 caractères pour l'analyse ; bilans : part propre de 6 000 caractères (minimum 3 000), contexte porté à 40 000.
- Réécriture conjointe : prompt de rédaction, critère de relecture n° 3, consigne Mon enfance. Formule : « hypothèse : X, appuyée par A et B ; il manque C » ; interdits : « vous avez », scores, verdict ; orientation vers un professionnel seulement en cas de danger, de forte gêne au quotidien ou sur demande.
- Sécurité : `safety_at` (premier Oui à un symptôme de sécurité, fixé par le serveur) remplace les alertes v1 : question directe d'Ambre au plus une fois par 14 jours, drapeau des bilans ; sous la question cochée, une ligne discrète avec le 3114.
- Budget : nouvelles références mesurées dans `v5-A5-budget` (rédaction ≤ +40 %, bilan ≤ +20 %), liées à la décision du propriétaire.

## Lots et agents
S0 (thèmes et canoniques) → S1 (7 rédacteurs en parallèle) → dédoublonnage → relecture clinique indépendante. En parallèle, S2 sur un petit jeu de test (20 symptômes, 6 sujets) : front générique et parité, backend 0029 et Edge, retrait v1. Puis S3 (Ambre). Lot P (vitesse) selon l'audit. 0030 en dernier. Relecture de code indépendante par lot.
