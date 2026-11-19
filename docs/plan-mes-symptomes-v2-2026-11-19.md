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

## 5. Vitesse des pages
(À compléter avec l'audit de performance.)

## 6. Lots et vérifications
- Lot S1 (contenu) : 7 rédacteurs, dédoublonnage, relecture indépendante, build `--check`.
- Lot S2 (moteur et page) : layout/JS/CSS génériques, classement par jeu de données, Edge, migration 0029, backend, retrait de la v1, migration 0030.
- Lot S3 (Ambre) : contexte, consignes, budget, Paramètres.
- Lot P (vitesse) : corrections de l'audit.
- Relecture de code indépendante par lot ; tests unitaires et navigateur (390 tactile, 1280, 1440) ; mesures avant/après ; déploiement (migrations tracées, Worker complet, fusion) ; recette réelle avec compte synthétique supprimé.
