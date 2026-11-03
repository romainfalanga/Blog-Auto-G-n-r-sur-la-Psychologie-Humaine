# Décode ton esprit — plan de reprise, portrait enrichi et trois personnages

Statut : **plan prêt à implémenter après le feu vert de Romain ; aucune fonctionnalité de ce plan n'est encore implémentée.**

Audit du 3 novembre 2026, base de code `fe97b244186e81e8df543f23e66714a5fe1fcc10`. Dépôt : [romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine](https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine). Site : [decodetonesprit.fr](https://decodetonesprit.fr/).

Ce document est la consigne de reprise pour le modèle chargé de l'implémentation. Il remplace les descriptions antérieures de la génération quotidienne uniquement lorsque les lots correspondants auront été livrés. Les choix de produit ci-dessous sont des propositions explicites pour rendre le chantier exécutable ; ils ne sont pas présentés comme des décisions historiques de Romain.

## 1. Résultat attendu et critères d'acceptation

| ID | Demande | Résultat observable | Lot responsable |
|---|---|---|---|
| R1 | Réparer « Découvrir ses histoires » | La première activation ouvre le bon personnage, avant et après recherche/suggestion, à la souris, au clavier et au toucher. Aucun déplacement de la cible lors du clic. | A |
| R2 | Une recherche directe et épurée | Plus de titre « Les personnages », de phrase d'introduction, de label visible ni d'aide verbeuse dans ce bloc. Une barre avec « Peur de décevoir, pensées en boucle… », les fonctions actuelles et une accessibilité conservée. | A |
| R3 | Ajouter l'entourage dans Mon portrait | Ajouter, modifier et supprimer des proches, leur rôle et la relation décrite par l'utilisateur ; sauvegarde privée et prise en compte vérifiable. | B, C |
| R4 | Ajouter une ligne de vie | Événements positifs, difficiles, déclics ou ambivalents, dates souples, liens avec les proches, modification et suppression. | B, C |
| R5 | Utiliser ces informations dans tout le système | Proches, événements, lectures et corrections alimentent le portrait, le classement public et le brief narratif avec des sources traçables. Fonctionne aussi avec du vécu renseigné et zéro like. | C, D, F |
| R6 | Mes histoires devient Mes personnages | Nouveau libellé partout et route `/mes-personnages/` ; anciens liens et épisodes conservés. | E |
| R7 | Trois sous-versions fictives | Trois identités stables, facettes nuancées du même utilisateur, continuités indépendantes et choix du personnage à poursuivre. | E, F |
| R8 | Sept histoires hebdomadaires partagées | Génération à la demande ; compteur commun ; aucune huitième consommation dans la même semaine, même avec concurrence, reprises ou changement de personnage. | B, E |
| R9 | Apprendre des retours positifs et négatifs | Ce qui ressemble, ce qui diffère, une correction autobiographique et un souhait de lecture ont des effets distincts. Les récits ne deviennent pas des faits sur l'utilisateur. | C, D, F |
| R10 | Plan optimisé et transmissible | Architecture, contrats, migration, coûts, tests et ordre de livraison définis ; revue contradictoire ; aucune mise en œuvre avant la demande suivante. | G, présent document |

## 2. État réellement audité

L'audit couvre les composants exécutables du dépôt : les 19 templates Hugo, les scripts et styles de l'interface publique, les quatre modules Worker, les quatre migrations D1, les tests, le runner, l'adaptateur de génération privé, les 2 807 lignes du générateur historique, les constructeurs de catalogues, les scripts de migration historiques, la configuration Netlify/Hugo et le workflow GitHub. Les 231 articles publics et les 20 personnages ont été examinés comme corpus/catalogue et données de continuité ; il ne s'agit pas d'une nouvelle validation scientifique phrase par phrase de ces articles.

| Composant | Fonction actuelle | Conséquence pour le chantier |
|---|---|---|
| Hugo + Netlify | Accueil, recherche locale, personnages et articles publics | Garder le site et les adresses existants. Correctif accueil limité. |
| `journal/src/ui.mjs` | Application privée, HTML/CSS/JS dans un module | Ajouter les écrans progressivement ; conserver les protections de brouillons et de rendu asynchrone. |
| `journal/src/worker.mjs` + D1 | Authentification, profil, likes, portrait, abonnements, jobs, IA, export et suppression | Faire évoluer les contrats et invariants avant l'interface des générations. |
| `recommendations.mjs` | Classement lexical TF-IDF, commentaires, négations et diversité, sans appel IA par consultation | Bonne baseline économique ; plusieurs erreurs sémantiques à corriger avant d'ajouter du vécu. |
| `generate_personal_articles.py` | Portrait, puis un personnage et un récit personnel | Séparer le portrait réel, le registre de personnages et la mémoire de chaque saga. |
| `generate_articles.py` | Générateur public historique ; plusieurs helpers réutilisés par le privé | Le privé réutilise les prompts, parseurs et relectures, mais pas le moteur public complet de continuité. |
| `run_journals.py` + Actions | Traitement séquentiel toutes les 15 minutes, autorisation OIDC par le Worker | Réutiliser l'exécution existante en V1, remplacer le déclenchement quotidien par des demandes explicites. |
| OpenRouter via Worker | Clé dans Cloudflare, modèles autorisés côté serveur | Conserver le fournisseur et le modèle configurés ; optimiser contexte et étapes avant d'en changer. |

Contrôles de baseline : **11 tests de recherche, 31 tests backend, 9 tests de recommandations et 9 tests Python passent**, soit 60 tests. Compilation Hugo 0.140.0 et contrôle des introductions extractives réussis. Ce sont des contrôles de l'existant, pas la validation des nouvelles fonctionnalités.

### Défauts et limites déterminants

1. **Navigation accueil : hypothèse forte, pas encore preuve de reproduction.** Les 20 liens pointent vers de vraies pages Hugo. Aucun JS de recherche ne les détourne. En revanche `character-search.js` masque les suggestions sur `focusout` et le CSS les place dans le flux : leur disparition peut déplacer une carte entre l'appui et le relâchement. Cela correspond au symptôme signalé. Deux contrôles en navigateur distant ont néanmoins réussi : champ vide focalisé vers Camille, puis recherche « peur de décevoir » vers Thomas. Ne pas prétendre que tous les cas sont reproduits ou que la cause est définitivement démontrée.
2. **Différences mal interprétées.** Un texte sans négation dans « Ce qui est différent pour moi » alimente aujourd'hui le vecteur positif. Une fixture « Le deuil familial » fait remonter le deuil au premier rang au lieu de nuancer l'affinité. La négation double « Je ne veux pas éviter la colère » est également mal interprétée.
3. **Provenance partielle.** Les validateurs contrôlent l'existence des IDs, pas le soutien réel d'une affirmation. Le plafond de confiance d'un like seul n'est imposé que par le prompt. Synthèse, forces et certaines descriptions peuvent survivre au retrait de leurs sources.
4. **Contexte tronqué.** Les retours transmis à l'histoire sont coupés à 120 caractères, l'historique aux 20 derniers résumés, les feedbacks aux 30 derniers. Le résumé historique lit seulement les 5 000 premiers caractères : une résolution finale peut disparaître.
5. **Catalogue incomplet pour le raisonnement.** Les 231 métadonnées exportées n'ont pas de champ de résumé narratif : le portrait reçoit souvent une description SEO à la place. Les trois inspirations sont choisies surtout par commentaire/récence.
6. **Quota encodé dans le schéma.** `stories` a `UNIQUE(user_id,day)` et les jobs sont uniques par utilisateur/type/version/jour. Les historiques sont filtrés par utilisateur seulement. Enlever une condition dans le Worker ne suffit pas.
7. **Coûts et équité.** Le budget de 20 appels est par job, sans budget cumulatif de tokens. Une nouvelle version peut recréer un budget. Les comptes anciens peuvent être favorisés. Un catalogue manquant pour un seul compte coupe une réservation de travail entière.
8. **Contrats narratifs divergents.** Certains prompts hérités imposent genre, France ou progression positive ; la validation finale et la longueur attendue ne coïncident pas toujours. `validate_coherence` réécrit davantage qu'il ne constate des violations.
9. **Export et liens.** L'export omet l'ID des histoires alors que les retours le référencent. Le raccourci `/mon-portrait/#suggestions` vise une ancre absente. À corriger dans les lots concernés.

## 3. Choix de produit retenus pour l'implémentation

| Sujet | Choix par défaut |
|---|---|
| Période du quota | Semaine civile du lundi 00:00 au lundi suivant 00:00, **Europe/Paris** ; frontière calculée avec le fuseau, pas avec une durée fixe de 168 heures. |
| Enveloppe | 7 histoires communes aux 3 personnages, non reportables ; aucune limite quotidienne supplémentaire. Le compteur affiche aussi la date de renouvellement. |
| Déclenchement | L'utilisateur choisit « Poursuivre ce personnage ». Aucun récit automatique sans demande, aucune création automatique des sept récits. |
| Continuité | Trois fils linéaires ; poursuivre le dernier épisode du personnage choisi. Lire un ancien épisode ne crée pas un embranchement. |
| Demandes simultanées | Une seule demande d'histoire active par compte. Les sept histoires peuvent être demandées successivement le même jour. |
| Décompte | Réservation lors de l'acceptation de la demande, consommation lors de la publication réussie. Échec terminal/annulation avant publication : libération. |
| Personnages | Initialisation des trois fiches incluse, sans consommer une histoire. Identités conservées aux réanalyses. Préparation pour un compte abonné, une seule fois pour les slots manquants. |
| Entourage et événements | Facultatifs et disponibles dans Mon portrait sans obligation de raconter un traumatisme ni de donner de vrais noms. |
| Accès au portrait | Une source personnelle explicite suffit pour commencer, même sans like ; un rôle familial seul n'est pas une source suffisante pour inférer un fonctionnement. |
| Conditions commerciales | Conserver le tarif et le seuil actuel de 10 likes pour une **nouvelle souscription**, hors demande de modification de ces règles. Un abonné peut ensuite continuer avec des sources personnelles sans like. Afficher clairement cette distinction. |
| Attente | V1 conserve la file Actions. Afficher « demande enregistrée / en préparation / prête », sans promettre un résultat immédiat ou garanti en 15 minutes. |
| Historique | Le personnage actuel et tous ses épisodes deviennent le slot 1 ; les slots 2 et 3 sont ajoutés. Les anciens IDs et liens restent valables. |

Ces paramètres sont centralisés et testés. Si Romain demande ensuite une semaine glissante ou une autre règle commerciale, la modifier explicitement, sans mélanger les deux définitions.

## 4. Lot A — accueil et navigation

### Diagnostic et correctif

Avant correction, mesurer en navigateur le scénario du signalement : focus vide, saisie, sélection d'une suggestion, clic direct carte, fermeture préalable avec Échap/Entrée, souris/tactile/clavier, retour navigateur. Relever URL, position de la carte et événements pointerdown/focusout/pointerup/click dans l'environnement de développement. Essayer un écran mobile avec clavier ouvert. Un test qui utilise seulement le moteur de recherche ou une activation DOM ne couvre pas ce défaut.

Implémentation prévue :

- `layouts/partials/character-directory.html` : retirer l'introduction du bloc et les textes visibles superflus ; garder `id="personnages"`, un nom accessible au champ, le compteur discret et l'état sans résultat. Placeholder exact : **Peur de décevoir, pensées en boucle…**. Corriger toutes les références ARIA aux éléments supprimés.
- `static/css/style.css` : une barre sobre cohérente avec le site, icône, effacement ; suggestions en panneau superposé ancré à la barre, sans influence sur la hauteur de la grille, avec défilement et limites de taille adaptés au mobile.
- `static/js/character-search.js` : conserver index, synonymes, fautes légères, combinaison de mots, classement et effacement. Adapter seulement le cycle de focus/fermeture nécessaire. Préserver clic secondaire, nouvel onglet, sélection de texte et clavier ; ne pas forcer la navigation sur `mousedown`.
- Garder la promesse de la recherche actuelle : prénoms, situations, émotions, notions et contenu des histoires. Ne pas réintroduire un grand bloc d'explications pour remplacer celui supprimé.
- Ajouter/réparer l'ancre des suggestions privées et le défilement après rendu lorsqu'un lien la demande ; conserver la navigation privée sans perte des paramètres.

Si l'overlay supprime le déplacement mais qu'un scénario reste défaillant, poursuivre le diagnostic sur l'événement réel et le routage de cette URL avant de déclarer R1 terminé.

## 5. Lot B — données privées et contrats partagés

### Entourage et ligne de vie

`personal_people` : `id`, `user_id`, `display_name` (prénom ou pseudonyme), `role`, `custom_role`, `relationship_text`, période facultative, `revision`, `created_at`, `updated_at`. Rôles proposés : père, mère, parent, partenaire, enfant, frère/sœur, ami, meilleur ami, autre. Le texte exprime la perception de l'utilisateur ; il ne prouve pas les intentions du proche.

`life_events` : `id`, `user_id`, `title`, `kind` (difficile/positif/déclic/ambivalent/autre), `description`, `meaning_now` facultatif, `date_precision`, date/période ou âge facultatif, `current_importance` facultatif, `fiction_use`, `revision`, dates techniques. `life_event_people` porte les associations multiples, avec vérification du même propriétaire.

- Accepter année seule, mois, intervalle, âge approximatif ou absence de date. Ne jamais inventer un jour ou exiger la date de naissance. Les événements non datés sont présentés à part ; conserver la distinction entre date du vécu et date de saisie.
- Limites V1 : nom 80 caractères, rôle libre 80, titre 160, relation/description 6 000, signification actuelle 2 000 ; pagination 50 éléments, 50 proches et 500 événements par compte, dix associations par événement pour borner abus et traitement. Valider aussi la taille totale du JSON. Erreur explicite, jamais troncature silencieuse. Constantes ajustables après mesure.
- Réutilisation fictive : par défaut transposition abstraite des thèmes, jamais copie des personnes réelles ; option simple « Ne pas mettre cet événement en scène » (`understanding_only`). Dans ce cas, le brief d'écriture ne reçoit pas le détail de la scène, seulement les besoins/limites abstraits admissibles.
- Supprimer un proche enlève les associations, pas les événements liés. Les textes déjà écrits par l'utilisateur ne sont pas réécrits silencieusement. L'interface explique ce résultat avant suppression.
- Toute écriture effective incrémente `profiles.version` une fois dans la même transaction ; écriture identique sans changement = aucune incrémentation. Contrôle optimiste `revision` pour deux onglets ; conflit 409 plutôt qu'écrasement.

### Schéma additionnel minimal

| Table/objet | Rôle et invariants |
|---|---|
| `personal_characters` | `id,user_id,slot` (1–3, unique par compte), identité fictive JSON stable, facette sourcée révisable, état narratif JSON, `canon_version`, `last_story_id`, dates. Un même prénom ne constitue pas l'identité. |
| `story_requests` | Registre des demandes ET des crédits : ID, compte, personnage, clé d'idempotence et empreinte des paramètres, semaine réservée, parent attendu, `status`, `credit_status` (reserved/consumed/released), source/canon versions, tentatives, appels/tokens cumulés, échéance, résultat/erreur. |
| `stories` évoluée | Garder IDs, dates et corps existants ; ajouter personnage, demande unique, numéro d'épisode, parent, versions utilisées, résumé complet et delta narratif. Remplacer l'unicité quotidienne par `UNIQUE(character_id,episode_number)` et `UNIQUE(request_id)`. |
| `journal_jobs` évoluée | Jobs `portrait`, `characters`, `story` ; références de demande/personnage, version, bail, compteur ; dédupliquer le portrait par version, l'initialisation par registre, l'histoire par demande. Retirer l'unicité fondée sur le jour. |
| `context_source_derivatives` | Extraction/synthèse par source et révision, version d'analyse, empreinte et manifeste. Reconstruction possible depuis les sources originales. Pas de seconde biographie indépendante. |
| `user_portraits` | Portrait structuré avec preuves, contradictions, version de sources et version du schéma. Plus de personnage fictif comme partie du portrait réel. |
| `feedback` évolué | Conserver les données anciennes, ajouter portée et éléments ciblés ; distinguer ressemblance, correction, préférence et souhait d'évitement. |

Toutes les lignes privées sont liées au propriétaire, supprimées avec le compte et filtrées par `user_id` côté serveur. Ajouter des contraintes composites/contrôles atomiques pour empêcher qu'un événement, un parent d'épisode ou un proche d'un autre compte soit référencé. Indexer compte/date, personnage/épisode, état/date de demande, source/révision. Ne pas stocker les prompts complets dans les logs ni dans Git.

### Contrats HTTP à définir avant les écrans

| Route | Contrat |
|---|---|
| `GET/POST /api/people`, `PATCH/DELETE /api/people/:id` | CRUD, pagination, révision, validation serveur, `source_version` retournée. |
| `GET/POST /api/life-events`, `PATCH/DELETE /api/life-events/:id` | CRUD, dates souples, associations du même compte, révision, `source_version`. |
| `GET /api/characters` | Trois slots, état de préparation, facettes, dernier épisode et quota commun. Un GET ne génère aucun récit. |
| `POST /api/characters/prepare` | Demande idempotente de préparation des slots manquants, compte éligible, sans crédit récit. |
| `POST /api/characters/:id/continue` | Clé d'idempotence obligatoire, parent attendu ; réservation atomique et réponse 202 avec demande/quota. Aucun appel IA dans cette requête web. |
| `GET /api/story-requests/:id` | État, erreur exploitable, résultat éventuel, attente ; jamais prompts, secrets ou données d'un autre compte. |
| `POST /api/story-requests/:id/cancel` | Annule une demande non publiée et invalide son bail ; si la publication a déjà gagné la course, retourner l'histoire existante. |
| `GET /api/stories?character_id=…&cursor=…` | Pagination stable par épisode puis ID ; lecture d'un ancien `story_id` retrouve le personnage associé. |
| Routes existantes portrait/recommandations/retours/export | Schéma versionné enrichi ; anciens clients compatibles pendant la transition. Export avec IDs, proches, événements, personnages, histoires et liens de feedback. |

Conserver authentification, contrôles Origin/CSRF applicables, limites de requêtes, requêtes préparées, échappement des textes et `private, no-store`. Le vieux champ `consent_ai` n'est pas une barrière effective actuelle : ne pas lui attribuer ce rôle. Expliquer au point de saisie que les informations enregistrées servent au portrait et aux histoires, et utiliser le même circuit privé déjà prévu, sans nouvelle étape d'accord inventée.

Les créations de proches/événements doivent également résister à un double envoi, via un identifiant de création idempotent. Les textes saisis et les extraits d'articles sont des données non fiables pour les consignes de l'agent : les délimiter comme sources, ignorer les instructions qu'ils pourraient contenir et ne jamais leur permettre de choisir un modèle, d'augmenter un budget ou d'accéder aux données d'un autre compte.

## 6. Lot C — comprendre les sources sans confondre l'utilisateur et la fiction

### Quatre catégories strictement séparées

1. **Déclarations personnelles** : ce que l'utilisateur écrit sur son vécu, ses relations et son ressenti. Toujours attribué à sa perspective.
2. **Affinités et souhaits de lecture** : likes, raisons, sujets souhaités ou évités. Aimer un récit de deuil ne signifie pas avoir perdu un parent.
3. **Hypothèses du portrait** : interprétations prudentes, liées à leurs preuves et contre-exemples, confirmables/nuançables/retirables.
4. **Fiction** : identités, scènes, relations inventées et événements des personnages. Jamais preuve autobiographique. Seule une déclaration explicite de l'utilisateur à leur sujet peut devenir une source personnelle.

Chaque source possède un ID typé (`like:…`, `feedback:…`, `memory:…`, `person:…`, `event:…`), une révision, un hash, sa date de saisie, sa temporalité éventuelle et son texte original. Le manifeste autorisé est construit côté serveur. Le rôle « père » ou « mère » n'autorise aucun diagnostic, aucune causalité ni attribution d'intention.

### Mise à jour et hiérarchie

- La correction explicite prévaut sur une interprétation portant sur **la même affirmation et la même période**. Une relation aujourd'hui apaisée n'efface pas automatiquement une période difficile passée.
- Les descriptions récentes renseignent l'état actuel ; les événements anciens restent disponibles sans être arbitrairement considérés comme actuels. Les forces, moments positifs et ressources comptent autant que les difficultés.
- Un like seul reste un signal faible d'intérêt. Une source répétée dans une synthèse, puis dans le portrait, ne devient pas trois preuves indépendantes.
- Ne pas convertir « ce n'est pas moi » en « ne me propose plus ce thème ». Un souhait d'éviter un sujet doit être explicite et peut être retiré.
- Conserver le texte intégral ; retirer les coupes aveugles à 120 caractères. Un texte trop long est refusé clairement à la saisie, puis résumé/récupéré avec références, pas tronqué à l'endroit où peut se trouver la correction.

### Pipeline d'analyse

1. Enregistrement transactionnel puis version globale incrémentée. Regrouper les modifications rapprochées ; aucune analyse à chaque frappe.
2. Projection canonique des sources autorisées, incluant mémoire, proches, événements et retours anciens pertinents. Supprimer la dépendance technique à un premier like dans Worker, recommandations, statuts, validateurs et UI. Centraliser `hasPersonalizationSources` : like valide, mémoire explicite, texte de relation, événement décrit ou retour personnel ; un alias/rôle seul ne suffit pas. Ne pas réintroduire automatiquement l'ancien questionnaire `profiles.data`. Charger un snapshot cohérent depuis D1, ou contrôler sa version avant/après lecture et recommencer sans appel IA s'il a changé.
3. Extraction incrémentale uniquement des sources nouvelles/modifiées : déclarations, intérêt, non-ressemblance, correction, évitement, ressources, temporalité. Conserver une citation courte ou des offsets vérifiables dans la source, plus les éventuelles ambiguïtés. Une extraction ambiguë ne devient pas un refus ferme.
4. Synthèse versionnée sur les dérivés à jour, avec récupération des originaux contradictoires et importants. Stocker couverture, sources non encore traitées et contraintes prioritaires. Ne pas prétendre avoir analysé toute une ligne de vie si un lot est encore en attente.
5. Validation de schéma et provenance en Python **et** Worker avec les mêmes fixtures : références existantes/révision actuelle, citation réelle, type de preuve permis, aucune biographie tirée d'un like seul, confiance d'un like seul plafonnée à 0,25. Ce chiffre est un garde de prudence interne, pas une probabilité clinique.
6. Vérification sémantique ciblée des nouvelles hypothèses : la citation soutient-elle réellement l'énoncé ? Cette vérification améliore la fidélité sans garantir à elle seule la vérité psychologique. Toute phrase de synthèse/force/état doit découler d'éléments sourcés validés.
7. Publication conditionnelle seulement si versions et bail sont toujours valables. Les anciennes interprétations ne sont jamais utilisées comme preuves autonomes.

Les calculs locaux continuent immédiatement à partir des signaux explicites déjà sûrs lorsque la synthèse IA attend. Une extraction sémantique pas encore actualisée ne doit pas être présentée comme à jour. Garder la limite gratuite de trois démarrages par 24 h et dix minutes entre démarrages ; les nouveaux formulaires ne permettent pas de la contourner. Les abonnés ont aussi un budget opérationnel, sans déclencher une génération à chaque modification.

### Retrait et corrections

Modifier/supprimer une source invalide ses dérivés et les portraits qui la citent, ainsi que les checkpoints de génération dépendants. Purger les extraits supprimés des caches actifs ; ne pas réintroduire la donnée par le résumé précédent. Les histoires déjà publiées restent des archives, sans être réinterprétées comme biographie. Leur canon fictif sert à la continuité de la fiction uniquement. La suppression d'un compte supprime aussi tous ces nouveaux ensembles ; export et suppression font partie du lot, pas d'un chantier ultérieur.

## 7. Lot D — recommandations publiques pertinentes et mesurables

### Catalogue

Conserver TF-IDF et la diversification comme baseline. Pour 231 articles, pas de base vectorielle/service supplémentaire nécessaire d'emblée. Enrichir une fois le catalogue avec résumé fidèle, concept canonique, contexte, tonalité, ressources/exercices, version et hash du contenu. Réutiliser les introductions extractives comme contexte pédagogique, pas comme preuve scientifique. Synchroniser `examples.mjs`, les données Hugo et `public_examples` D1 depuis le même manifeste.

Les métadonnées de contenu sont produites hors des consultations personnelles. Vérifier sur le texte les résumés de matrice avant de les reprendre. Garder distincts les thèmes d'un article et les traits généraux du personnage ; ne pas attribuer tous les traits de Camille à chaque lecteur d'un article de Camille.

### Classement V2

- Séparer des canaux : demande actuelle explicite, contexte personnel déclaré, résonance expliquée, affinité de lecture, hypothèses prudentes, non-ressemblance et évitement volontaire.
- Pondérations initiales relatives pour la V1 : demande explicite 4, déclaration contextualisée pertinente 3, résonance expliquée 2, like seul 1, hypothèse non confirmée au plus 0,5. Ce sont des paramètres de départ à évaluer, pas des coefficients scientifiquement optimaux.
- Calculer les contributions avec leur fiabilité, leur portée et leur temporalité **avant** normalisation finale ; conserver l'intensité absolue. Plafonner chaque source et dédupliquer la même preuve entre canaux.
- La non-ressemblance baisse la raison « cela vous ressemble » sur l'élément ciblé ; elle ne renforce jamais cet élément. Les éventuelles nouvelles déclarations positives dans ce texte sont extraites séparément, avec preuve. Un simple mot dans `differences` est négatif pour la ressemblance par défaut.
- Un évitement explicitement confirmé filtre le sujet ciblé. Un texte libre ambigu ou une double négation ne crée pas de filtre ferme. Le score de ressemblance d'une histoire ne doit pas être appliqué indistinctement à tous ses thèmes ou à son style.
- Exclure les articles déjà aimés comme aujourd'hui, diversifier uniquement parmi les candidats pertinents, limiter la répétition d'un personnage/thème. Garder au plus deux articles du même personnage dans les six premiers lorsqu'il existe assez d'autres résultats pertinents.
- Appliquer un seuil de pertinence calibré sur les fixtures du corpus. Retourner moins de propositions si nécessaire ; séparer clairement les pistes d'exploration. Ne pas remplir artificiellement douze places de score nul.
- Raisons affichées fondées sur la source réelle : intérêt de lecture, souhait formulé, contexte déclaré ou exploration. Ne jamais transformer un score en diagnostic ou en « pourcentage de ressemblance ».

Ajouter un historique d'exposition **minimal**, daté et borné, pour éviter les recommandations identiques ; sa réception ne constitue ni un like ni une preuve personnelle. Ne pas utiliser une absence de clic comme désaccord psychologique. Les embeddings ne seront ajoutés que si une comparaison sur les mêmes scénarios montre un gain suffisant pour leur coût et leur complexité.

## 8. Lots E et F — trois personnages, quota et écriture

### Trois facettes et trois continuités

Préparer ensemble les trois fiches manquantes pour assurer leur complémentarité. Chaque fiche contient identité fictive, milieu, relations inventées, motivations, ressources, tension explorée et liens prudents avec les sources personnelles. Il ne s'agit pas d'imposer trois types psychologiques prédéfinis ni de recopier les proches réels. Avec peu de données, présenter des propositions exploratoires plutôt que fabriquer trois diagnostics.

L'identité reste stable. Le portrait peut modifier les thèmes à explorer, les limites et le lien avec une facette ; il ne renomme pas le personnage, ne change pas arbitrairement sa famille et n'efface pas ses épisodes. Une même expérience peut éclairer plusieurs facettes, mais les personnages ont des contextes et ressources suffisamment distincts. Évolution possible, retours en arrière plausibles, moments ordinaires et positifs ; pas de progression thérapeutique forcée à chaque chapitre.

La mémoire par personnage comporte un canon stable, les événements importants, acquis, relations fictives et questions ouvertes, le dernier épisode et une version. Chaque épisode conserve un delta narratif avec provenance vers le passage du récit final. Les faits anciens nécessaires restent récupérables au-delà de vingt épisodes ; aucune concaténation de tous les romans dans chaque prompt.

### Réservation et publication atomiques

Le registre `story_requests` remplace un compteur mutable isolé. Invariant : pour un compte et une semaine, `réservées + consommées <= 7`.

Machine d'états : `waiting_portrait → queued → running → completed`, avec retour à `waiting_portrait` si les sources changent, et sorties terminales `failed`, `cancelled`, `expired`. Les trois premiers états réservent un crédit et sont « actifs » pour l'index unique ; `completed` le consomme ; les autres le libèrent. Une attente de préparation de personnages se traite avant l'ouverture du bouton de continuation. Les transitions et les réponses HTTP portent des codes stables, dont `quota_exhausted`, `request_in_progress`, `parent_changed` et `context_changed`.

1. Une clé d'idempotence désigne une demande unique par utilisateur. Le même appel rejoué retrouve la même demande ; même clé avec autre personnage/parent = conflit.
2. Dans une écriture conditionnelle atomique D1 : contrôler abonnement actif, propriété du personnage, dernier parent attendu, absence de demande active et nombre de crédits ; insérer la réservation. Index unique partiel sur les états actifs par compte. Ne pas faire un `SELECT count` puis un `INSERT` indépendant.
3. Lier le job à cette demande dans la même transaction, uniquement si la réservation existe. D1 `batch()` permet l'atomicité du groupe ; des clauses conditionnelles cohérentes restent nécessaires, car une écriture affectant zéro ligne n'est pas une erreur SQL.
4. Avant chaque appel IA et à la publication, vérifier bail, abonnement, source version, canon version, personnage et parent attendu. Une lecture préalable seule ne protège pas la publication d'une course.
5. Publication transactionnelle : insérer l'histoire unique pour la demande, avancer le dernier épisode/canon du personnage, consommer la réservation et terminer le job. Toutes les mises à jour dépendent de l'insertion/du résultat de cette même demande. Un `/complete` rejoué retourne le même `story_id` avec succès, sans nouvel épisode ni débit.
6. En cas d'erreur temporaire : même demande, semaine et budget cumulatif, nouveau bail ; trois tentatives maximum. En cas d'échec terminal, libérer le crédit. Un job expiré ne peut plus publier.

Cas limites fixés :

- Une demande le dimanche achevée lundi débite sa semaine de réservation ; elle ne consomme pas deux semaines. Aucune migration silencieuse vers la nouvelle semaine.
- Une modification du portrait suspend le vieux traitement ; rebase après la synthèse actuelle en conservant demande, crédit, semaine, tentatives et coût déjà dépensé. Jamais vingt nouveaux appels gratuits parce qu'une version a changé.
- Demande active limitée à 24 h : après délai total, erreur terminale et crédit libéré dans sa semaine d'origine. La nouvelle semaine n'est pas augmentée par cette libération. L'état informe l'utilisateur, qui peut demander de nouveau selon son quota actuel.
- Annulation/expiration effective de l'abonnement avant publication : annuler la demande et libérer ; arrêt du renouvellement avec droits encore payés : poursuivre jusqu'à `paid_until`. Aucun achat Stripe réel en recette.
- Ne pas bloquer la file entière pour un catalogue manquant, un profil en erreur ou un compte supprimé. Isoler l'erreur et traiter les suivants.
- Ordonnancement FIFO des demandes payantes, équité entre comptes, alternance bornée avec les portraits gratuits. La préparation du portrait/personnage nécessaire à une demande payante hérite de sa priorité sans devenir un nouveau crédit.

### Écriture et qualité

Construire un brief immuable pour une tentative : version des sources, contraintes et corrections actives, synthèse sourcée, originaux pertinents, personnage choisi, canon, parent, quelques passages nécessaires et inspirations choisies par pertinence/diversité. Aucune donnée personnelle sensible ne devient un modèle de texte public.

Pipeline cible : cadrage structuré → rédaction → contrôle structuré → correction ciblée si nécessaire → validation finale → résumé/delta du récit **final complet** → publication atomique. Conserver les helpers historiques utiles, mais retirer les consignes contradictoires et les réécritures complètes sans justification.

Le contrôle vérifie identité, relations, chronologie, acquis, attribution des faits, contraintes utilisateur, absence de copie littérale d'événements exclus et continuité du seul personnage sélectionné. Harmoniser parseur, longueur/structure éditoriales et validation finale ; vérifier explicitement la conclusion au lieu de résumer uniquement le début. Aucun personnage n'apprend les événements d'un autre par mélange de requêtes SQL.

Pour la partie pédagogique, introduire un référentiel limité de concepts avec sources éditoriales vérifiées et formulations permises. Ne pas demander de nouvelles études inventées pour satisfaire un ancien prompt SEO. Le site accompagne la réflexion, sans poser de diagnostic à l'utilisateur ni à ses proches.

### Budgets et exploitation

- Conserver provisoirement le plafond dur actuel de 20 appels, mais **par demande cumulée**, et un maximum de trois tentatives ; viser une chaîne nominale de cinq à six appels, à mesurer sur la qualité obtenue.
- Ajouter les tokens entrants/sortants, coût observé lorsqu'il est fourni, durée, tentatives et cause d'abandon ; métriques sans textes privés. Plafonds configurables par épisode et par compte/semaine incluant les échecs et reprises. Ne pas annoncer une économie chiffrée sans mesure.
- Réserver le budget fournisseur avant l'appel. Une réponse réseau perdue ou une annulation après envoi peut néanmoins être facturée ; ne pas rembourser ce budget technique sur une réponse incertaine ni promettre un appel fournisseur exactement une fois. Limiter également la création répétée de nouvelles demandes échouées. Dimensionner les plafonds tokens/coût sur la recette pour permettre sept récits normaux et les reprises prévues ; ne pas activer commercialement un budget qui rendrait les sept récits promis inaccessibles dans le cas nominal.
- Valider `max_tokens`, taille d'entrée et modèle côté serveur ; budgets par étape, timeout inférieur au temps restant du bail. Respecter une température basse pour extraction/validation et celle d'écriture pour rédaction au lieu d'imposer 0,75 partout.
- Checkpoints utilisables seulement si sources, prompts, modèle et canon compatibles ; purger/invalider ceux qui dépendent d'une source retirée. Caches privés toujours indexés par utilisateur ; seul le catalogue public est partagé.
- Conserver Actions toutes les 15 minutes en V1, afficher le statut et mesurer l'attente. Les retards ou désactivations de planification GitHub nécessitent une alerte d'exploitation ; ne pas promettre une livraison instantanée. Un autre exécuteur ne sera étudié qu'après mesure de la capacité réelle.

## 9. Interface privée et compatibilité

**Mon portrait** conserve synthèse, correction libre, articles aimés et suggestions. Ajouter « Mes proches » et « Ma ligne de vie » sous forme de sections claires : aperçu, ajout court, modification/suppression, états vides utiles, dates souples et association facultative de proches. Enregistrer explicitement, garder les brouillons après erreur et annoncer l'actualisation du portrait. Ne pas transformer l'écran en questionnaire imposé. Respecter navigation clavier, labels, focus après fermeture d'un formulaire et petits écrans.

**Mes personnages** présente trois cartes reconnaissables et un compteur commun. Chaque carte expose nom, perspective explorée, dernier épisode et bouton de continuation. L'histoire ouverte conserve l'identité du personnage visible et son historique propre. États distincts : fiches en préparation, attente de portrait, demande enregistrée, génération, prête, erreur avec action adaptée. Le coût en crédit n'est pas caché.

Formulaire de retour : conserver résonance et texte libre, distinguer « Ce qui me ressemble », « Ce qui est différent pour moi » et les corrections. Une option légère précise si le retour concerne situation, réaction, personnage ou style ; un contrôle séparé permet de demander moins d'un sujet. Ne pas exiger ces détails pour aimer simplement une histoire.

Route canonique `/mes-personnages/?personnage=<id>&story=<id>`. Ne pas utiliser le paramètre public existant `character` pour ce nouvel ID. Conserver les query strings de `/mes-histoires/`, `/journal/` et les anciens liens d'épisode lors des redirections. Le serveur peut retrouver le personnage depuis un ancien `story` seul.

Fichiers coordonnés : `config.toml`, `journal/src/ui.mjs` (PAGES, menu, navigation, rendu, capturePageDrafts, reset et renderGuard), `journal/src/worker.mjs` (routes), `netlify.toml` (avec/sans slash, proxy, headers), `static/js/article-likes.js` et textes d'accroche. Mettre à jour toutes les mentions « quotidienne », « une histoire par jour », « Mes histoires » dans les parcours actifs et documentations. Les titres historiques d'articles ne doivent pas être réécrits par un remplacement global aveugle.

## 10. Migration et déploiement futur sans perte d'historique

La migration SQL est un livrable critique. `feedback` référence `stories` avec `ON DELETE CASCADE` ; différer les contraintes étrangères ne désactive pas les cascades. Ne jamais remplacer la table parent sans préserver ses enfants.

1. Sur copie de recette : sauvegarde, inventaire des comptes/portraits/récits/retours, empreintes des corps et contrôle des liens. Préparer une version de transition Worker/runner compatible et des flags séparés pour lecture enrichie, génération à la demande et ancien ordonnanceur.
2. Ajouter les tables et colonnes compatibles, sans changer les comptes ni réinitialiser les abonnements.
3. Suspendre les nouvelles réservations de génération ancienne ; terminer ou invalider proprement les baux en cours. Aucune requête de l'ancien runner ne doit publier après bascule.
4. Créer les identifiants stables du slot 1 depuis le personnage existant, conserver l'identité et associer tous les récits dans l'ordre historique. Prévoir compte sans portrait, sans histoire, ou portrait incomplet : ne pas inventer une identité contredisant les archives ; état « à préparer » si nécessaire. **Les anciens portraits V1 ne deviennent pas des portraits V2 validés par simple renommage de champs.** Leur prose et leurs preuves sans révision ne sont pas des sources nouvelles. Extraire seulement l'identité fictive utile à la continuité, marquer le portrait « en actualisation » et reconstruire le portrait V2 depuis les sources originales. Si un aperçu historique est conservé temporairement, le signaler comme ancien et l'exclure du classement/brief V2. Histoires archivées lisibles pendant cette reconstruction ; nouvelles générations V2 seulement après validation du nouveau portrait.
5. Reconstruire `stories` et les références dépendantes avec le nouveau schéma dans une migration atomique éprouvée : préserver/copier explicitement `feedback` avant toute destruction de parent, garder IDs/corps/dates, recréer les FK/index, vérifier nombres et contenu. Reconstruire aussi les jobs si leur CHECK/unicité l'exige ; anciens jobs invalidés/archivés, aucun ancien quotidien transformé en demande utilisateur.
6. Importer les récits de la semaine de bascule comme consommations dans le nouveau registre, d'après leur `day` historique Europe/Paris, pour ne pas attribuer sept histoires en plus de celles déjà livrées cette semaine. Prévoir des clés déterministes pour rendre la migration rejouable sans doublons.
7. Déployer Worker et runner compatibles, puis UI/Netlify, conserver les secrets/bindings/modules existants. Désactiver définitivement la création quotidienne ; activer la demande explicite après recette.
8. Vérifier sur comptes synthétiques : historique et retours intacts, anciens liens, trois slots, nouvelle demande, quota, exports et suppression. Garder un kill switch sur les nouveaux jobs sans couper la lecture.

Retour arrière : désactiver les nouvelles demandes et revenir à la **version de transition compatible avec le schéma**, pas au vieux Worker qui utilise `ON CONFLICT(user_id,day)`. Après nouvelles écritures, ne pas restaurer aveuglément une sauvegarde qui perdrait les épisodes récents. Documenter un retour avant/après point de bascule et une procédure de réparation en avant.

La configuration Stripe et la récupération de mot de passe sont des dépendances commerciales déjà signalées dans `journal/README.md`. Leur disponibilité réelle sera revérifiée à la recette ; ce plan ne les suppose pas actives et n'autorise aucun achat. Le tarif d'abonnement ne constitue pas une estimation des coûts IA.

## 11. Tests, mesure de pertinence et critères de sortie

| Domaine | Cas obligatoires et preuve |
|---|---|
| Accueil | Matrice sans recherche/avec saisie/suggestion/Échap/Entrée, clic réel souris/touch/clavier, cible stable, 20 href valides, retour et refresh, mobile avec clavier. |
| Données | CRUD et no-op, conflits de révision, dates partielles, événement sans proche, suppression de proche conservant événement, export avec IDs, suppression du compte, isolation entre deux comptes et associations interdites. |
| Sources | Zéro like avec événement déclaré ; rôle seul sans diagnostic ; négation double ; différence sans « pas » ; désaccord partiel ; correction après le 120e caractère ; événement ancien réinterprété ; suppression d'une source ; preuve répétée non multipliée ; toute synthèse sourcée. |
| Classement | Comparaison baseline/V2 sur le corpus réel, pertinence annotée top-6/top-12 et NDCG, diversité parmi candidats pertinents, pas de remplissage nul, évitement explicite respecté, raisons fidèles, absence de fuite entre comptes. |
| Personnages | Trois identités stables, diversité des facettes sans étiquette imposée, même identité après réanalyse, aucun mélange entre sagas, événements anciens retrouvés après 30+ épisodes, résolution finale présente dans le résumé. |
| Quota | Sept succès, huitième refus, double clic, deux onglets/deux personnages simultanés, idempotence, retry completion, échec/crédit libéré, semaine Paris et changements d'heure, dimanche→lundi, expiration d'abonnement, portrait modifié en cours, bail expiré, compte supprimé. |
| Migration | Comptes vides et anciens, ancien portrait complet puis ancien portrait dont les sources ont été supprimées (aucune reconstruction inventée), conservation exacte des textes/IDs/retours, FK sans orphelins, crédits de transition corrects, redémarrage/reprise, rollback compatible. |
| Exploitation | File équitable, un compte défaillant ne bloque pas les autres, limites de coût/bail, pas de logs personnels, statuts honnêtes, alerte file bloquée/cron silencieux. |

Constituer avant ajustement au moins 30 scénarios français synthétiques/annotés, sans données réelles privées dans le dépôt. Réserver une partie des scénarios à l'évaluation finale. Comparer V1 actuelle et V2 sur les mêmes entrées ; ne pas valider les poids uniquement sur les exemples qui ont servi à les choisir. Exiger zéro violation des invariants de quota/isolation/provenance sur ces tests et aucune régression critique. Une amélioration de pertinence doit être montrée par classement annoté et exemples, pas seulement par des tests qui reproduisent la formule.

Pour les récits, plusieurs épisodes par personnage sur des profils contrastés, vérifiés avec une grille de continuité, pertinence, nuance et fidélité aux contraintes. Mesurer appels, tokens, coût observé, latence et reprises. Les tests hors ligne et synthétiques constituent une condition de mise en service, pas la preuve d'une compréhension parfaite des utilisateurs.

Commandes de baseline et de reprise, depuis la racine :

```bash
node --test tests/character-search.test.cjs
npm --prefix journal test
npm --prefix journal run check
python -m unittest discover -s journal/tests -p 'test_*.py'
python scripts/build_article_intros.py --check
hugo --minify --destination /tmp/decode-plan-check
```

Ajouter les tests de migration et concurrence sur SQLite/D1 local **réel**, puis en recette D1 ; les mocks du backend seuls ne prouvent pas l'atomicité. Ne pas lancer `build_examples.py` comme simple contrôle sans examiner ses écritures générées. Ne pas exécuter les scripts historiques de migration du corpus pendant ce chantier privé.

## 12. Ordre de travail et reprise par un modèle moins coûteux

| Étape | Livrable | Dépendances et contrôle avant de poursuivre |
|---|---|---|
| 0 | Relire ce plan, statut Git et éventuels changements récents ; fixtures de référence | Ne pas repartir d'une ancienne copie ni écraser une modification apparue depuis l'audit. |
| A | Accueil épuré + diagnostic/correctif du clic + ancre suggestions | Isolé ; recherche et navigation réelles vérifiées. Peut être livré séparément après autorisation d'implémenter. |
| B | Schéma/contrats, migrations de recette, primitives atomiques, sources CRUD | Tests données/concurrence et historique avant de brancher les nouveaux écrans. |
| C | Projection/extraction/synthèse sourcées, invalidation, zéro-like explicite | Fixtures de correction/suppression et validateurs Python/JS cohérents. |
| D | Catalogue et classement V2 | Évaluation comparative ; garder baseline activable pendant recette. |
| E | Registre des trois personnages, demandes/quota, routes et UI | Contrats B/C stables ; le runner ne lance plus de quotidien automatique. |
| F | Brief, continuité propre, pipeline contrôlé, budgets et feedback | C/D/E ; récits synthétiques, canon et crédit publiés atomiquement. |
| G | Recette bout en bout, migration répétée, bascule et vérification live | Seulement après implémentation autorisée, tests passés et préparation d'un retour compatible. |

Découpage de code recommandé : conserver le Worker comme routeur, extraire les fonctions de contexte, personnages et quotas dans des modules testables au fur et à mesure des lots, sans réécriture générale du site. Un seul responsable écrit `worker.mjs` ou `ui.mjs` à la fois. Interfaces et schémas d'abord ; paralléliser ensuite seulement les fichiers réellement indépendants. Chaque lot dispose de son diff, de ses preuves et de sa note de reprise.

Ne pas multiplier les changements de fournisseur, framework, base vectorielle, exécuteur ou facturation dans cette livraison. Les gains recherchés viennent d'abord de la qualité des sources, des contrats, des corrections, du contexte narratif et des transactions.

## 13. Revue du plan et limites honnêtes

Trois audits indépendants ont couvert interface/navigation, backend/migrations et algorithmique/génération. Une quatrième revue a challengé le plan consolidé. Leurs points contradictoires ont été intégrés : bug de clic encore conditionnel, distinction non-ressemblance/évitement, source explicite sans like, mémoire au-delà des derniers épisodes, risque de cascade SQL, reconstruction obligatoire des anciens portraits sans inventer des preuves, quota réservé, budget conservé lors d'une réanalyse et rollback de schéma compatible.

La revue finale a vérifié R1–R10, les courses de publication, les règles de semaine et l'absence de confusion entre vécu et fiction. Son avis est **prêt à implémenter**, après intégration de la reconstruction des portraits V1 et de son cas de test. Les décisions sont cohérentes et les tests pour les démontrer sont définis. **« Prêt à implémenter » ne signifie ni « déjà livré », ni « optimum mathématique », ni « efficacité psychologique prouvée ».** La pertinence et les budgets finaux seront ajustés à partir des évaluations prévues.

Références techniques primaires consultées pour les décisions d'infrastructure :

- [Cloudflare D1 — batch et transactions](https://developers.cloudflare.com/d1/worker-api/d1-database/).
- [Cloudflare D1 — clés étrangères et cascades](https://developers.cloudflare.com/d1/sql-api/foreign-keys/).
- [GitHub Actions — limites des événements planifiés](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

Les diagnostics spécifiques à Décode ton esprit proviennent du code du commit audité et des contrôles décrits ci-dessus.
