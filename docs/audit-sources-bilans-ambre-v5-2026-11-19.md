# Audit v5 — sources personnelles et bilans d'Ambre (19 novembre 2026)

Périmètre : usage par Ambre du portrait, de la synthèse « Ambre », de Mes proches (et des observations), de Ma ligne de vie et des bilans. Audit, recherche et propositions **sans modification du code** (aucun commit). Base : `main` à `a5ada4a`, plan v4.1 (`docs/plan-ambre-v4-2026-11-18.md`), avancements S et R.

Méthode : lecture du code (`journal/src/`), puis rejeu du **vrai moteur** (`createChatBackend({engineEnabled:true})`, SQLite en mémoire avec toutes les migrations, fournisseur simulé capturant chaque appel) sur un profil rempli réaliste. Scripts : `/tmp/claude-0/-home-user-Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/0d730b71-6a7b-5067-a403-c056872fe890/scratchpad/sources-v5/` (non versionnés).


Profil de mesure (« rempli réaliste ») : présentation 2 400 car. + perception, schémas, émotions, contexte 800 car. chacun (marqueurs de début et de fin dans chaque champ) ; 6 proches (rôles variés, 1 300 car., période) ; 12 moments (dates, âge, importance, proches liés, description 1 000 car., sens aujourd'hui 300 car.) ; cartographie verrouillée ; portrait V2 publié (synthèse 2 000 car., deux axes, état actuel, évolution, forces, incertitudes) ; 3 bilans prêts (initial, point 1, point 2) ; mémoire avec objectif, essai, alliance ; 8 échanges. Quatre messages types : un proche nommé (« Léa m'a coupé la parole… comme avec mon père »), une suite de bilan (« ce qu'on s'était dit la dernière fois »), un moment (« depuis le déménagement à Lyon »), le portrait (« comme vous l'aviez noté dans mon portrait »). Deux comportements simulés de l'analyse : ne choisit rien (`none`) et choisit « idéalement » le bilan, la synthèse, la présentation puis le proche et le moment cités (`ideal`).

Tests existants rejoués : les 19 fichiers qui touchent aux bilans, 129/129 réussis.

## 0. Constats majeurs (résumé)

| # | Constat | Preuve | Gravité |
|---|---|---|---|
| A | **Le moteur de production ne transmet pas le dernier bilan à chaque message.** `last_review` et `JOURNEY_GUIDANCE` n'existent que dans l'ancien chemin `generate()` (`chat-backend.mjs` l. 384–436), inactif depuis `engineEnabled:true` (`worker.mjs` l. 168). Le moteur ne place les 4 derniers bilans que comme lignes d'index `review` (`knowledgeIndex`, l. 182), sans date ni rang. Le README (« Le dernier bilan accompagne chaque tour ») et le test `journey.test.mjs` l. 94 (qui crée le service **sans** moteur) décrivent l'ancien chemin. | Simulation : dans 8 tours sur 8 (deux stratégies × quatre messages), aucun texte de bilan n'atteint la rédaction, même quand l'analyse le choisit. | Critique (demande 4) |
| B | **La sélection des détails ignore l'ordre de priorité de l'analyse.** `selectDetails` prend `index.filter(id choisi)` dans l'**ordre de l'index** (proches, puis moments, puis présentation, synthèse, cartographie, … bilans en dernier), puis coupe à 8 détails / 6 500 car., puis `while(...>6500) details.pop()` retire encore les derniers pour payer le plan de changement et les règles du lot S. | Simulation `ideal` : 8 identifiants choisis (bilan, synthèse, présentation, Léa, Marc, Lyon…) → 3 détails reçus : Léa, Marc, Lyon. Bilan, synthèse, présentation éliminés **à chaque fois**. | Critique |
| C | **Le portrait déclaré n'arrive presque jamais en entier.** La présentation est un seul texte `self_description: … self_perception: … recurring_patterns: … emotions: … current_context: …` coupé à 1 100 car. dans un détail et à 1 600 car. dans un bilan. Dès que « Ma présentation » dépasse ~1 000 car., **perception, schémas, émotions et contexte actuel ne sont jamais lus** (ni au tour, ni au bilan initial, ni au point). | Marqueurs `PERC_*`, `SCHEM_*`, `EMO_*`, `CTX_*` : absents de toutes les entrées mesurées. | Majeur (demande 1) |
| D | **La synthèse « Ambre » est en grande partie invisible pour Ambre.** L'index n'expose que `summary`, `strengths`, `uncertainties` ; les axes, « Ce que vos choix suggèrent aujourd'hui » (`current_state`) et « Évolutions observées » (`evolution`), affichés sur la page, ne sont transmis ni au tour ni au bilan. La synthèse choisie est coupée à 1 100 car. | Marqueurs `AXE_`, `ETAT_ACTUEL`, `EVOLUTION_` absents partout ; `SYNTH_FIN` absent. | Majeur |
| E | **Au point d'étape, le bilan ne voit plus les récits des proches ni des moments.** La boucle de réduction à 34 000 car. retire **toutes** les sources non-mémoire (textes des proches et des moments) avant de toucher aux listes de la cartographie, qui occupe 16–20 Ko (47–60 % du contexte). | Point simulé : `sources` = la seule présentation (1 618 car.) ; `people` = prénoms et rôles ; `events` = titres et dates ; cartographie 16 078 car. Bilan initial : 3 proches sur 6 et 2 moments sur 12 avec leur texte. | Majeur |
| F | **Les bilans n'ont pas de mémoire cumulée.** Le point reçoit le seul bilan précédent (5 000 car.) et la conversation depuis (6 échanges sur 8, tronquée) ; ni l'analyse initiale, ni les bilans plus anciens, ni la mémoire (`goals`, `tried`, `alliance`, `closed`, `safety_history`), ni le plan de changement (objectifs, étapes, essais), ni le suivi déclaré (prochaine séance). Le texte est en prose libre : rien n'est extrait pour suivre les objectifs d'un bilan à l'autre. | `MEM_GOAL`, `MEM_TRIED`, `BILAN_INIT`, `BILAN_P1` absents du contexte du point. | Majeur (demande 4) |
| G | **Ligne de vie : données utiles perdues à la transmission.** `current_importance` n'est pas dans les métadonnées d'index (`knowledgeIndex`) ; les proches liés n'y sont que des identifiants ; l'âge au moment de l'événement n'est jamais calculé alors que la date de naissance et la date sont connues ; la fin des descriptions est coupée (1 100 car.). | `DESC_FIN_ev5`, `SENS_ev5` absents de tous les tours. | Moyen |
| H | **Observations sur les proches : capture quasi nulle.** `isRelationReport` n'accepte qu'une phrase de 20–280 car. commençant par le prénom et suivant une quinzaine de tournures fixes (« me coupe la parole », « m'écoute »…), sans guillemets ni sujet sensible. La phrase type « Hier soir Léa m'a encore coupé la parole au dîner et je me suis sentie minuscule… » ne produit aucune note. | Simulation : `observation_candidates` vide. | Moyen |

Coût mesuré d'un tour nominal sur ce profil (analyse + rédaction + relecture, caractères / 3, hors schémas) : 9 450 à 9 880 jetons d'entrée ; rédaction 15 700–16 800 car. ; détails 3 à 4 (sur 8 possibles), chacun plafonné à 1 100 car. Un bilan de point : ≈ 12 500 jetons d'entrée, 3 200 de sortie au plus.

## 1. Sources : formulaires, transmission réelle, pertes

### 1.1 Mon portrait (présentation déclarée)

**Formulaire** (`ui.mjs` l. 406, validation `personal-backend.mjs` l. 48–60) : prénom ou pseudonyme, nom, date de naissance, genre (+ description libre 200), activité (4 000) ; textes libres « Ma présentation » (10 000), « Comment je me perçois » (4 000), « Les schémas que je remarque » (4 000), « Les émotions présentes » (4 000), « Mon contexte actuel » (4 000).

**Ce qu'Ambre reçoit à chaque tour** : `identity` (prénom, genre) et `self_context` (âge calculé, genre, activité) toujours ; la présentation n'est qu'**une ligne d'index** `profile Votre présentation` et, si elle est choisie **et** survit au tri B, un extrait de 1 100 car. du texte concaténé (donc le début de « Ma présentation »). En pratique (simulation) : jamais reçue.

**Bilans** : extrait de 1 600 car. du même texte (début de la présentation seulement).

**Pertes** : constats B et C. Les cinq rubriques sont des questions cliniquement distinctes (identité, image de soi, schémas, état affectif, contexte actuel) que la concaténation transforme en un seul bloc dont seul le début compte.

**Ce qui manque pour un psychologue** (entretien initial, formulation de cas : motif, problèmes actuels, facteurs précipitants et de maintien, ressources, attentes) : *ce qui m'amène / ce que je voudrais changer* (demande et objectifs en ses mots ; aujourd'hui seulement dans la conversation) ; *ce qui m'aide déjà* (ressources actuelles, distinctes des ressources d'enfance de la cartographie) ; *mon rythme de vie* (sommeil, travail, activité, consommations : retentissement fonctionnel, sans chiffres de poids ni calories) ; *mes suivis et traitements* (déclaratif, facultatif, sensible : seulement si la personne le souhaite) ; *ce que je préfère dans la façon d'échanger* (déjà en partie dans `memory.style`).

### 1.2 Synthèse « Ambre » et pistes

**Contenu affiché** (`appendPortrait`, `ui.mjs` l. 202) : résumé, état actuel, évolution, axes (libellé, pourquoi, niveau de confiance en mots), forces, incertitudes ; carte des pistes (`psychological_map` : statut, observation, fonction possible, alternatives, question suivante, preuves, contre-indices, influence, interactions).

**Transmis au tour** : ligne `portrait:summary` (résumé + forces + incertitudes, extrait 1 100 car. si sélectionnée) ; une ligne `hypothesis` par piste (libellé seul dans l'index ; JSON complet coupé à 1 100 car. si sélectionné). **Jamais** : axes, état actuel, évolution. **Bilans** : résumé, forces, incertitudes, `psychological_hypotheses` (pistes) ; pas les axes ni l'état actuel ni l'évolution.

**Mal utilisé** : la `next_question` de chaque piste et les contre-indices sont exactement ce qu'un psychologue apporte d'une séance à l'autre (hypothèse de travail + test suivant, Persons), mais ils ne sont lus que si la piste est choisie et survit au tri.

### 1.3 Mes proches

**Formulaire** (`ui.mjs` l. 631) : prénom ou pseudonyme (80), « Cette personne est… » (père, mère, parent, partenaire, enfant, frère ou sœur, ami, meilleur ami, autre + précision 80), « Comment vivez-vous cette relation ? » (6 000, libre), « Depuis » et « Jusqu'à » (80, libres). Observations automatiques (`people_observations`, 6 axes, 60 par proche) affichées et supprimables.

**Transmis au tour** : 6 lignes d'index `person Prénom — rôle` (rôle en code anglais : `sibling`, `father`) ; détail = JSON {rôle, période, prénom, texte} coupé à 1 100 car. (≈ 950 car. de récit) ; observations du proche **seulement s'il est nommé** dans le message (6 au plus). **Bilan initial** : liste prénom/rôle/période (20 au plus) + textes des proches qui tiennent dans 10 500 car. (3 sur 6 mesurés) ; **point** : aucun texte (constat E) ; 24 dernières observations.

**Ce qui manque** (génogramme de McGoldrick et Gerson : lignes de *qualité émotionnelle* distinctes des liens structurels — proximité, distance, conflit, rupture, fusion, soutien, méfiance — ; modèle du convoi de Kahn et Antonucci : cercles de proximité qui évoluent au cours de la vie) :
- *Nature et qualité du lien* : choix multiples en mots (proche, distant, conflictuel, coupé, fusionnel, soutenant, ambivalent, je ne sais pas), sans score.
- *Place aujourd'hui* : présent au quotidien / régulièrement / rarement / plus de contact / décédé.
- *Ce qui m'aide dans cette relation* et *ce qui me blesse ou me pèse* (deux champs courts, 600 car.).
- *Comment le lien a évolué* (court, ou repères liés à la ligne de vie).
- *Sécurité* : « Je me sens en sécurité avec cette personne : oui / pas toujours / non » (alimente les signaux IPV existants sans confrontation).
- Liens entre proches (facultatif, plus tard) : « en conflit avec », « allié de » — seulement si la personne le souhaite.

### 1.4 Ma ligne de vie

**Formulaire** : titre (160), type (difficile, positif, tournant, ambivalent, autre), repère temporel (sans date, année, mois, jour, période, âge approximatif), description (6 000, requise), « Ce que cela signifie aujourd'hui » (2 000), « Place dans votre vie aujourd'hui » (1 à 5), proches associés (10).

**Transmis au tour** : 12 lignes d'index (titre seul) ; détail = JSON {type, dates, âge texte, identifiants des proches, description, sens} coupé à 1 100 car. ; `current_importance` absente de l'index ; aucun âge calculé ; proches en identifiants. **Bilans** : titres, dates et identifiants des proches ; textes de 2 moments sur 12 à l'initial, aucun au point.

**Ce qui manque** (life chart du NIMH de Leverich et Post : événements psychosociaux placés sur une frise avec l'état de la personne à chaque période ; Life Story Interview de McAdams : chapitres, point haut, point bas, tournant ; LEC-5 : *comment* l'événement a été vécu — m'est arrivé, j'en ai été témoin, je l'ai appris pour un proche — sans exiger de détails) :
- *Âge à ce moment* calculé automatiquement depuis la date et la date de naissance (affiché, transmis).
- *Ce que j'ai ressenti alors* et *ce que je ressens aujourd'hui en y repensant* (le second remplace ou précise « Ce que cela signifie aujourd'hui »).
- *Ce que cela a changé pour moi* (impact, court).
- *Le soutien que j'ai reçu* (aucun / un peu / beaucoup + qui, lié aux proches).
- *Comment je l'ai vécu* : m'est arrivé / j'en ai été témoin / je l'ai appris (LEC-5), facultatif.
- *Chapitres* de vie (facultatif) : périodes nommées qui regroupent les moments.
- Garde-fou existant à conserver : la description reste un récit choisi ; aucun détail traumatique demandé.

### 1.5 Observations sur les proches

Le vocabulaire fermé protège la vie privée mais rend l'outil quasi inutile (constat H). Proposition : garder les gardes (danger, santé, sensible, fiction, refus) et remplacer le motif fixe par une **proposition validée** : Ambre propose « Noter pour Léa : “elle me coupe la parole au dîner” ? » ; la note n'est écrite qu'après confirmation (même mécanique que `makeProposal`, citation exacte non niée).

## 2. Bilans

### 2.1 Fonctionnement réel

- **Déclenchement** : uniquement à la demande. `POST /api/chat/start` crée l'analyse initiale ; `POST /api/chat/review` un point. Premier point au plus tôt 1 h après l'analyse initiale, puis 24 h entre deux points ; 4 bilans par jour au plus ; un seul en préparation (les messages sont bloqués `409 review_in_progress` pendant ce temps). Aucun bilan suggéré ni lié à une séance déclarée.
- **Exécution** : tentative immédiate de 25 s sous bail, reprise par le cron (lots de 5, 3 tentatives), garde fournisseur commune, réservation hebdomadaire et quotidienne, refus remboursés (lot R).
- **Contexte** (`reviewContext`) : identité, présentation (extrait), synthèse (résumé, forces, incertitudes), pistes, proches, moments, cartographie complète, 24 observations, couverture ; pour un point, **le seul** bilan précédent (5 000 car.) et les échanges depuis ce bilan (10 000 car., 40 au plus). Plafond 34 000 car. ; réduction dans l'ordre : échanges (jusqu'à 6), notes d'Ambre, pistes, repères contextuels, conséquences exploratoires, preuves, puis **toutes les sources**, puis proches et moments.
- **Consignes** (`psychology-framework.mjs` l. 49–57) : 600–900 mots, prose sans titres ni listes, hypothèses citées, sources manquantes, sécurité ; initial = fonctionnements, forces, zones sensibles, histoire/présent, à éclaircir, une question ; point = évolutions, ce qui se confirme ou s'infirme, objectifs, essais réels, à éclaircir, 2–3 questions, 1–2 pistes, une question.
- **Conservé** : texte (1 200–14 000 car.), `sources` (comptes et identifiant du bilan précédent), message publié dans la conversation, export, suppression, page « Mes bilans », partage au psychologue (rubrique `reviews`).
- **Contrôles** : longueur seulement. Pas de relecture critique (les 15 critères ne s'appliquent pas aux bilans), pas de vérification des citations, pas de contrôle des numéros, alors que le texte est long et publié tel quel.

### 2.2 Vérification

Les 129 tests des fichiers concernés passent ; la simulation publie bien un point (`ready`, 2 472 car., `previous_review_id` = dernier bilan). Le mécanisme fonctionne ; c'est son **usage** qui est défaillant : constats A, E, F, et absence de relecture.

### 2.3 Ce qu'est un bon bilan d'étape

- **Formulation de cas comme hypothèse révisée** : Persons traite chaque cas comme un test d'hypothèse — formulation des mécanismes qui causent et maintiennent les difficultés, intervention guidée par elle, recueil de données pour la tester, révision ; ce suivi permet de repérer tôt un traitement qui échoue. Kuyken, Padesky et Dudley : conceptualisation collaborative, descriptive puis transversale puis longitudinale, qui évolue dans le temps et intègre explicitement les forces et la résilience.
- **Objectifs et progrès suivis** : Goal Attainment Scaling (Kiresuk et Sherman) : objectifs individualisés, chacun avec des niveaux d'atteinte définis à l'avance, aujourd'hui souvent choisis et évalués par la personne.
- **Retour de la personne et alliance** : le suivi des résultats avec retour (OQ, PCOMS) donne un effet faible à modéré, plus net pour les personnes qui ne progressent pas (méta-analyses de Lambert et al. 2018, de Jong et al. 2021) ; l'alliance est liée au résultat (r ≈ 0,28 sur 295 études, Flückiger et al. 2018). Le Session Rating Scale demande à la personne si elle s'est sentie entendue, si l'on a travaillé ce qui comptait pour elle, si l'approche lui convient.
- Conséquence pour Ambre : un bilan utile n'est pas un résumé, c'est **une formulation versionnée, des objectifs suivis avec un état, ce qui a été réellement essayé et son effet, le retour de la personne sur l'accompagnement, ce qui change dans la suite**. Les mesures chiffrées sont exclues par la charte (aucun score) : on garde des niveaux en mots.

### 2.4 Structure proposée du bilan

Sortie JSON du modèle : `content` (texte publié, inchangé dans sa forme : paragraphes, sans titres ni listes) **et** `digest` (non affiché tel quel, ≤ 2 000 car., chaque élément ancré par une citation exacte de la personne ou marqué « hypothèse d'Ambre ») :
- `formulation` : {`holds` (ce qui se confirme), `revised` (ce qui est nuancé ou contredit, avec la phrase qui l'a nuancé), `new` (nouvelle hypothèse)}, 3 éléments chacun au plus ;
- `goals` : [{`key` stable, `words` (ses mots), `status` ∈ nouveau, en cours, avance, atteint, en pause, abandonné, `evidence` (citation), `next_step`}], 5 au plus (aligné sur les `goals` du plan de changement quand ils existent) ;
- `tried` : [{`what`, `result` ∈ a aidé, sans effet, difficile, pas encore essayé, `evidence`}] ;
- `alliance` : retour de la personne sur l'accompagnement (citation) ou vide ;
- `open_questions` (2–3), `next_focus` (ce sur quoi la suite portera), `source_gaps`, `safety` (rappel si nécessaire).

Consigne du point modifiée en conséquence : commencer par ce qui a changé ; reprendre **chaque objectif suivi** avec son état ; distinguer proposé / essayé / résultat ; dire ce que la formulation garde et ce qu'elle révise ; finir par la question et l'invitation à corriger le bilan.

### 2.5 Fil des bilans (mémoire cumulée)

Nouvelle valeur dérivée, recalculée à chaque bilan publié (déterministe, sans appel modèle) : `review_thread` = {dernier bilan : date, type, `next_focus`, 2 lignes de formulation ; objectifs : pour chacun la suite des états datés (« 2026-11-01 nouveau → 2026-11-08 en cours → 2026-11-15 avance »), dernière preuve ; ce qui se confirme depuis au moins deux bilans ; ce qui a été révisé ; essais et effets}. Bornes : 1 200 car. (≈ 400 jetons) au tour, 3 000 car. au bilan ; au-delà, les objectifs atteints ou abandonnés depuis plus de deux bilans sortent les premiers. Supprimer un bilan recalcule le fil ; l'export et Paramètres l'affichent ; la personne peut corriger un état d'objectif.

### 2.6 Usage à chaque tour

- La rédaction reçoit **toujours** `review_thread` (comme `memory`), hors de la concurrence des détails ; consigne courte : « s'appuyer sur le fil sans le réciter, tenir compte de ce que la personne a nuancé depuis, revenir à l'objectif en cours quand c'est pertinent, jamais quand une contrainte `no_exploration`, `listen` ou un danger s'applique ».
- L'analyse reçoit 2 lignes (objectif en cours, `next_focus`) pour choisir les détails.
- Le texte intégral du dernier bilan reste une ligne d'index datée (`review 2026-11-15 Point d'étape`) consultable en détail.
- La relecture reçoit les objectifs en cours pour vérifier qu'Ambre ne contredit pas un état sans raison.

## 3. Propositions priorisées

### P0 — corriger ce qui est cassé (sans migration, coût stable)

1. **Fil minimal du dernier bilan à chaque tour** : en attendant `digest`, transmettre `last_review` borné à 900 car. (début et fin du dernier bilan : la fin contient les pistes et la question) dans `data` de la rédaction ; paiement par `details` 6 500 → 5 600 car. Test : sur le moteur réel, le marqueur du dernier bilan est dans l'entrée de rédaction pour 4 messages types ; budget ≤ +10 % (`v4-S-budget`).
2. **Ordre des détails = ordre de l'analyse** : dans `selectDetails`, parcourir `ids` dans l'ordre fourni, puis le classement lexical ; dans la réduction `while(...) details.pop()`, retirer d'abord les détails non choisis par l'analyse. Test : avec 8 identifiants choisis, les 3 premiers survivent.
3. **Présentation découpée par rubrique** : une entrée d'index par champ (`profile:self_description`, `profile:self_perception`, …) avec son libellé de formulaire ; détail de 600 car. par rubrique, début + fin. Même découpage dans `promptContext` des bilans (une source par rubrique). Les empreintes du portrait (`source.hash`) restent celles de la source globale : le découpage ne touche que la présentation au modèle.
4. **Synthèse complète dans l'index** : entrées `portrait:axes`, `portrait:current_state`, `portrait:evolution` (et les axes dans les bilans). Test : marqueurs présents quand le message porte sur le portrait.
5. **Réduction du contexte des bilans** : retirer d'abord les preuves et repères de la cartographie et raccourcir chaque source (1 600 → 800 car.) **avant** de supprimer les récits des proches et des moments ; réserver 30 % du plafond aux proches et moments. Test : au point, au moins 6 proches et 8 moments avec texte sur le profil rempli.
6. **Mettre la documentation et les tests au niveau du moteur** : README (`last_review`), test de `journey` rejoué avec `engineEnabled:true`.

### P1 — bilans utiles et suivis (migration 0023, additive)

7. `ALTER TABLE ambre_reviews ADD COLUMN digest TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(digest))` ; `ALTER TABLE ambre_reviews ADD COLUMN feedback TEXT` (retour de la personne : « Ce bilan vous ressemble » oui / en partie / non + correction libre 600 car., sans compteur) ; table `review_goal_states(user_id, goal_key, review_id, status, evidence, created_at)` ou calcul à la volée depuis les `digest` (préférable : pas de table, fil dérivé et mis en cache comme l'index).
8. Contexte du point enrichi : mémoire (`goals`, `tried`, `alliance`, `closed`, `safety_history`), plan de changement (`serializeSharedPlan` : objectifs, étapes, essais), suivi déclaré (prochaine séance), fil des bilans (3 000 car.), retour de la personne sur le bilan précédent.
9. Validation du `digest` : citations exactes (`quote-guard.mjs`, non niées, ≥ 12 car.), états dans la liste fermée, objectifs rattachés aux objectifs connus ou explicitement nouveaux ; un `digest` invalide n'empêche pas la publication du texte (il est vidé et compté).
10. **Relecture du bilan** : les critères 1–4, 9–11 et les contrôles déterministes (numéros vérifiés, diagnostic, enregistrement annoncé) appliqués au texte, une réécriture au plus, sinon publication avec repli prudent. Coût : un petit appel (≈ 3 000 jetons) par bilan, pas par message.
11. **Suggestion de point** (jamais automatique) : Ambre peut proposer « faire le point » après 15 échanges depuis le dernier bilan ou dans les 48 h avant une séance déclarée ; le bouton existant suffit.

### P2 — formulaires

12. Proches (migration 0024 : colonnes additives, toutes facultatives) : `bond_quality` (liste en mots, plusieurs choix), `presence_now`, `helps` (600), `hurts` (600), `evolution` (600), `feels_safe`. Transmis en métadonnées d'index (libellés courts) et en tête du détail.
13. Moments : `felt_then` (600), `felt_now` (600, ou réutiliser `meaning_now` relibellé), `impact` (600), `support` (aucun / un peu / beaucoup + proches), `exposure` (m'est arrivé / témoin / appris), âge calculé (non stocké). `current_importance` et l'âge ajoutés aux métadonnées d'index ; proches liés par prénom.
14. Portrait : rubriques « Ce qui m'amène / ce que je voudrais changer », « Ce qui m'aide déjà », « Mon rythme de vie » (facultatives, 2 000).
15. Observations : proposition validée au lieu du vocabulaire fermé (§ 1.5).

Ces formulaires respectent les choix publiés : champs facultatifs repliés, aucune introduction sous les titres, aucune échelle chiffrée ni compteur affiché (l'importance 1–5 existante pourrait passer en mots : peu présent / présent / très présent).

### Budget en jetons (rédaction, par tour)

| Bloc | Aujourd'hui | Proposé |
|---|---|---|
| `details` | ≤ 6 500 car. (mesuré 3–4 détails × 1 100) | ≤ 4 900 car. (5–7 détails × 700, ordre de l'analyse) |
| Fil des bilans | 0 | ≤ 1 200 car. |
| Repères permanents (objectif en cours, 3 lignes de synthèse, cercle des proches : prénom, lien, présence) | 0 | ≤ 400 car. |
| Total | 6 500 car. | 6 500 car. (coût stable, ±0 %) |

Les nouveaux blocs fixes sont placés avant la partie variable pour profiter du cache fournisseur. Analyse : +2 lignes (≈ 60 jetons) prises sur l'index (2 600 car.). Bilan : +3 000 car. de fil et +1 500 de mémoire et plan, compensés par la réduction de la cartographie (§ P0.5) : plafond de 34 000 inchangé.

### Transparence, consentements, révisions

Le fil des bilans et le `digest` apparaissent dans Paramètres (« Ce qu'Ambre retient de vos bilans »), dans `GET /api/chat/insight` et dans l'export ; supprimer un bilan le retire du fil. Les nouveaux champs suivent les révisions et la clé idempotente existantes et invalident le portrait comme les champs actuels. Le partage au psychologue (`reviews`) peut inclure le fil (plus utile qu'une pile de textes) avec l'accord déjà prévu.

### Tests à ajouter

`v5-sources.test.mjs` (profil rempli, moteur réel) : chaque rubrique du portrait atteignable ; axes/état/évolution présents ; ordre des détails ; dernier bilan et fil présents dans 4 messages types ; budget ±10 % contre `v4-S-budget`. `v5-reviews.test.mjs` : contexte du point contenant mémoire, plan, fil, proches et moments avec texte ; `digest` validé (citation inventée rejetée, état inconnu rejeté) ; suppression d'un bilan recalcule le fil ; relecture du bilan (numéro non vérifié réécrit) ; retour de la personne transmis au bilan suivant. Migration 0023/0024 rejouée sur une base à 0022.

## 4. Sources

- McGoldrick, Gerson, Petry, *Genograms: Assessment and Intervention* (lignes de relation émotionnelle) : [EBSCO Research Starters](https://ebsco.com/research-starters/social-sciences-and-humanities/genogram), [TherapistAid](https://therapistaid.com/therapy-guide/genograms).
- NIMH Life Chart Method (Leverich, Post) : [manuel rétrospectif](https://www.gatewaypsychiatric.com/wp-content/uploads/2014/04/PatientRetrospectiveManual.pdf), [validation NIMH-LCM-p](https://resolve.cambridge.org/core/journals/psychological-medicine/article/validation-of-the-prospective-nimhlifechart-method-nimhlcmtmp-for-longitudinal-assessment-of-bipolar-illness/8E493183C61D28841E09B22B51B01C3A).
- McAdams, Life Story Interview : [plan](https://web.lemoyne.edu/~hevern/psy444_24F/documents/The%20Life%20Story%20Interview%20Outline%2020241022.pdf), [consignes](https://web.lemoyne.edu/~hevern/psy444_24F/documents/Life%20Story%20Interview%20Instructions.pdf).
- Modèle du convoi et cartes de réseau biographiques : [FQS 2022](https://qualitative-research.net/index.php/fqs/article/download/3715/4800?inline=1).
- LEC-5 (Weathers et al., 2013) : [NovoPsych](https://novopsych.com/assessments/diagnosis/life-events-checklist-5-lec-5/), [Weathers et al.](https://www.policingtrauma.sociology.cam.ac.uk/files/weathers_et_al_2013_lec5.pdf).
- Persons, formulation de cas : [Guilford](https://www.guilford.com/books/The-Case-Formulation-Approach-to-Cognitive-Behavior-Therapy/Jacqueline-Persons/9781462509485), [Persons et Tompkins 2020](https://jacqueline-persons.squarespace.com/s/Persons-Tompkins2020Case-formulation-chapterEells-third-edition.pdf), [Persons, Beckner, Tompkins](https://escholarship.org/content/qt6g80q15k/qt6g80q15k.pdf?t=rwpr52).
- Kuyken, Padesky, Dudley, *Collaborative Case Conceptualization* : [Guilford](https://www.guilford.com/books/Collaborative-Case-Conceptualization/Kuyken-Padesky-Dudley/9781462504480), [article « crucible »](https://www.cambridge.org/core/product/B7BB53DDFB77CC42572ED069B20835EC).
- Goal Attainment Scaling : [Kiresuk et Sherman 1968](https://idainstitute.com/fileadmin/user_upload/documents/Kiresuk%20and%20Sherman%20Goal%20Attainment%20Scaling.pdf).
- Suivi des résultats : [Lambert, Whipple, Kleinstäuber 2018](https://digitalcommons.usu.edu/psych_facpub/1871), [Barkham et al. 2023](https://eprints.whiterose.ac.uk/197689/13/Routine%20Outcome%20Monitoring%20ROM%20and%20Feedback%20Research%20Review%20and%20Recommendations.pdf), de Jong et al. 2021 (*Clinical Psychology Review* 85).
- Alliance : [Flückiger et al. 2018](https://www.societyforpsychotherapy.org/wp-content/uploads/2018/10/Fluckiger-et-al-2018.pdf), [Norcross et Lambert 2018](https://www.ovid.com/journals/psythy/pdf/10.1037/pst0000193~psychotherapy-relationships-that-work-iii) ; Session Rating Scale : [CORC](https://corc.uk.net/outcome-experience-measures/session-rating-scale-srs/).

## 5. Limites

Fournisseur simulé : les choix de l'analyse sont imposés (deux stratégies extrêmes), pas observés sur un vrai modèle ; les tailles en jetons sont estimées (caractères / 3). Les propositions de formulaires n'ont pas été testées auprès de personnes. Aucun fichier du dépôt modifié en dehors de ce rapport ; aucun commit.
