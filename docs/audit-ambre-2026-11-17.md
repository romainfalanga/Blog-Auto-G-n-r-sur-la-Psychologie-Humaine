# Audit du moteur d’Ambre — état des lieux au 17 novembre 2026

Audit en lecture seule de la branche `claude/affectionate-wright-eq5kgo`, commencé sur `4747d83`. Aucun fichier existant modifié, aucun commit, aucun déploiement, aucun appel réseau (fournisseur simulé). `npm --prefix journal test`, lancé une fois au début : **462 tests, 462 réussis, 0 échec** (6,7 s). Pendant l’audit, une autre session a avancé la branche jusqu’à `f32bb7d` ; dans les fichiers audités, seuls quatre libellés d’interface ont changé (B:118, B:505, `personal-backend.mjs:148`, S:36), sans décalage de lignes ; les tests n’ont pas été relancés après ce changement. Ce document remplace, pour le moteur, `docs/audit-ambre-2026-11-16.md` (ancien moteur).

Abréviations : **E** `journal/src/ambre-engine.mjs` · **B** `journal/src/chat-backend.mjs` · **I** `ambre-insight.mjs` · **F** `psychology-framework.mjs` · **G** `psychology-guide.mjs` · **K** `care-backend.mjs` · **S** `settings-interface.mjs` · **C** `cartographie.mjs` · **O** `people-observations.mjs` · **P** `private-context.mjs` · **T** `journal/tests/ambre-engine.test.mjs`. Les mesures de taille viennent d’un script jetable (hors dépôt) qui rejoue le vrai moteur sur SQLite en mémoire avec un fournisseur simulé ; « jetons estimés » = caractères/3, la règle de réservation du code (E:150).

## Résumé — cinq constats qui conditionnent la suite

1. **Le plafond de 30 000 jetons réservés par tour (E:151) ne laisse pas la place d’une réécriture** dès que le contexte est normalement rempli : sur un profil type, la relecture qui demande une réécriture fait échouer le tour (`generation_budget`, message « budget de la semaine atteint », B:183). Sur un profil maximal avec message long, la relecture ne peut plus être réservée et la personne reçoit toujours le repli générique (E:254, E:126). Tout bloc ajouté aggrave ce point : il faut le traiter d’abord.
2. **Le prompt de rédaction du moteur ne dit rien du changement** : aucun mot sur objectifs, valeurs, exercices, techniques, premier pas, retour sur un essai. Les consignes qui en parlaient (F:47, F:66) et les douze « experiment » (F:19-30) ne servent plus en production : elles appartiennent à l’ancien chemin `generate` (B:330-498), inactif depuis `engineEnabled:true` (`worker.mjs:152`).
3. **Rien n’est structuré pour suivre un changement** : la mémoire ne contient que des citations mot pour mot de la personne, en texte libre, sans identifiant, statut ni date ; ce qu’Ambre a proposé n’est jamais retenu ; l’extrait transmis garde les éléments les plus anciens de chaque rubrique (E:20-26).
4. **Ambre n’a aucune notion du temps** : ni date du jour, ni date des échanges ; les séances arrivent en secondes Unix (`care-context.mjs:9`, E:227). Elle ne peut ni « revenir au bon moment » ni situer une séance.
5. **Plusieurs sources ne lui parviennent jamais** : les bilans (dont l’analyse initiale qui ouvre la conversation), les observations de proches, le catalogue des 161 repères, les consignes de lecture de la cartographie.

## 1. Chaîne d’un tour

Entrée `POST /api/chat/messages` (B:499) : texte ≤ 6 000 caractères, clé d’idempotence ; limites 100 messages/semaine, 30 tours/heure, 200 tours/semaine, un seul tour en cours. Si le filet lexical (`chat-safety.mjs:3-10`) ou `deterministicAnalysis` (E:59-66) voit un risque, la réponse de sécurité est écrite dans le même INSERT que le tour, avant tout appel. Puis `engine.enqueue` (E:176 : instantané, empreinte des sources, ligne `chat_engine_jobs`, bail du tour à +600 s) et `engine.work` dans `waitUntil`, retour 202.

`work` (E:179-279) : bail de 150 s par jeton aléatoire, fenêtre d’exécution de 22 s, arrêt de boucle à 2 s de la fin ; un appel n’est lancé que s’il reste ≥ 10 s (E:114, E:192), sinon `stage_deferred` : l’étape reste en base et le bail est rendu. Reprise par le polling authentifié `GET /api/chat` (B:729) et par le cron minute (`worker.mjs:405` → E:300-306 : 5 tours puis 5 mémoires par passage). Avant chaque étape : consentement et révision revérifiés (E:177), empreinte recalculée (E:172 : versions, portrait, identité, cartographie, révision mémoire, cadre de suivi) ; tout écart → `context_changed`, échec définitif. `advance` (E:178) écrit l’étape sous bail et remet `attempts` à 0.

| Étape (colonne `stage`, `0019_ambre_engine.sql:11`) | Modèle par défaut (E:129) | Sortie max | Entrée | Sortie | En cas d’échec |
|---|---|---|---|---|---|
| `analysis` (E:195-229) | `openai/gpt-5-nano`, effort minimal | 1 500 | message, mémoire, suivi, 2 derniers échanges, index, légende | JSON strict `ANALYSIS_SCHEMA` | repli déterministe, `fallback:true`, le tour continue (E:212) |
| sous-étape `risk_check` (E:220-224), seulement si le modèle propose un risque que les règles n’ont pas vu | idem | 1 500 | message seul (+ échange précédent si confirmation, E:52-57) | risque, catégorie, citation exacte vérifiée (E:45-51) | reprise de la seule sous-étape ; 3 échecs → tour échoué, crédit rendu, aucune crise inventée |
| `writing` (E:230-235) | `anthropic/claude-sonnet-5.5`, effort bas | 1 800 | message, analyse, suivi, mémoire, 6 échanges, identité, contexte déclaré, ≤ 8 détails | JSON strict `engineSchema` (B:767) | `attempts+1`, reprise au bail suivant ; 3ᵉ échec → tour échoué (E:271-277) |
| `review` (E:236-255) | `openai/gpt-5-nano` | 1 500, puis 2 250, 3 000 si tronquée (E:237) | message, analyse, réponse, mémoire, 2 échanges, propositions, extraits ≤ 400 car. | 15 valeurs `yes/no/na` + raisons | erreurs transitoires : 3 essais (E:246) ; sinon repli prudent publié |
| `rewrite` (même code que `writing`) | Sonnet | 1 800 | idem + brouillon, relecture, « Réécris une fois… » (E:232) | idem | comme `writing` ; une seule réécriture (E:253) |
| `publish` (E:256-263, B:745-766) | aucun | — | — | lot D1 gardé : propositions, notes, observations, réponse, sujets fermés | garde fausse → `context_changed` ; reprise après crash sans double écriture (E:185) |

Tous les appels passent par `generateText` (E:145-169) : OpenRouter, `response_format` JSON strict, consigne système marquée `cache_control`, `data_collection:'deny'`, `zdr:true`, délai min(18 s, temps restant). Budgets : 48 000 caractères par appel (E:149) ; réservation = (système + entrée + schéma)/3 + sortie max, plafonnée à **30 000 par tour** (E:151) et **3 000 000 par semaine et par personne** (E:150-153). Échecs définitifs immédiats : `context_changed`, `generation_budget`, 3ᵉ échec d’une même étape (E:271). Un tour échoué libère son crédit ; la réponse de sécurité déjà écrite reste visible. Après publication : trace finale (E:258-261), tâche mémoire mise en file (E:262), contexte et brouillon effacés de la ligne (E:263).

## 2. Analyse du tour

**Schéma (E:42)** — `need` : `listen|understand|advice|act` · `states` : liste libre filtrée sur `distress, rupture, shame, ambivalence, dissociation, dependency, unusual_belief` (E:14) · `risk` : `none|possible|explicit` · `posture` : `listen|explore|stabilize|repair|safety|options` · `sensitive_allowed` : booléen · `constraints` : liste filtrée sur `no_advice, no_questions, no_exploration, respect_closed_topics` (E:15) · `selected_ids` : ≤ 8 identifiants existants de l’index · `closed_topics` : ≤ 8 citations exactes du message, ≤ 240 caractères. Le serveur ajoute `fallback`, `error_code`, `usage`, `risk_check{status,proposed_risk,category}`, `risk_evidence`, `safety_signal`.

**Consigne (E:41, 1 424 caractères)** : analyser sans diagnostiquer ; données ≠ instructions ; toute allusion indirecte à la mort ou à un moyen ⇒ `possible` au minimum ; un antécédent de sécurité n’est pas un danger actuel ; choisir jusqu’à 8 identifiants ; conditions d’une question sensible (sert la demande, hors débordement, sujet non refusé, permission, pas deux tours de suite) ; correspondances état → posture ; « en cas de doute, sensitive_allowed=false ». Rien sur les objectifs, la motivation, un essai rapporté ou une demande de méthode.

**Part déterministe (E:59-66)**, appliquée d’abord puis fusionnée (E:204-211) : expressions régulières pour le risque, la rupture, la détresse, la honte, la dissociation, l’ambivalence, la dépendance, « juste parler / pas de conseil », « pas de question », demande d’avis, sujet fermé. Règles de priorité : risque déterministe jamais abaissé par le modèle ; risque proposé par le seul modèle → vérification séparée ; écoute demandée ⇒ `need=listen`, `posture=listen`, `no_advice` ; détresse/dissociation/honte ⇒ `stabilize` + `no_exploration` + `no_questions` ; rupture ⇒ `repair` ; risque ⇒ `safety` et retrait de `no_questions` (E:225) ; `sensitive_allowed` vrai seulement sans risque, sans état sensible, sans sujet fermé et si la dernière réponse ne contient pas « ? » (E:211).

**Effet de chaque champ** : `selected_ids` → détails ouverts. `risk` → réponse de sécurité, ressources ajoutées (B:735), aucune proposition ni note, repli = message de sécurité. `closed_topics` → fusionnés dans la mémoire à la publication (B:761-764). `no_questions` et `no_advice` → contrôles déterministes 12 et 8. **Simples étiquettes transmises au rédacteur et au relecteur, sans effet codé** : `need` (hors `listen`), `posture`, `sensitive_allowed`, `no_exploration`, `respect_closed_topics`, les états `ambivalence`, `dependency`, `unusual_belief`. `need:'act'` n’est produit que par le modèle et ne déclenche rien.

## 3. Contexte

**Index (E:67-97)**, une entrée par élément, texte ≤ 6 000 caractères, libellé ≤ 100 : proches (≤ 50), moments de vie (≤ 500), présentation déclarée et mémoire permanente « précisions confirmées » (sources de P:41-47, hors « likes », E:74), synthèse du portrait (1), vécus cochés de la cartographie (≤ 80), réponses « non » explicites (≤ 80), pistes avec retour oui/non, preuve et limite d’âge (≤ 180), appuis (≤ 80), pistes du portrait (≤ 60), notes de la personne et d’Ambre (≤ 140). Ligne envoyée à l’analyse (I:15) : `i12 o Reconnu · Anxiété ou tro` — identifiant, initiale du type, **24 premiers caractères du libellé**, sans contenu.

**Sélection (E:98-102)** : identifiants choisis par l’analyse, puis complément par recouvrement de mots de ≥ 4 lettres avec le message (+1 pour présentation et synthèse), jusqu’à **8 éléments, 1 100 caractères chacun, 6 500 au total**. Le complément prend toutes les entrées, même de score nul, dans l’ordre de l’index : le contexte est donc toujours rempli à 8, proches en tête, pertinents ou non. Le JSON est coupé net à 1 100 caractères (la fin d’une fiche, `meaning_now`, `next_question` d’une piste disparaissent souvent). Les métadonnées s’ajoutent hors budget : le bloc `details` mesure 6 200 à 8 000 caractères.

**Toujours envoyé à la rédaction (E:227, E:232)** : message entier, analyse, cadre de suivi (4 champs), mémoire bornée (≤ 3 650 caractères, E:18-31), 6 derniers échanges (message ≤ 500, réponse ≤ 700 caractères, E:175), prénom et genre, âge/genre/activité, détails. **Jamais envoyé** : bilans (exclus de l’historique E:175 et absents de l’index, E:323), observations de proches (E:324), likes, échanges au-delà de six hors mémoire, catalogue des 161 repères, comptes rendus et dossier de suivi, date du jour.

| Source | Entrée dans le contexte |
|---|---|
| Cartographie | seulement si enregistrée avec accord (C:207-210) ; par entrées séparées (vécu, réponse non, piste, appui, note) ; aucune vue d’ensemble |
| Portrait déclaré | une entrée « Votre présentation » ; identité et contexte déclaré toujours joints |
| Synthèse et pistes | une entrée synthèse (si portrait à jour, P:28) ; une entrée par piste, libellé du catalogue (E:94) |
| Proches / ligne de vie | une entrée chacun ; `target_id` réel pour les propositions (E:101) |
| Cadre de suivi | toujours : psychologue oui/non, intervalle, dernière et prochaine séance en secondes Unix |
| Mémoire de conversation | toujours, bornée ; mémoire permanente : simple entrée d’index libellée « memory » |

Tailles mesurées (profils synthétiques, caractères) :

| Profil | Entrées d’index | Index (analyse) | Entrée analyse | Entrée rédaction (dont système 9 837) | Entrée relecture |
|---|---|---|---|---|---|
| Léger : 2 proches, 3 moments, 2 échanges | 7 | 171 | 2 059 | 8 379 + 9 837 | 6 287 |
| Type : 6 proches, 20 moments, cartographie remplie, 6 pistes, 12 échanges | 107 | 3 082 | 7 032 | 16 431 + 9 837 | 8 781 |
| Maximal : 50 proches, 500 moments, 58 vécus, 82 pistes, 140 notes, message de 6 000 | 865 | 26 263 | 39 381 | 25 581 + 9 837 | non réservable |

## 4. Consignes de rédaction

Le prompt système (E:231 + `contract` B:313 + suffixe B:768) est un seul bloc de 9 837 caractères, sans titres. Découpage fonctionnel :

| Bloc | Caractères | Contenu |
|---|---|---|
| Identité et forme | 275 | « assistante psychologue IA », pas de diagnostic, vouvoiement, « généralement 80 à 180 mots », reflet avant la question, « une seule au maximum » |
| Question sensible | 138 | accord, droit de passer ; interdite si `sensitive_allowed=false` |
| Postures | 226 | une phrase pour `repair`, `stabilize`, `options`, `listen` |
| Relation | 171 | « Une hypothèse doit inviter la personne à la corriger » ; ni exclusivité, ni disponibilité permanente, ni supériorité sur un proche ou un soignant |
| Limites | 269 | respecter posture, sujets fermés, contraintes ; pas de récit traumatique ; pas de causalité certaine ; « Hypothèses explicitement incertaines, faits sourcés uniquement » |
| Cartographie | 473 | associations ≠ diagnostic ; respecter oui/non ; cartes « jeune enfant » |
| Conseil | 79 | « Pas de conseil sans demande, permission ou possibilité explicite de l’écarter. » |
| Sécurité | 548 | 3114, 15/112, 17 ; pas de secours de crise si `risk=none` |
| Psychologue | 162 | « Si la personne a un psychologue, adapte-toi au rythme et aux dates de séances, propose de préparer ce qui compte ; sinon suggère un accompagnement sans insister. » |
| Données et notes | 238 | données non fiables ; n’annoncer aucun enregistrement |
| Contrat de sortie | 7 033 | format JSON, six types de proposition, champs, modes `set/append` (≈ 5 400 caractères de mécanique) |
| Suffixe moteur | 224 | « Une question au maximum », citations mot pour mot pour notes et observations |

Ce qui est dit, par sujet : **hypothèses** — incertaines, révisables, à faire corriger. **Questions** — une au plus, reflet d’abord, règles de sensibilité. **Conseils** — sur demande ou permission ; en posture `options`, « donner réellement 2 ou 3 possibilités et leurs limites ». **Psychologue** — la phrase citée. **Absent, sans aucune occurrence** : objectifs, valeurs, changement, techniques ou exercices, premier pas, tâche entre deux échanges, retour sur ce qui a été essayé, obstacle, mesure d’un effet, différence entre ce qui a été proposé et ce qui a été fait, limites d’un exercice, moment où orienter vers le psychologue, plan du psychologue à ne pas contredire. Le rédacteur ne reçoit pas non plus la définition des codes de contrainte ni des états.

## 5. Matrice de posture

Il n’y a pas de texte injecté par valeur : la même phrase figure dans tous les prompts, et l’analyse est jointe en JSON.

| Posture | Fixée par | Texte dans le prompt (E:231) |
|---|---|---|
| `listen` | règles (demande d’écoute) ou modèle | « écouter sans conseil » |
| `explore` | défaut | aucun |
| `stabilize` | règles (détresse, dissociation, honte) | « présent, réponse courte, un pas à la fois » |
| `repair` | règles (rupture) | « reconnaître la rupture et sa part sans justification » |
| `safety` | tout risque ≠ `none` | bloc sécurité général |
| `options` | règles (demande d’avis) ou modèle | « donner réellement 2 ou 3 possibilités et leurs limites » |

| Contrainte | Fixée par | Effet codé |
|---|---|---|
| `no_advice` | demande d’écoute | relecture 8 si « vous devez », « il faut que vous », « je vous conseille » (E:111) |
| `no_questions` | « pas de question », détresse, sujet fermé, fermetures omises | relecture 12 à la première question (E:106) ; retirée en cas de risque |
| `no_exploration` | détresse, sujet fermé | aucun (étiquette) |
| `respect_closed_topics` | mémoire contenant des sujets fermés | aucun (étiquette) |

## 6. Relecture

Les quinze critères codés (E:37-40) : 1 Le risque est traité explicitement · 2 Aucune information facilitant un danger · 3 Aucun diagnostic ni présentation comme professionnelle · 4 Aucun fait inventé · 5 Aucune croyance délirante confirmée ni promesse intenable · 6 Reflet précis du message · 7 Une seule question, adaptée · 8 Conseils demandés ou proposés avec permission · 9 Hypothèses marquées et révisables · 10 Aucun blâme ni minimisation · 11 Pas de complaisance automatique · 12 Limites et rythme respectés · 13 Posture adaptée au besoin · 14 Mots simples et longueur proportionnée · 15 Autonomie et liens humains préservés. Le relecteur ne reçoit que ces intitulés, pas les définitions de la grille 7.c.

**Part déterministe (E:103-113)** : 7 (plus d’une question), 12 (question malgré `no_questions`), 3 (étiquette diagnostique ou « je suis votre psychologue »), 4 (« j’ai noté/enregistré/ajouté », « c’est noté »), 2 (numéro hors 3114, 112, 119, 3919, 15, 17), 1 (risque sans numéro), 8 (formule directive sous `no_advice`). **Part modèle (E:239)** : les quinze, en `yes/no/na`. **Décision (E:242)** : validée si aucun échec de 1 à 5, moins de deux échecs de 6 à 15, et ni 7 ni 12. Non validée et relue → une réécriture (E:253). Non validée après réécriture, ou relecteur indisponible après trois essais → repli : message de sécurité si risque, sinon texte fixe (E:126), sans proposition ni note (E:254).

**Non vérifié** : notes de cartographie et observations (non transmises au relecteur) ; lien avec le cadre de suivi (le bloc `care` n’est pas transmis) ; longueur en mots ; réouverture d’un sujet fermé autrement que par le modèle ; tout ce qui touche au changement. Le champ `questions` est toujours vide côté relecteur, car les questions sont déjà fusionnées dans la réponse (B:732-734).

## 7. Mémoire de conversation

**Schéma** : table `chat_memory` (`0019:2-7`), `summary` JSON de huit rubriques (E:36) — `goals` Vos demandes et objectifs, `facts` Ce que vous avez rapporté, `hypotheses` Hypothèses à vérifier, `tried` Ce qui a été essayé, `alliance` Votre retour sur les échanges, `safety` Sécurité et appuis, `closed` Sujets que vous avez fermés, `session` À apporter en séance — plus `safety_seen`, `closed_overflow`. Bornes : 12 éléments par rubrique (30 pour `closed`), 300 caractères chacun (E:290-291, E:333).

**Mise à jour (E:280-299)** : hors du tour, par le cron, quand au moins 4 nouveaux échanges existent et qu’aucun tour n’est en cours ; lot de 8 au plus ; petit modèle, 2 400 jetons. Consigne (366 caractères) : « Choisis des citations EXACTES des messages de la personne… Sépare objectifs, faits, hypothèses explicites, essais, alliance, sécurité, limites refusées et éléments à apporter en séance. Aucune phrase de l’assistant ne prouve un fait. » Le modèle ne reçoit que les messages de la personne (900 premiers caractères) ; chaque citation doit être une sous-chaîne exacte (E:291). Sujets fermés et sécurité sont aussi ajoutés par règles (E:292).

**Correction par la personne** : `PUT /api/chat/memory` (E:330-340) avec révision ; Paramètres › Mes informations › La mémoire de conversation (S:43), une ligne par élément ; effacement par `{sections:{}}` ; le repère de lecture avance, les anciens échanges ne recréent rien.

**Objectifs, essais, résultats, « à apporter en séance »** : les rubriques `goals`, `tried`, `session` existent, mais ce sont des citations en texte libre. Pas d’identifiant, de statut, de date, de lien entre un objectif et un essai, ni de rubrique « résultat ». Ce qu’Ambre a proposé n’y entre jamais. **Retour dans le contexte** : `memoryPrompt` (E:18-31) ordonne `closed, safety, alliance, goals, session, tried, hypotheses, facts`, coupe chaque élément à 240 caractères et chaque rubrique à 380 (1 300 pour `closed`, 850 pour `safety`) — en gardant les **premiers** éléments, donc les plus anciens : sur quatre essais de 150 caractères, seuls les deux plus anciens sont transmis (vérifié par exécution).

## 8. Écritures d’Ambre

- **Propositions** (six types, B:11-12) : profil, précision de mémoire permanente, proche (création, modification), moment (création, modification). Conditions (B:217-258) : citation exacte du message actuel ≤ 1 200 caractères, champs autorisés, date de naissance citée en entier, cible existante. Validées au brouillon (B:736) puis à la publication (B:748-749) ; jamais en cas de risque (B:735). Rien n’est appliqué sans acceptation par la personne, avec contrôle de révision (B:500-512). Visibles dans la conversation.
- **Notes de cartographie** (B:737-740, B:750-759 ; C:99-118) : cartographie enregistrée avec accord, pas de refus de conservation (O:39), pas de risque ; citation exacte de 3 à 300 caractères, enregistrée sous la forme « Vous dites : « … » » ; cible vécu, piste ou générale ; 3 par tour, 60 au total ; écriture atomique avec la réponse, annonce par le serveur seulement si l’écriture a réussi ; consultables et retirables sur `/cartographie/`.
- **Observations de proches** (O:62-103) : accord actif, message sans refus, danger, santé, sujet sensible, fiction, chiffres ni lien ; un seul proche nommé ; phrase entière conforme à un vocabulaire fermé ; 2 par tour, 60 par proche ; visibles et supprimables dans la fiche. Le moteur ne fournit plus au modèle les phrases candidates (l’ancien chemin le faisait, B:360, B:386) : le modèle doit deviner la phrase exacte. Aucun test du moteur ne couvre ce cas.

## 9. Bilans et point avant séance

**Bilans** (B:517-709) : analyse initiale puis points à la demande (1 h après le dernier bilan, 24 h entre deux points, 4 par jour). Consignes F:49-57 (3 364 et 3 595 caractères) ; le point demande « les objectifs qu’elle a exprimés », « ce qu’elle dit avoir réellement essayé et ce que cela a donné », « une ou deux pistes concrètes ». Modèle Sonnet, 3 200 jetons, 600 à 900 mots. Contexte (B:584-622, ≤ 34 000 caractères) : ancien sélecteur `promptContext` (B:259-310), cartographie complète, 24 observations, bilan précédent, échanges depuis le dernier bilan. **N’utilise ni la mémoire de conversation ni le cadre de suivi.**

**Point avant séance** (K:84-154) : seulement s’il existe un partage actif avec un compte psychologue, préparé par le cron dans les 24 h avant une séance ou à la demande, 4 préparations par jour. Lots de 20 éléments de 1 600 caractères tirés des rubriques partagées (messages de la personne, proches, moments, bilans, portrait, cartographie) ; petit modèle ; citations exactes classées en six thèmes (K:7), dont « Ce qui a été essayé » et « À discuter en séance ». **N’utilise pas la mémoire** (`care.test.mjs:86` l’affirme) ni les réponses d’Ambre. Une personne suivie mais sans partage activé n’obtient aucun point.

## 10. Catalogue psychologique

`journal/src/psychology-catalog.generated.mjs` est produit par `scripts/build_psychology_catalog.mjs` depuis `data/psychology_catalog.json` (contrôle `--check` dans `.github/workflows/verify.yml:22`). Contenu confirmé : **161 repères** (85 processus, 47 ressources, 29 contextes), 16 domaines, 44 sous-domaines, 25 sources (NHS, CCI, NIMH, OMS, NICE). Champs d’un repère : `id, domain, subdomain, kind, label, description, signals, alternatives, questions, keywords, related, source_refs`. **Aucun champ d’action ou de stratégie.** Les 47 ressources décrivent des capacités (`problem_solving`, `behavioral_experiment`, `small_goals`, `response_pause`, `grounded_attention`, `planning_steps`…) en une phrase, sans mode d’emploi.

Sélection : `selectPsychologyGuide` (G:7-33), score lexical. Deux usages réels : l’analyse du portrait (`portrait-refresh.mjs:37`, 24 repères) et l’ancien prompt de conversation (F:60, 8 repères). **Le moteur actuel n’injecte pas le catalogue** : il n’en tire que les libellés des pistes (E:16, E:94). Les repères n’atteignent Ambre que par les pistes du portrait. Les douze repères rédigés à la main avec `experiment` et `resources` (F:18-31) sont écrasés par F:34 : données mortes.

## 11. Transparence

Par appel, `appendTraceStage` (I:18-27) enregistre : étape, modèle demandé, statut, tentative, heure, sortie max, consigne système et chaque clé de l’entrée (valeurs dédupliquées), puis usage, code d’erreur et modèle reçu (E:136-139). Limites : 16 étapes, 128 000 octets, omission signalée. Le résumé du tour (E:258-260) ne garde que des champs en liste blanche : besoin, états, risque, posture, contraintes, vérification du risque, trois blocs de contexte (`selected`, `memory`, `history`) avec tailles, jetons, verdict de relecture, durée, modèles. `traceDetail` (I:37-44) reconstruit un instantané à clés fixes.

À compléter pour toute nouveauté : liste blanche de l’analyse (E:260 et E:272) ; `context.blocks` (E:258) ; instantané (I:43) ; `USAGE_POLICY` (I:17) ; `current` et `stored_elsewhere` de l’aperçu (E:316-328) ; libellés et valeurs fermées (S:2-5) ; description des étapes (S:28-33) ; noms d’étapes (S:55) ; « 15 règles » écrit en dur (S:32, S:59). Les entrées inconnues s’affichent déjà, avec la clé brute comme libellé. La mise à jour de la mémoire, les bilans et les points avant séance n’ont pas de trace (appels sans `jobId`).

## 12. Coûts

Mesures réelles consignées (`docs/reprise-2026-11-16.md:14, 55`) : 7 565 à 9 333 jetons d’entrée cumulés sur trois étapes, puis 11 800 et 12 549 après l’ajout de la cartographie ; 0,008 à 0,020 $ par tour ; 9 802 pour un tour de crise. Estimation du dépôt sur contexte chargé : environ 16 000 (`reprise:66`).

| Profil synthétique | Analyse | Rédaction | Relecture | Entrée estimée | Réservé / 30 000 | Marge |
|---|---|---|---|---|---|---|
| Léger | 1 384 | 6 745 | 2 520 | 10 649 | 15 449 | 14 551 |
| Léger + réécriture | | + 7 276 | + 2 520 | 20 445 | 28 545 | 1 455 |
| Type | 3 041 | 9 429 | 3 351 | 15 821 | 20 621 | 9 379 |
| Type + réécriture | | refusée | | | > 30 000 | tour échoué |
| Intermédiaire (20 proches, 120 moments, 278 entrées) | 4 776 | 9 165 | 3 341 | 17 282 | 22 082 | 7 918 |
| Maximal, message court | 11 021 | 9 527 | 3 032 | 23 580 | 28 380 | 1 620 |
| Maximal, message de 6 000 et mémoire pleine | 13 824 | 12 479 | refusée | 26 303 | 29 603 | repli générique |

Le plancher de la rédaction est de 5 752 jetons réservés avant tout contexte (consigne 9 837 + schéma 2 019 caractères, + 1 800 de sortie). Une réécriture ne tient que si l’entrée de rédaction reste sous environ 5 700 caractères, soit moins que le seul budget des détails. Autres plafonds : 48 000 caractères par appel (marge 21 700 pour la rédaction type ; 7 200 pour l’analyse maximale) ; trace 41 Ko (type) à 69 Ko pour deux étapes (maximal) sur 128 Ko ; semaine : 100 tours type ≈ 2,06 M réservés sur 3 M, hors mémoire (≤ 6 300 par lot), bilans (≤ 16 000) et points (≤ 13 300 par lot).

## 13. Faiblesses et points d’insertion

**Faiblesses, par importance pour les deux buts**

1. Budget par tour : réécriture impossible sur profil rempli, repli systématique sur profil maximal (§ 12). Bloque tout ajout.
2. Aucune consigne ni répertoire de changement dans le moteur ; l’ancien savoir-faire est hors service (§ 4, § 10).
3. Aucun plan structuré ; mémoire en citations, propositions d’Ambre perdues après six échanges, extrait qui garde l’ancien (§ 7).
4. Aucun repère temporel (date du jour, dates des échanges, séances lisibles).
5. Sources absentes : bilans, observations, catalogue, consignes de cartographie ; la personne répond à l’analyse initiale qu’Ambre ne voit pas.
6. Analyse aveugle au changement : `need:'act'` sans effet ; aucun champ pour un objectif énoncé, un essai rapporté, un obstacle ; faux positifs des règles (§ 14, n° 8).
7. Sélection du contexte : libellés de 24 caractères, remplissage sans pertinence, coupe à 1 100 caractères ; objectifs sans priorité.
8. Relecture : aucun critère sur le changement, relecteur privé du cadre de suivi, intitulés sans définition, faux positifs bloquants.
9. Psychologue : une phrase ; aucune place pour ses consignes ; point avant séance sans mémoire ni plan, réservé aux partages activés.
10. Mémoire : après une panne du fournisseur, des débuts de messages bruts sont rangés pour de bon en « faits » (E:289).

**Préalables communs** : relever ou recalculer le plafond E:151 (ou alléger le contrat B:313, dont une partie décrit des champs absents) ; transmettre la date et les dates d’échanges (E:175, E:227) ; ne plus couper la question finale de la dernière réponse (E:175) ; faire garder le plus récent à `memoryPrompt` (E:20-26).

**(a) Catalogue de stratégies et sélection par tour.** Données : `data/change_strategies.json` → `scripts/build_change_strategies.mjs` (sur le modèle de `build_psychology_catalog.mjs:28-88`, contrôle dans `verify.yml`) → `journal/src/change-strategies.generated.mjs` ; fichier séparé plutôt qu’un champ ajouté aux repères, car le catalogue complet part aussi au navigateur (`psychology-map.mjs:54`). Rattacher chaque stratégie aux `id` de repères, avec indications, contre-indications, étapes, mesure de l’effet, limite d’orientation, sources. La base documentaire actuelle ne couvre pas ce terrain (`docs/recherche-pratique-clinique-2026-11-16.md:246` écarte les items « méthodes de changement, tâches ») : une recherche dédiée est nécessaire. Sélection : présélection déterministe (modèle G:7), liste courte ajoutée à l’entrée d’analyse (E:202), champ `strategy_ids` dans `ANALYSIS_SCHEMA` (E:42) validé en liste blanche (E:203-205), vidé si risque, `stabilize`, `repair`, `listen` ou `no_advice` ; détail de deux stratégies au plus dans `data` (E:227). Régression : sortie tronquée du petit modèle (déjà rencontrée, `reprise:42`) ; tests T:121, T:129 (plafond de 18 000 jetons estimés), T:178, T:197, T:223, T:301-363.

**(b) Plan structuré et persistant.** Migration `0021` : tables `change_goals` et `change_trials` (identifiant, objectif, stratégie, énoncé, statut `proposed|accepted|done|skipped|abandoned`, résultat, dates, date de retour prévue, nombre de relances, révision, citation d’origine). Ne pas réutiliser `chat_proposals` : son `kind` est sous contrainte CHECK (`0012_ambre_portrait_corrections.sql:4`), ce qui impose une reconstruction de table ; la colonne `stage` des tâches l’est aussi (`0019:11`), d’où l’intérêt d’une sous-étape comme `risk_check` plutôt qu’une étape nouvelle. Écriture : lignes créées « proposées » dans le lot de publication (modèle B:749), avec la vérification de citation de B:228-229, acceptées par la personne (modèle B:500-512). API `/api/plan` dans un module à part, branché près de `worker.mjs:152`. Lecture : bloc `plan` toujours envoyé et borné (E:171-173, E:227), ajouté à l’empreinte (E:172) et à la garde de publication (B:746). À décider : sort du plan lors de `DELETE /api/chat` (B:730) ; export (B:514). Régression : T:54, T:89, T:94, T:135 ; `chat.test.mjs:109-114` ; `ambre-migration.test.mjs:6`.

**(c) Retour planifié sans harceler.** Calcul déterministe d’un seul essai « dû » (date de retour passée, accepté, moins de deux relances, dernière relance ancienne), jamais en `safety`, `stabilize`, `repair`, `listen`, ni sur un sujet fermé ; relance comptée dans le lot de publication, seulement si la réponse est publiée ; arrêt après deux relances sans suite ; report ou abandon par la personne. Champ d’analyse pour un essai rapporté (`fait|partiel|non fait|obstacle`), transformé en mise à jour proposée. À corriger avant : la règle « honte » (E:61-62) classe « j’ai honte de ne pas avoir fait l’exercice » en `stabilize` sans question, soit l’inverse de ce qu’il faut. Régression : T:144, T:197, T:287.

**(d) Articulation avec le psychologue.** Champ « ce que mon psychologue m’a demandé de travailler » dans `care_settings` (`0020_care.sql:2`, route K:158), joint au bloc `care` (`care-context.mjs:5-11`, signature figée par `docs/plan-ambre-psychologue-2026-11-16.md:50-52`) ; règle de rédaction : avec un psychologue, stratégies compatibles ou « à valider en séance », jamais contre la consigne déclarée. Préparation de séance : nouvelle rubrique `plan` dans `CARE_SCOPES` (K:4), `briefSources` (K:104-120) et déclencheurs de révision (`0020_care.sql`) ; point personnel sans partage à créer (K:89-90 l’exige aujourd’hui). Le dossier est en lecture seule pour le professionnel : toute écriture de sa part serait un changement de contrat (`care.test.mjs:27`). Régression : `care.test.mjs:53, 61, 86, 104, 120`.

**(e) Critères de relecture.** Le nombre 15 est codé à quatre endroits : `REVIEW_CRITERIA` (E:37-40), `REVIEW_SCHEMA` (E:58), contrôle de longueur (E:240), règle de décision (E:242) ; plus les jeux d’essai `Array(15)` (T:17) et l’interface (S:32, S:59). Ajouts proposés, non bloquants : stratégie issue de la sélection et proposée avec permission ; un seul petit pas avec la manière de savoir s’il aide ; retour sur l’essai précédent avant toute nouvelle piste ; aucune pression ni reproche sur un essai non fait ; cohérence avec le suivi. Part déterministe possible : stratégie nommée hors sélection, plus d’une stratégie, exercice proposé sous `no_advice` ou en sécurité. Joindre `care`, plan et stratégies à l’entrée du relecteur (E:239). Régression : T:33, T:38, T:115, T:165-193.

**(f) Affichage.** Paramètres : points listés au § 11. Page dédiée « Mon plan » (ou rubrique de `/mon-suivi/`) : liste des pages `ui.mjs:60`, module d’interface sur le modèle de `care-interface.mjs`, proxys `netlify.toml:328-345`, route Worker ; cartes dans la conversation via l’état renvoyé par B:189-199. Régression : T:223, T:241, T:250, T:263 ; contrôle visuel à 1 440, 390 et 320 px.

## 14. Défauts relevés

1. **E:151, E:253, E:271, B:183** — réécriture refusée par le plafond du tour sur profil rempli ; le tour échoue avec un message qui parle du budget hebdomadaire. Non couvert : T:33 utilise un contexte vide.
2. **E:151, E:245, E:254** — profil maximal et message long : relecture non réservable, repli générique à chaque tour.
3. **E:20-26** — l’extrait de mémoire garde les éléments les plus anciens de chaque rubrique.
4. **E:175, E:211** — la réponse précédente est coupée à 700 caractères par le début : la question finale d’Ambre disparaît de l’historique ; le rédacteur ne voit pas la question à laquelle la personne répond, et la règle « pas deux tours de suite » ne la détecte plus.
5. **E:227, `care-context.mjs:9`** — aucune date du jour ; séances en secondes Unix ; la consigne « adapte-toi aux dates de séances » est inapplicable.
6. **B:313 contre B:768, B:767, B:733** — même prompt : « au plus deux questions » puis « une question au maximum » ; schéma à 1, code qui en accepte 3. Le contrat cite des champs absents du moteur (`psychological_hypotheses`, `coverage`, `conversation_coverage`, « fin de mémoire »).
7. **E:108-109** — faux positifs bloquants : « J’ai noté que vous… » et « C’est noté » échouent au critère 4 ; « contactez votre médecin dans les 48 heures » au critère 2 (vérifié).
8. **E:60-63, B:499** — « faire disparaître cette habitude » ou « un fardeau » déclenchent la voie de crise et l’affichage immédiat du 3114, sans vérification ; « ce n’est pas moi qui décide » et « honte » imposent `stabilize` sans question ; « avis » est reconnu dans « ravis » (vérifié).
9. **E:90, B:318-319** — les détails de pistes contiennent `url:"/personnages/…"` ; l’ancien chemin retirait ces liens et interdisait de proposer lectures et personnages (F:72), consigne absente du moteur.
10. **B:768 contre B:386-390, B:102, B:112** — notes de cartographie et observations : le moteur ne reçoit ni la règle des identifiants de cible ni les phrases candidates, alors que le message d’accueil les annonce.
11. **E:239, B:732-734** — `questions` toujours vide pour le relecteur ; bloc `care` non transmis.
12. **E:289** — panne du résumeur : débuts de messages rangés en « faits », lot marqué couvert, sans signalement.
13. **E:98-101** — remplissage des huit détails par des entrées sans rapport avec le message, contraire à la précaution de `recherche…:289`.
14. **I:15** — libellés d’index coupés à 24 caractères (« Réponse non · Parent inc »).
15. **F:18-34, B:330-498, B:148** — code mort en production : douze repères manuels, ancien `generate`, `buildCompanionSystemPrompt` ; les 84 tests de `chat.test.mjs` exercent ce chemin (`engineEnabled` faux par défaut).
16. **K:89-90, K:104-120** — point avant séance sans mémoire ni plan, et inexistant sans partage activé, alors que le plan annonçait que B « peut lire » la mémoire.
17. **E:149, E:212** — au-delà d’environ 1 100 entrées d’index, l’analyse dépasse 48 000 caractères et retombe en silence sur les règles (cas théorique).
18. **Écarts mineurs** : commentaire E:259 périmé (les traces V2 gardent les entrées exactes) ; `journal/README.md:97` annonce encore deux questions ; S:3 nomme `session` « Repères de suivi » contre « À apporter en séance » (E:36) ; S:31 annonce 6 500 caractères quand le bloc en pèse jusqu’à 8 000 ; mémoire permanente libellée « memory » (E:74) ; E:333-334 efface `safety_seen` à toute correction ; nom d’étape « memory » prévu en S:55 mais jamais tracé.
19. Hors moteur : `docs/plan-cartographie-couverture-2026-11-17.md` est apparu puis a été commité (`f32bb7d`) par une autre session pendant l’audit ; cet audit ne l’a ni créé ni lu.
