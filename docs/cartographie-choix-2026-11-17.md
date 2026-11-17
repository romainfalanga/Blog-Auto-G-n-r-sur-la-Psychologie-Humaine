# Cartographie — choix explicites et séparation du contrôle parental

État : PR 92 et correctif PR 93 fusionnés, Worker et site publiés et vérifiés. La dernière section indique la recette réelle finale ; les paragraphes intermédiaires sont historiques. Travail isolé sur `codex/cartographie-choix`, base `origin/main` f9aa647. Décisions du propriétaire : ajouter l’incertitude, réserver Non/Oui aux événements uniques, distinguer règles strictes et contrôle psychologique. Ne pas rétablir les ressources, ne pas ajouter la question sur les violences amoureuses adolescentes, ne pas confirmer le lien négligence affective–bipolaires, ne pas activer de plafond de dépense.

## Contrat de données et historique

- `answers[id] = "unknown"` signifie « Je ne sais pas ». `0` reste Non, `1` Oui, `2` Souvent ; l’absence de clé reste sans réponse. L’incertitude est conservée par l’API, l’export, les brouillons et les versions ; elle ne produit aucun lien, point, classement ni décompte ACE. Ambre la reçoit dans `explicit_uncertain`, séparément des réponses négatives et des vécus déclarés.
- `response_mode: "binary"` retire Souvent des nouveaux choix pour `adoption`, `divorce`, `deces_proche`, `parent_incarcere`, `migration`, `traumatisme_ponctuel`. Ces libellés portent sur une survenue. Les situations durables ou explicitement répétées gardent Souvent. Une ancienne valeur `2` reste `2`, avec la mention « Ancienne réponse : Souvent ». Aucun remplacement automatique par Oui ; une modification suppose un nouveau choix de la personne.
- `controle` conserve son identifiant, son libellé, sa description, ses exemples et ses liens. Il devient `historical: true`, avec `replaced_by: ["parents_stricts", "controle_psychologique"]`. L’ancienne ligne n’est affichée qu’en présence d’une réponse, d’une précision ou d’une note d’Ambre, avec une invitation à répondre séparément aux deux nouvelles questions. Elle reste lisible et effaçable ; les nouvelles réponses ne la suppriment pas.
- Les deux nouveaux identifiants commencent sans réponse et sans lien. Ni une ancienne réponse Oui/Souvent, ni une note libre, ni un Non ne sont copiés ou répartis entre eux. Le sens des 9 liens historiques n’est pas réattribué. Ambre reçoit les réponses aux deux nouveaux champs comme contexte déclaré, avec l’absence d’association documentée.
- Le plafond de 14 questions par thème s’applique aux questions actives : climat = 14, plus éventuellement la ligne historique distincte ; 60 questions actives et un identifiant historique conservé.

## Sources effectivement ouvertes le 17 novembre 2026

Le site PubMed a renvoyé une page de vérification sans article ; les résumés ont donc été lus via l’API publique Europe PMC, source MED :

- [Salaam & Kyere 2025, PMID 40244958](https://europepmc.org/article/MED/40244958), DOI 10.1037/fam0001315. Méta-analyse de 37 études, 48 échantillons, 27 150 jeunes, 18 pays/groupes : contrôle **psychologique** associé aux symptômes internalisés, effet modulé par les répondants et l’âge. Ce résultat agrégé ne suffit pas à recopier tous les liens anxiété/dépression de l’ancien item.
- [Pinquart 2017, PMID 28459276](https://europepmc.org/article/MED/28459276), DOI 10.1037/dev0000295. Méta-analyse de 1 435 études : contrôle comportemental et contrôle psychologique sont distingués ; associations de directions différentes avec les symptômes externalisés, certaines bidirectionnelles. Une simple règle stricte ne doit donc pas hériter des associations du contrôle psychologique ou sévère.

Aucun lien nouveau ou réaffecté dans ce lot. Les fragments et forces existants ne changent pas. Les deux nouvelles questions servent à préciser le contexte avec les mots de la personne ; leurs rapprochements scientifiques restent à instruire séparément, source par source. Les copies de lecture sont des intermédiaires dans `work/`.

## Vérifications prévues

Validation stricte et persistance de l’incertitude ; absence d’effet sur le calcul ; parité page/Worker ; coexistence de l’incertitude avec les alternatives exclusives ; anciens `2` et `controle` conservés par sauvegarde, rechargement et export ; aucune réponse nouvelle déduite ; édition des notes et conflits conservés ; interface à 390 et 1440 px, clavier/focus et aucun débordement. L’audit des correspondances porte une attestation de relecture du graphe : son empreinte finale sera actualisée après la relecture indépendante, sans déclarer prématurément cette étape réussie.

## Résultats de l’implémentation

- 61 identifiants de facteurs : 60 questions actives, 1 question historique. 113 sujets, 518 liens et 272 sources conservés ; aucun lien ajouté, transféré, retiré ou repondéré. Le calcul des résultats garde sa parité et l’ancienne valeur Souvent garde son poids historique ; aucun ancien 2 n’est converti en 1.
- `node --test journal/tests/cartographie*.test.mjs journal/tests/v5-A5-budget.test.mjs` : 56 réussites. La suite inclut validation, 300 profils mixtes en parité, API réelle/SQLite, export, conflits, contexte d’Ambre, borne de contexte existante et budgets de référence.
- `node --test journal/tests/v5-C5a-cartographie.test.mjs journal/tests/v4-C-cartographie.test.mjs journal/tests/v5-A5-budget.test.mjs tests/cartographie.test.cjs` : 47 réussites (certains tests de budget sont communs à la commande précédente).
- Nouveau `tests/cartographie-choices-browser.cjs` : 7 parcours réussis, dont interface à 390/1440 px, localStorage, conservation des anciens 2 et de `controle`, absence de nouvelle réponse implicite, séparation de l’incertitude des alternatives exclusives, conflit 409 et effacement volontaire de la seule réponse historique. Commande locale : `CHROMIUM_PATH=/usr/bin/chromium node tests/cartographie-choices-browser.cjs`.
- Suite navigateur existante `tests/cartographie-browser.cjs` : 16 scénarios réussis, dont 320/390/1440 px. Aucun débordement ni erreur JavaScript. Captures mobiles contrôlées dans `work/cartographie-choix/`.
- `git diff --check` et `node scripts/build_cartographie_module.mjs --check` : réussis.
- `python scripts/build_cartographie.py --check` : blocage documentaire attendu, « audit des correspondances périmé : les facteurs, sujets, liens ou ressources ont changé ». Aucune autre erreur de génération. L’audit et son attestation de relecture restent inchangés jusqu’à la revue indépendante.

Empreinte du cœur du nouveau graphe à relire : `a2b1572f81f0ac0233e0aa5d3d41664df1b4fb4e3eba92aeb82739b218e61b69`. Après validation, actualiser l’attestation `final_validation` des correspondances selon l’avis du relecteur, régénérer l’audit puis les données/module, et relancer les deux `--check`. Aucune publication ni aucun commit effectué par l’agent.

## Relecture indépendante et intégration

Verdict non bloquant. Le relecteur a exécuté 79 tests et 7 parcours navigateur (390/1440, rechargement, conflits, anciens 2/controle, effacement historique), et comparé le graphe à la base : 518 liens, 272 sources, 113 sujets, protections, classement, exclusivités et correspondances inchangés. Les champs originaux des 59 facteurs restent inchangés ; seules des métadonnées et deux nouveaux identifiants sans liens sont ajoutés. L’empreinte ci-dessus a été reproduite indépendamment.

L’attestation et l’audit sont actualisés uniquement pour ce delta de questionnaire. Il ne s’agit pas d’une nouvelle revue clinique des 518 liens. Les libellés des nouvelles données affichées dans Paramètres sont français.

## Stabilisation du contrôle de reprise

Le test à horloge virtuelle du cron dépendait encore de l’attente réelle du threadpool SHA-256 : sous charge, il comptait parfois une cinquième passe, contre trois isolément. Le hachage est désormais simulé dans ce seul test par le même SHA-256 exact, retourné en Promise. Seuils, latences, empreintes et code de production inchangés. Correction relue indépendamment ; trois tests isolés passent (42 s, 46 s, 4 s simulées).

## Publication et vérification intégrées

PR 92 : source `da9cce8a6d5e5f69ebb192612649c57e4ea94bb2`, fusion `29331d9153072896eead2b0c98cc298cdae1f39e`, arbre `a1e2f108a38cd06e08f36c5f88f69f171b92d7a1`. CI `44079629158` réussie. Suite intégrée : 792 journal, 41 site, 35 parcours navigateur, 43 tests Python et toutes les commandes `verify.yml`. Les vérifications d’audit annoncées comme bloquées dans l’historique ci-dessus passent après l’attestation du delta relu.

Worker complet publié avant fusion : `fad51c8803c94152a88d508dfd8bfe8b`, 17 novembre 2026 à 12:07 UTC ; 44 empreintes identiques, liaisons et cron conservés. Aucune migration supplémentaire. Le premier build Netlify de main a échoué (code 2), malgré la prévisualisation prête ; cause confirmée dans le journal public : erreur 502 Netlify lors de la récupération de ses extensions avant la commande du site. Hugo 0.140.0 extended réussit localement. Ne pas déclarer le nouveau questionnaire en production statique avant vérification du déploiement suivant.

## Refus explicite d’affichage du lien négligence affective–bipolaires

La question posée au propriétaire portait sur **l’affichage** de cette association ; sa réponse était « Non, ne pas confirmer ce lien ». La vérification finale a constaté que le lien était déjà présent dans la version antérieure. Le simple maintien du graphe à 518 liens ne respectait donc pas ce refus. Le correctif utilise le mécanisme existant `remove: true` pour `negligence_emotionnelle` → `icd_6a60`, sans modifier les réponses enregistrées, les notes ni les autres associations.

Relecture indépendante du 17 novembre : une seule paire retirée, 517 autres liens identiques, 272 sources et 113 sujets conservés ; absence de cette paire dans les résultats, preuves, résumé et index transmis à Ambre. Les réponses et notes restent disponibles. 60 tests puis 7 tests finaux réussis ; vérification du module et génération en mémoire sans erreur. La spécificité du seul sujet `icd_6a60` passe automatiquement de 2100 à 2262 avec l’algorithme existant ; aucun autre paramètre de classement modifié.

Nouvelle empreinte du cœur : `d9ab1373af12b3bdc9d3634d64115bc258d430b6f4b0aa7cd9effa4c80292308`. L’attestation ne couvre que ce retrait, sans prétendre à une nouvelle revue clinique globale. Les comptes à 518 liens ci-dessus décrivent l’état avant ce refus appliqué.

## Publication finale après retrait

PR 93 : source finale `4ec5733788e90781503f33cdce750295b01ef5bd`, fusion `e77c7d5df29522ea0011c47f396f74881adbc9ef`, arbre `108b567080b95704578c5f593d75e686ae44a74c`. CI `44081960833` entièrement réussie. Validation finale : 794 tests journal, 41 site, 81 ciblés ; parcours Cartographie 16 + 6 + 7 réussis. Les 6 parcours P2 validés précédemment restent inchangés, soit 35 parcours navigateur pour l’ensemble du lot. Tous les contrôles de `verify.yml` passent, y compris le compteur de couverture régénéré à 517.

Worker complet `5b2c044141344319885d92c7d99fbb92`, publié à 12:17 UTC : 44 empreintes relues après upload sans écart, 7 liaisons et cron conservés. Netlify `6afc468786ed9578fc51b512`, ready, publié à 12:20:58 UTC (13:20 Paris), même commit de fusion. Aucun changement de schéma après 0024.

Recette production terminée le 17 novembre à 12:21:54 UTC : retrait de la paire confirmé dans le dataset public, les résultats et le résumé API avec la réponse conservée ; six questions binaires ; « Je ne sais pas » sauvegardé, rechargé et exporté ; questions distinctes et indépendantes ; six anciens Souvent, réponse historique et notes préservés. Parcours à 1440/390 px, six captures, aucune erreur JavaScript/console ni débordement, aucun appel IA. Compte synthétique supprimé (DELETE 200 puis session vide), credentials effacés. Contrôle D1 des quatre comptes utilisés pour P2 et Cartographie : 43 tables, zéro reliquat, aucune violation de clé étrangère. Les rapports et captures restent dans le répertoire de travail de recette ; aucun identifiant de connexion n’est versionné.
