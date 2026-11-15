# Plan de la page « Cartographie » — 15 novembre 2026

## Objectif

Une page publique `/cartographie/` où une personne coche ce qu'elle a vécu ou ressenti dans son enfance et son adolescence (événements, climat familial, ressentis). La page affiche les **conséquences psychologiques associées dans la recherche** : troubles de la CIM-11, schémas précoces, styles d'attachement, tendances et rôles (par exemple le « rôle de sauveur »). Quand plusieurs éléments cochés convergent vers une même conséquence, celle-ci remonte et la convergence est expliquée. Chaque résultat renvoie vers l'article ou le personnage du site qui l'illustre.

## Ce que la recherche permet d'affirmer, et ce qu'elle ne permet pas

- Les expériences adverses de l'enfance augmentent le **risque** de nombreuses difficultés, avec un effet cumulatif (relation dose-effet : plus il y a d'expériences adverses, plus le risque moyen augmente ; Felitti 1998, Hughes 2017).
- Ces liens sont **statistiques et non spécifiques** : une même expérience est associée à plusieurs troubles (multifinalité) et un même trouble a plusieurs chemins (équifinalité ; Cicchetti et Rogosch 1996). Beaucoup de personnes exposées ne développent aucun trouble ; la génétique, le tempérament, les ressources, les relations ultérieures et les soins comptent.
- La page ne pose donc **jamais de diagnostic** et n'affiche aucun pourcentage individuel. Elle classe des **pistes de compréhension** selon le nombre d'éléments convergents et la solidité des preuves, et rappelle qu'une évaluation se fait avec un professionnel.
- Certains troubles **ne sont pas causés par l'éducation** (autisme, TDAH, troubles du développement intellectuel, la plupart des troubles neurodéveloppementaux) : la page le dit explicitement et ne les relie pas à des facteurs éducatifs.
- Les schémas précoces de Young, l'analyse transactionnelle ou la parentification sont des **modèles cliniques** : leur niveau de preuve est signalé comme tel.

## Architecture

| Élément | Fichier | Rôle |
| --- | --- | --- |
| Référentiel des facteurs | `editorial/cartographie/facteurs.json` | Liste fermée des cases à cocher, groupées, avec description et exemples |
| Fragments de recherche | `editorial/cartographie/liens-*.json` | Liens facteur → conséquence, force des preuves, résumé, sources ; rédigés par des agents de recherche distincts |
| Cibles | `editorial/cartographie/cibles-*.json` | Troubles ayant un parcours publié, repères psychologiques, articles d'origine |
| Assemblage et contrôle | `scripts/build_cartographie.py` (`--check`) | Fusionne les fragments en `data/cartographie.json`, vérifie identifiants, sources, liens vers des pages existantes, absence de liens interdits |
| Page | `content/cartographie.md`, `layouts/cartographie/single.html`, `static/js/cartographie.js`, `static/css/cartographie.css` | Formulaire à cocher et résultats calculés entièrement dans le navigateur |
| Tests | `tests/cartographie.test.cjs` | Calcul des convergences, rendu sans facteur, interdits |

### Modèle de données

- `factors` : `id`, `group`, `label`, `description`, `examples`, `ace` (expérience adverse de l'enfance au sens des études ACE).
- `outcomes` : `id`, `kind` (`trouble`, `schema`, `attachement`, `tendance`), `label`, `code` CIM-11 éventuel, `description` courte, `url` vers le site.
- `links` : `factor`, `outcome`, `strength` (`forte` méta-analyses concordantes ou critère de classification ; `moderee` études longitudinales ou méta-analyse avec effet modeste ; `limitee` données transversales ou rares ; `theorique` modèle clinique), `summary`, `sources`.
- `sources` : `id`, `citation`, `url`, `type`.

### Calcul

Pour chaque conséquence : liste des facteurs cochés qui y mènent, score = somme des poids (forte 3, modérée 2, limitée 1, théorique 1). Affichage en trois paliers : **plusieurs éléments convergent** (au moins deux facteurs, dont un lien au moins modéré), **piste à explorer** (un facteur fort ou modéré), **lien théorique ou limité**. Un compteur indique le nombre d'expériences adverses de type ACE cochées et explique l'effet cumulatif à partir de quatre, sans en tirer de risque individuel.

### Confidentialité et sécurité

Aucune donnée n'est envoyée ni enregistrée : tout se calcule localement et disparaît à la fermeture. En tête de page : ce n'est pas un test diagnostique. En pied : 3114 (prévention du suicide), 119 (enfance en danger), 15 ou 112 en cas d'urgence, 3919 (violences conjugales). Section « ce qui protège » : des facteurs de protection documentés, sans calcul.

## Étapes

1. Référentiel des facteurs et schéma (pilote).
2. Trois agents de recherche distincts, chacun sur un groupe de facteurs, lisant les sources primaires (CDDR de l'OMS, HAS, NICE, méta-analyses) : maltraitance et adversité familiale ; climat éducatif et ressentis ; contexte social, pertes, protections et limites.
3. Assemblage, contrôle automatique, page et tests (pilote).
4. Relecture indépendante des liens et de la page par un agent distinct (exactitude des sources, force annoncée, formulations non déterministes, sécurité).
5. Déploiement, vérification en ligne. Plus tard : branchement à Ambre une fois la cartographie validée.
