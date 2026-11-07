# Reprise éditoriale du 7 novembre 2026

Cette étape reprend la demande de trame à partir de l’article sur l’alexithymie en couple, ainsi que la couverture des référentiels. Le point de départ est le commit `a9ade8d0b840d3e9beadfb860aa109edd0fce12c` : 1095 articles, 92 personnages et 59 parcours cliniques.

## Travail réalisé

Les 387 articles des 33 personnages d’origine ont reçu une lecture intégrale par personnage, puis des corrections ciblées. Au total, 371 corps diffèrent du point de départ. Les exemples, objectifs, actions et utilités ont été lus pour leur contenu ; une prose déjà complète n’a pas été remplacée pour obtenir une étiquette. Les conseils « Du côté des proches » ont été comparés aux scènes.

Les corrections comprennent notamment neuf conclusions mal placées ou absentes, une section initialement vide sur la colère, des exercices incomplets, des exemples génériques, des titres collés à leur paragraphe et des affirmations causales ou promesses trop fortes. Les exceptions nécessaires dans une ouverture, un dénouement ou une conclusion ont été consignées avant/après. Les métadonnées et les adresses des anciens articles sont conservées.

Quatre parcours ajoutent 48 épisodes : Malo et la catatonie associée à un autre trouble mental ; Mireille et le trouble neurocognitif léger ; Anatole et la Tourette ; Doriane et la personnalité évitante dans le DSM-5-TR. Chacun comporte un dossier, douze aspects distincts, des scènes suivies, trois manifestations avec deux exemples, trois techniques à trois actions, un dénouement, une conclusion et le point de vue des proches. Les 48 épisodes ont aussi reçu une seconde lecture éditoriale indépendante.

La nouvelle version compte 1143 articles, 96 personnages et 63 parcours cliniques (756 épisodes). Cette relecture éditoriale ne constitue pas une expertise clinique externe.

Les 19 nouvelles liaisons de concepts sont reconnues par les recommandations et la carte de lecture, y compris les provenances combinées. Elles restent des liens entre des textes publics et des concepts ; elles ne deviennent pas des preuves sur un lecteur.

## État et preuves à consulter avant toute suite

- [État de couverture](etat-couverture.md) : chiffres générés et tous les cas cliniques encore candidats.
- [Registre de couverture](../data/coverage_status.json) : chaque code CIM-11, avec distinction entre parcours publié, cas à examiner et simple renvoi.
- [Lectures des originaux](../editorial/reviews/originaux-v24.json) : preuve du corps et du miroir, empreintes et résumés des actions.
- [Relecture des nouveaux parcours](../editorial/reviews/nouveaux-parcours-v24.json) et [registre des manuscrits](../editorial/clinical/releases.json).
- [Revue des liens et lacunes conceptuelles](../data/psychology_coverage_review.json) : passages exacts, liens directs ou secondaires, rapprochements refusés et angles ouverts des autres référentiels.

Les empreintes de lecture ne sont pas recalculées pour approuver automatiquement un texte modifié. Il faut relire le passage, conserver la décision puis mettre à jour sa preuve. Le contrôle de structure accepte un équivalent complet en prose quand la preuve est courante ; il ne certifie pas la validité psychologique par une expression régulière.

## Ce qui reste ouvert

Les 27 anciennes priorités sont terminées, mais 91 cas du chapitre 06 restent à documenter et à traiter sous la forme appropriée. Les 61 variantes par substance, causes de démence et syndromes secondaires peuvent appeler des comparaisons ou dossiers ; les 30 autres cas doivent être cadrés individuellement. La cartographie ne vaut pas publication. Les sujets associés hors chapitre 06 restent visibles dans le registre.

Les personnalités obsessionnelle-compulsive et dépendante restent nommées pour la comparaison DSM-5-TR. L’inventaire DSM diagnostic par diagnostic n’est pas déclaré terminé. Les standards APA, le RDoC et la CIF demeurent des grilles distinctes ; le détail de leurs angles ouverts est dans la revue des concepts. Les 161 repères internes reliés ne sont pas tous les concepts existants de la psychologie.

## Générer puis contrôler

```sh
python3 scripts/clinical_editorial.py check
python3 scripts/build_article_intros.py
python3 scripts/build_examples.py
python3 scripts/build_coverage_status.py
node scripts/build_navigation.mjs --check
node scripts/build_psychology_catalog.mjs --check
python3 scripts/check_editorial_pack.py
python3 scripts/integrate_editorial_pack.py --check
python3 scripts/check_original_article_structure.py --audit editorial/reviews/originaux-v24.json --require-reviewed
python3 scripts/build_article_intros.py --check
python3 scripts/build_examples.py --check
python3 scripts/build_coverage_status.py --check
node --test tests/*.test.cjs
npm --prefix journal test
python3 -m unittest discover -s journal/tests -p 'test_personal*.py'
hugo --minify
```

Hugo 0.140.0 est fixé dans la configuration de publication. Avant de publier une nouvelle série clinique, documenter et vérifier les sources, relire dans l’ordre et enregistrer ses empreintes avec `clinical_editorial.py record-review`, puis intégrer. Les outils de contrôle ne remplacent pas cette lecture. Après publication, vérifier le commit réellement servi et synchroniser le catalogue d’Ambre avec le même contenu.

## Vérifications de cette étape

397 tests passent : 19 pour le site, 335 pour le journal et 43 pour les traitements Python. Les 756 épisodes cliniques et les 387 articles d’origine ont été contrôlés dans le rendu Hugo ; intertitres et paragraphes des proches concordent avec les sources. Les 387 empreintes de corps et de miroirs sont courantes, sans défaut mécanique ni chemin de preuve inconnu.
