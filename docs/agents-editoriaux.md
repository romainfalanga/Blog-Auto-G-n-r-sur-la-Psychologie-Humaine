# Agents éditoriaux et reprise de la production

Le catalogue public est produit par de vrais sous-agents ChatGPT dans la conversation, conformément à la consigne de rédaction ici et sans génération par GitHub Actions. Le contrôleur `scripts/editorial_pipeline.py` attribue les missions, conserve leur état et vérifie les remises. Il ne contient aucun appel de modèle, n'écrit pas les histoires et ne continue pas après la fermeture de la session. Le site et son accompagnement conservent leur fournisseur existant ; aucune nouvelle clé d'API n'est nécessaire à ce dispositif.

## Rôles

| Rôle | Travail remis | Condition de passage |
| --- | --- | --- |
| Créateur de personnage | Dossier de sources individuel et plan de douze épisodes | Canon, biographie, douze aspects distincts, chronologie diagnostique et sources autorisées validés |
| Rédacteur d'histoires | Douze épisodes d'un même personnage, écrits dans l'ordre | Trame de l'article modèle, continuité et concordance exacte avec le plan |
| Relecteur indépendant | Rapport sur les douze épisodes et corrections précises éventuelles | Personne distincte du rédacteur, empreintes des trois fichiers et preuves de lecture par épisode |
| Pilote | Intégration, tests, déploiement et vérification en ligne | Aucune correction ouverte, fichiers publics identiques aux versions relues, déploiement effectivement observé |

Les cartes permanentes sont dans `editorial/agents/characters.md`, `stories.md` et `review.md`. La charte et l'article d'alexithymie en couple restent les références éditoriales. Chaque épisode explique un aspect réel du trouble ; les codes de classification ne comptent pas comme des épisodes rédigés.

Les personnages avancent en parallèle. Les épisodes d'un même personnage restent confiés au même rédacteur et s'écrivent dans l'ordre, afin de suivre les décisions et leurs conséquences. Le plan et le dossier sont gelés pendant la rédaction. Un problème de source remonte au pilote et impose une nouvelle validation des dépendances.

## Commandes du pilote

```bash
python3 scripts/editorial_pipeline.py init
python3 scripts/editorial_pipeline.py status
python3 scripts/editorial_pipeline.py claim --role character --owner createur_louise --character louise
python3 scripts/editorial_pipeline.py brief --job IDENTIFIANT
python3 scripts/editorial_pipeline.py checkpoint --job IDENTIFIANT --token JETON
python3 scripts/editorial_pipeline.py finish --job IDENTIFIANT --token JETON
```

Le pilote transmet le brief et le jeton retournés à un vrai sous-agent, avec le fichier qu'il est autorisé à modifier. Après `finish` du créateur, une mission `--role story` est attribuée à un propriétaire différent ; après les douze épisodes, une mission `--role review` est attribuée à un autre agent. Une commande `claim` seule ne lance pas ChatGPT. Le script historique `generate_articles.py` ne constitue plus un point d'entrée de génération publique ; ses helpers importés par des outils anciens restent disponibles.

La politique limite le nombre de travaux simultanés et interdit les conflits d'identité entre rôles. Les jetons bruts ne sont jamais enregistrés dans l'état versionné. Les verrous sont courts et interprocessus ; les sauvegardes d'état sont atomiques. Les fichiers des histoires restent ceux des rédacteurs et ne sont pas réécrits par le contrôleur.

## Reprendre après une interruption

Lire `editorial/automation/state.json`, puis `status`. Les étapes sont `character`, `story`, `review`, `ready_to_publish`, `published` et `blocked`. Vérifier les fichiers réellement présents : un compteur d'épisodes sauvegardés ne certifie ni la qualité ni la validation de ces épisodes.

Une mission expirée reste réservée à son propriétaire. Ce dernier reprend avec `claim --resume` ; les brouillons sont conservés. Si les consignes ou les sources ont changé, la mission devient périmée et nécessite un nouvel instantané. Un changement validé du plan exige aussi de revalider sa remise avant de poursuivre la rédaction. Après deux boucles de corrections infructueuses, le pilote doit examiner les remarques au lieu de continuer indéfiniment.

Pour une perte de l'environnement, récupérer d'abord le dernier commit sauvegardé. Ne pas supposer qu'une écriture lancée avant la panne a abouti. Les plans Louise, Constance, Mila et Fanny ont été restaurés le 7 novembre 2026 à partir des messages d'écriture des créateurs ; leurs huit SHA256 ont été comparés aux empreintes initiales et correspondaient octet pour octet.

## Publication

Le rapport indépendant doit être accepté et lié aux SHA256 actuels du dossier, du plan et du manuscrit. `clinical_editorial.py record-review` et `integrate` revérifient ces éléments. Le fichier `editorial/automation/legacy-publications.json` épingle les 63 manuscrits déjà relus au commit `c66c1d6078ef5bf29f29a608200175662fbba852`. Cette exception ne vaut que pour leurs octets exacts ; une nouvelle rédaction requiert les preuves indépendantes même si les fichiers de plan et de rapport ont disparu.

Après intégration, régénérer les introductions, le catalogue d'exemples et l'état de couverture, puis exécuter les vérifications du projet et le rendu Hugo. Publier le code, vérifier le commit réellement prêt chez l'hébergeur, synchroniser le catalogue du service d'accompagnement et ouvrir les nouveaux articles et portraits en ligne.

Après le rendu Hugo et le déploiement, `python3 scripts/check_published_arcs.py ID1 ID2 --report /tmp/publication.json` compare le texte intégral de chaque récit servi (y compris la rubrique des proches) au rendu local. Une page d’erreur, une section absente ou un texte différent fait échouer la commande. Ce contrôle de fidélité ne remplace ni la relecture clinique ni la vérification du catalogue d’Ambre.

Exécuter ensuite seulement `editorial_pipeline.py publish --character ID` et sauvegarder l'état mis à jour. Cette commande contrôle les fichiers intégrés et les empreintes locales ; elle n'interroge pas elle-même l'hébergeur et ne prouve pas un déploiement distant. Le pilote consigne séparément sa vérification en ligne. `status.publication_files_current` est une mesure de fraîcheur des preuves et fichiers locaux, pas une mesure de disponibilité du site.

Les tests CI vérifient le dispositif ; ils ne génèrent aucun article. L'exhaustivité reste mesurée dans `docs/etat-couverture.md` et `data/coverage_status.json`, avec les cas non rédigés explicitement conservés dans la file.
