# Navigation, proches et parcours cliniques

La régression de navigation venait d’un appel à `closeSiteMenu` après un changement d’adresse dans l’espace personnel. Le menu partagé avait repris le masquage de la navigation et la fonction précédente avait disparu. Le correctif vérifie le changement d’écran après chaque clic, puis le retour et l’avance du navigateur. Sur téléphone le logo est centré, les repères des fiches passent à la ligne, et le compteur d’Ambre reste sur une ligne dans son bandeau. Les liens de lecture et de retour n’ont plus de flèche décorative.

## Du côté des proches

Chaque article dispose de son propre texte lié à la scène. Ce texte propose un geste ou une phrase du proche, en fonction de ce que le personnage peut observer, du choix de la personne et des limites de la relation. Les passages consacrés à une situation dangereuse privilégient la mise en sécurité et le relais vers une aide adaptée. Le contrôle de publication exige un texte pour chaque article. Ambre peut s’appuyer sur cette rubrique lorsqu’elle suggère une lecture pour comprendre une relation. Le passage est transmis en entier pour conserver les limites et conditions utiles.

## Repères privés et premier échange

Le premier message invite à raconter une réaction, un échange ou un souvenir, puis indique les pages où écrire ces informations à la main. Dans Mes proches, un réglage unique autorise la mémoire des repères. Après activation, Ambre peut conserver un court extrait exact d’un comportement ou d’un souhait décrit dans la conversation. Le prénom ou pseudonyme unique d’un proche déjà enregistré doit apparaître dans la phrase. Les citations, classées par axe, alimentent la cartographie et le portrait relationnel de cette personne.

Les extraits restent modifiables par retrait dans la fiche et sont inclus dans l’export du compte. Le réglage coupe l’ajout de nouveaux repères et la réutilisation des repères dans Ambre. Les cas ambigus, les demandes de discrétion, les situations de crise et les propos repérés comme sensibles sont écartés de l’enregistrement automatique. La fiche distingue les faits rapportés et les hypothèses. Le stockage automatique ne réécrit pas les champs manuels.

## Parcours cliniques et agent éditorial

La [cartographie CIM-11](cartographie-clinique-v17.md) couvre le chapitre 06 dans la version française 2026-01. Chaque code est rattaché à un parcours ou accompagné de sa raison de regroupement ou d’exclusion. Le canon de chaque parcours donne une vie quotidienne, un enjeu hors diagnostic, des ressources, des liens et une temporalité du diagnostic. Une fiche distincte documente les sujets connexes hors de ce chapitre.

L’action « Atelier éditorial des parcours cliniques » produit soit le dossier d’un personnage, soit des brouillons d’épisodes. Elle reprend le principe de continuité du générateur précédent et transmet au modèle le canon, les points sourcés, les choix des épisodes précédents et la charte issue des relectures. Le mode brouillon requiert le secret GitHub Actions `OPENROUTER_API_KEY` et utilise par défaut le même modèle qu’Ambre. La variable `CLINICAL_EDITOR_MODEL` permet de choisir un autre modèle. Chaque exécution traite un personnage. Les fichiers sortent dans un artefact de travail à relire.

La publication passe par le registre des manuscrits relus, avec l’empreinte du fichier vérifié. Elle intègre les douze articles, les repères, la fiche du personnage et les conseils pour les proches. Les contrôles vérifient l’ouverture, la continuité documentée, les sources, les liens et la couverture du catalogue. La relecture porte sur le sens et les faits, en plus de ces contrôles. La charte conserve les enseignements de chaque vague. Les dossiers cliniques des autres vagues sont à constituer avant leur rédaction.

## Trois vagues relues

| Vague | Parcours | Épisodes |
| --- | --- | ---: |
| 1 | Mathis, Noémie, Inaya et Émilien | 48 |
| 2 | Nour, Antonin, Selma et Gabin | 48 |
| 3 | Garance, Jules, Clémence et Mina | 48 |

Mathis et Jules reçoivent leur diagnostic à l’épisode 3 à l’issue d’une évaluation professionnelle. Les dix autres commencent avec un diagnostic établi. Les 144 nouveaux épisodes et les 387 articles précédents donnent 531 rubriques pour les proches. Il reste 138 parcours et 1 656 épisodes à documenter, rédiger et relire dans le programme du chapitre 06.
