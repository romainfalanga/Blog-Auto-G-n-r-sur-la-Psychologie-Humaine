# Navigation, proches et parcours cliniques

La régression de navigation venait d’un appel à `closeSiteMenu` après un changement d’adresse dans l’espace personnel. Le menu partagé avait repris le masquage de la navigation et la fonction précédente avait disparu. Le correctif vérifie le changement d’écran après chaque clic, puis le retour et l’avance du navigateur. Sur téléphone le logo est centré, les repères des fiches passent à la ligne, et le compteur d’Ambre reste sur une ligne dans son bandeau. Les liens de lecture et de retour n’ont plus de flèche décorative.

## Du côté des proches

Chaque article dispose de son propre texte lié à la scène. Ce texte propose un geste ou une phrase du proche, en fonction de ce que le personnage peut observer, du choix de la personne et des limites de la relation. Les passages consacrés à une situation dangereuse privilégient la mise en sécurité et le relais vers une aide adaptée. Le contrôle de publication exige un texte pour chaque article. Ambre peut s’appuyer sur cette rubrique lorsqu’elle suggère une lecture pour comprendre une relation. Le passage est transmis en entier pour conserver les limites et conditions utiles.

## Repères privés et premier échange

Le premier message présente Ambre et explique comment comprendre son propre fonctionnement et ses relations à partir des échanges. Il indique ensuite les pages où écrire soi-même. Avec l’accord de conversation Ambre, un court extrait exact d’un comportement ou d’un souhait rapporté peut être conservé automatiquement. Le prénom ou pseudonyme unique d’un proche déjà enregistré doit apparaître dans la phrase. Les citations, classées par axe, alimentent sa cartographie et son portrait relationnel.

Les extraits peuvent être retirés dans la fiche et sont inclus dans l’export du compte. Le retrait de l’accord Ambre arrête l’ajout et le rappel des repères dans la conversation. L’effacement de la conversation supprime les repères associés. Une désactivation enregistrée avec l’ancien réglage reste respectée jusqu’à un nouvel accord qui inclut la mémorisation. Les cas ambigus, les demandes de discrétion, les situations de crise et les propos repérés comme sensibles sont écartés de l’enregistrement automatique. Les champs manuels restent distincts des citations.

## Lecture du portrait

Le menu d’onglets s’aligne sur le panneau du portrait. Sur téléphone, Mon portrait occupe la première ligne. Les pistes et la synthèse disponibles précèdent l’état de leur mise à jour. Une erreur de chargement permet de retenter l’affichage en conservant le brouillon de présentation.

Le contrôle de formulation distingue les pronoms ambigus des tournures impersonnelles dont le sujet est clair. La validation de l’accord, des citations et des sources reste en place. Le planificateur réserve une tentative aux analyses qui portent l’ancien motif de rejet de formulation, dans les plafonds déjà prévus.

## Parcours cliniques et agent éditorial

La [cartographie CIM-11](cartographie-clinique-v17.md) couvre le chapitre 06 dans la version française 2026-01. Chaque code est rattaché à un parcours ou accompagné de sa raison de regroupement ou d’exclusion. Le canon de chaque parcours donne une vie quotidienne, un enjeu hors diagnostic, des ressources, des liens et une temporalité du diagnostic. Une fiche distincte documente les sujets connexes hors de ce chapitre.

La rédaction se fait directement avec un agent dédié par personnage. La commande `python3 scripts/clinical_editorial.py agent <identifiant>` réunit son canon, la charte commune, les sources vérifiées, sa direction éditoriale et les choix déjà écrits. Le dossier individuel se trouve dans `editorial/clinical/dossiers/` et le manuscrit de douze épisodes dans `editorial/clinical/`. Les points d’étape conservent les faits et les engagements d’une scène à la suivante. La commande `progress` distingue les dossiers de sources, les épisodes rédigés et les parcours relus.

Après la relecture du parcours, `record-review` inscrit la date, les observations et les empreintes du manuscrit et de son dossier dans le registre. L’intégration prépare les douze articles, les repères, la fiche du personnage et les conseils pour les proches. Les contrôles vérifient l’ouverture, la continuité documentée, les sources, les liens et la couverture du catalogue. Les affirmations et les raccords sont relus dans leur contexte. La charte conserve les enseignements de chaque vague.

Les histoires d’origine servent de référence. Chaque fiche clinique indique le trouble dans sa présentation et en tête de ses cinq mots-clés. Le titre et l’introduction annoncent ce que la scène permet de comprendre. Les cartes des épisodes reprennent cette introduction et trois repères propres à l’épisode. La scène, l’explication, les pistes concrètes, le retour au récit et la perspective des proches développent cet angle. L’[audit des 32 parcours](audit-parcours-cliniques-v19.md) détaille les corrections.

## Vagues relues

| Vague | Parcours | Épisodes |
| --- | --- | ---: |
| 1 | Mathis, Noémie, Inaya et Émilien | 48 |
| 2 | Nour, Antonin, Selma et Gabin | 48 |
| 3 | Garance, Jules, Clémence et Mina | 48 |
| 4 | Nils, Solène, Meriem et Armand | 48 |
| 5 | Blandine, Sacha, Adèle et Côme | 48 |
| 6 | Hélène, Roxane, Owen et Agathe | 48 |
| 7 | Célian, Maïwenn, Lison et Augustin | 48 |
| 8 | Irène, Zéphyr, Diane et Gasparde | 48 |

Mathis, Jules et Sacha reçoivent leur diagnostic à l’épisode 3 à l’issue d’une évaluation professionnelle. À l’épisode 2, Diane revient sur le diagnostic posé pendant son hospitalisation. Ces 384 épisodes portent le nombre de rubriques pour les proches à 771. Le [registre des relectures](../editorial/clinical/releases.json) conserve la vérification de chaque parcours.

La suite se décide à partir des [priorités de couverture](cartographie-clinique-v17.md#priorités-de-couverture) et de l’apport de chaque sujet. Les canons du registre restent des pistes de travail.

## Reprise de la trame de Youssef

L’[article sur l’alexithymie en couple](https://decodetonesprit.fr/posts/comprendre-et-maitriser-ses-emotions/alexithymie-en-couple/) sert de modèle. Une scène ouvre chaque épisode sur un aspect précis du trouble. Une explication précède trois manifestations avec leurs exemples. Trois techniques adaptées au contexte préparent le retour à l’histoire. La conclusion et « Du côté des proches » restent liées à la scène. Les titres commencent par le sujet et les références restent dans les dossiers éditoriaux.

Les 48 parcours déjà publiés à l’ouverture de cette reprise, soit 576 épisodes, suivent la trame. Le registre conserve les dates, les empreintes des manuscrits et les notes de relecture. Les épisodes reprennent leurs adresses et leur place dans les parcours.

Le catalogue clinique compte 59 parcours et 708 épisodes à cette trame. Les 27 sujets de la file prioritaire actuelle ont un parcours publié. Les dossiers comparatifs et les autres cas restent suivis dans la cartographie.

Le [dossier préparatoire de Malo](../editorial/clinical/dossiers/malo.json) documente la catatonie parmi les autres sujets à cadrer.

`python3 scripts/clinical_editorial.py progress` indique les priorités, les dossiers, les manuscrits et les parcours déjà relus. Pour une reprise, partir de ces états et du registre avant de rédiger ou d’intégrer une série.
