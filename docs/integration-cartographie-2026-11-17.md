# Intégration des recherches de cartographie

Document de travail. Les neuf lots produisent des propositions ; leurs quotas ne constituent pas des objectifs de publication. Les décisions ci-dessous précèdent la relecture indépendante et la régénération finale.

## Décisions initiales — lots 2 et 8

- **Violence communautaire admise** : instrument ACE-IQ et deux associations limitées issues d'une revue effectivement consultée ; nouvelle question `violence_quartier`, groupe de recouvrement conservateur `physical_violence`. Aucun nouveau point ACE. Total provisoire : 59 questions, six thèmes, aucun thème au-delà de14.
- **Témoin de violences sur la fratrie reporté** : vécu distinct reconnu par MACE, mais les preuves proposées sont deux études transversales. Le seuil de type d'étude du §0 est conservé. Les propositions restent documentées hors graphe ; une absence de carte ne signifie pas absence de souffrance ni absence de littérature.
- **Exemples enrichis sans nouvelles cases** : cyberviolence, auteur institutionnel, violence sexuelle par un pair, discrimination liée à l'identité ou au handicap, violence collective sans exil, décès par suicide et contrôle psychologique. Les identifiants et réponses existantes sont préservés. Aucun exemple de violence ne définit une identité ou une orientation comme pathologique.
- **Boulimie distincte, ancien agrégat préservé** : `icd_6b81` désigne historiquement plusieurs troubles alimentaires. Le transformer en boulimie convertirait aussi les retours déjà enregistrés en un sens plus étroit. L'agrégat conserve son sens ; la carte exacte utilise exceptionnellement `icd_6b81_boulimie`, code vérifié6B81, avec les trois paires spécifiques Caslini. Aucun ancien retour ni note n'est déplacé vers la nouvelle carte. Les cartes anorexie/hyperphagie proposées restent distinctes.
- **Suicidalité** : les trois nouveaux sujets distinguant idées, tentatives et automutilation sans intention suicidaire restent contextuels. L'agrégat existant devient contextuel ; ses associations ne sont pas transférées automatiquement. L'encadré3114 est retiré de la page à la demande du propriétaire ; la voie sécurité d'Ambre conserve ses ressources.
- **Définitions corrigées** : trouble neurologique fonctionnel et détresse corporelle ne sont plus définis par une simple absence de maladie médicale. Les contextes de classification seuls deviennent des repères contextuels.
- **Notice de lecture enrichie** : les métadonnées de sources existantes relues sont mises à jour sans créer de références doublons. Une classification reste un contexte, jamais une estimation empirique.

## Lot 1

- **Bipolarité** : le sujet historique `icd_6a60` indiquait déjà qu'il représentait les troubles bipolaires, tout en affichant le code exact du type I. Son libellé devient un groupe clinique et son code exact est retiré. Les références6A60/6A61 sont explicitement agrégées. Les quatre associations conditionnelles de Palmier-Claus sont admises pour ce groupe ; aucune preuve propre au typeII n'est inventée.
- **TOC** : les liens portent sur la sévérité chez des personnes déjà concernées. La carte est donc contextuelle, sans suggestion automatique à partir des vécus. Périnatalité et symptômes prémenstruels restent aussi contextuels et symptomatiques, sans code diagnostique abusif.

## Garde-fous techniques en cours

Le générateur refuse plus de14questions par thème, plus de72au total et les résumés de liens de plus de350caractères. Les sujets contextuels ne sont jamais des suggestions automatiques d'Ambre. Son résumé réserve de la place à plusieurs catégories, départage par familles de preuves puis reconnaissance déclarée, et signale les omissions. Les preuves sont réparties à raison d'un lien par sujet avant d'en ajouter d'autres, sous le même budget de6000caractères. La liste publique unique et son tri par nombre de vécus sont conservés ; les cartes restent repliables.

Vérification intermédiaire :38tests de calcul et de backend cartographie passent sur le jeu publié, avant régénération des recherches. Les garde-fous de génération et la couverture finale restent à vérifier sur le jeu intégré.
