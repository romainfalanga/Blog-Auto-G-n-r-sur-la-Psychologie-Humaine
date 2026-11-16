# Intégration des recherches de cartographie

Intégration locale achevée : 59 questions, 109 cartes actives et 493 liens. Les neuf lots ont été arbitrés, les décisions sémantiques revalidées et les fichiers régénérés. Relecture finale indépendante favorable dans `docs/relecture-finale-cartographie-2026-11-17.md`. Aucun déploiement effectué par l’intégrateur ; les quotas des lots ne constituent pas des objectifs de publication.

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

## Garde-fous techniques

Le générateur refuse plus de14questions par thème, plus de72au total et les résumés de liens de plus de350caractères. Les sujets contextuels ne sont jamais des suggestions automatiques d'Ambre. Son résumé réserve de la place à plusieurs catégories, départage par familles de preuves puis reconnaissance déclarée, et signale les omissions. Les preuves sont réparties à raison d'un lien par sujet avant d'en ajouter d'autres, sous le même budget de6000caractères. La liste publique unique et son tri par nombre de vécus sont conservés ; les cartes restent repliables.

Vérification intermédiaire :38tests de calcul et de backend cartographie passent sur le jeu publié, avant régénération des recherches. Les garde-fous de génération et la couverture finale restent à vérifier sur le jeu intégré.

## Intégration finale des neuf lots

Les fragments et rapports des neuf chercheurs sont conservés intacts. Les arbitrages d’intégration résident dans `corrections.json` et dans le seul fragment `liens-integration-2026-11-17.json`. Les propositions placées hors graphe restent consultables avec leurs références ; elles ne deviennent pas des liens du seul fait de leur présence dans un registre.

| Lot | Décision d’intégration |
| --- | --- |
| 1 — Humeur, anxiété, TOC | Trois nouvelles cartes et huit paires directes ; quatre paires conditionnelles vers le groupe bipolaire historique admises. Groupe bipolaire sans code exact, TOC limité au contexte de sévérité. |
| 2 — Psychose, dissociation, corps | Six cartes et 23 paires directes ; une nouvelle carte de boulimie avec trois paires Caslini distinctes de l’ancien agrégat alimentaire. Définitions fonctionnelles/corporelles corrigées ; toutes les cartes suicidaires contextuelles. |
| 3 — Addictions, conduites, sommeil, sexualité | Quatre cartes symptomatiques et neuf paires admises. Pas de diagnostic opioïde, gaming, explosif intermittent ou conduite créé depuis une étude isolée ou un agrégat. |
| 4 — Personnalité et défenses | Quatre dimensions et douze paires admises. Propositions diagnostiques Waxman/dépendance maintenues hors graphe : étude transversale isolée ou construit non résolu. Traits sensibles et motif borderline contextuels. |
| 5 — Schémas, attachement, émotions | Quatre cartes et vingt paires admises. Honte distincte d’Imperfection ; rumination partielle ; surcompensation et parent exigeant explicitement modèles cliniques. La liste des schémas reste fermée à dix-huit. |
| 6 — Rôles, cognition, relations | Deux cartes et neuf paires admises. Récompense immédiate chez les jeunes contextuelle ; parentification requalifiée comme vécu contextuel. Codépendance limitée, sans diagnostic consensuel. |
| 7 — Santé, trajectoires, ressources | Quatre repères de santé contextuels, quatorze paires et trois ressources. Catégorie `health_outcome`, domaine Corps et comportements : pas de « tendance » psychologique pour une maladie. |
| 8 — Vécus | Une seule question nouvelle (`violence_quartier`), deux liens conditionnels admis et cinq enrichissements directs. Témoin de violence sur la fratrie reporté ; exemples ajoutés aux cases existantes sans déplacer leurs réponses. |
| 9 — Révérification | Les 231 décisions de paire ont été lues : 29 retraits, 25 abaissements, 48 reformulations, trois remplacements et 126 maintiens. Les maintiens ne réécrivent pas les arbitrages plus récents des lots 1/2/8 ou les corrections spécifiques des autres lots. |

Les sources de type `etude` des nouveaux liens ont été vérifiées dans les rapports pour distinguer une cohorte réelle d’une analyse transversale effectuée au sein d’une cohorte. Les propositions strictement reportées par les lots 3 et 4 restent reportées. Une revue systématique de travaux transversaux peut soutenir une association limitée ; une étude transversale isolée ne reçoit pas cette dérogation.

### Corrections de construits et d’historique

- **Honte et schémas** : les trois paires Meng vers `sch_imperfection` sont retirées. Les cinq paires nouvelles de honte ont leur cible propre `ten_honte`. Peter/Chokhani ne deviennent pas une mesure du schéma Méfiance ; outre le retrait du lot 9 pour les violences émotionnelles, les deux paires Chokhani physique/sexuelle sont retirées pour le même motif. Caba mesure un domaine, pas le seul schéma Isolement : sa paire est retirée. Grethel ne mesure pas le schéma Méfiance ni une exposition infantile définie.
- **Attachement** : l’ancien `att_desorganise` décrivait une dynamique adulte. Son sens, son identifiant et ses retours restent dans `retired_outcomes`. Les cinq paires infantiles relues vont vers une nouvelle carte `att_desorganisation_enfant`, `contextual_only` et `young_child_only`, sans URL de couple. Les paires précarité et ressenti de peur ne sont pas transférées. Aucun ancien « Je me reconnais » ne valide la nouvelle carte.
- **Personnalité générale et borderline** : l’alias `icd_6d10 → icd_6d11_5` est retiré. Les quatre paires documentaires dont les sources et résumés portaient réellement sur borderline sont déclarées explicitement avec cette cible ; aucune équivalence entre les deux identifiants n’est publiée. `icd_6d10` reste accepté comme identifiant historique sans suggestion automatique.
- **Migration** : la paire migration → psychoses est retirée. La dernière paire de `pat_migration_adjustment` étudiait des enfants réfugiés, pas toute migration ou changement de région ; cette carte devient historique. Aucun retour ancien ne change de sens ni de cible.
- **Ressentis** : sur les dix-sept paires examinées par le lot 9, quatorze glissements vers pratiques parentales ou construits contemporains sont retirés. Les rapprochements définitionnels avec les schémas restent exclus des convergences. Les trois associations circonscrites effectivement retenues ne représentent pas toutes les expositions possibles.
- **Emprise religieuse** : le retrait de dépression proposé par le lot 9 est étendu à qualité relationnelle, car Saldaña 2022 ne fournit pas d’estimation isolée pour l’entrée dans l’enfance. Cette seconde décision a été revérifiée par l’agent de couverture sur le résumé original : présence d’entrées infantiles et adultes, sans estimation infantile propre. Les modèles de Young sur soumission/punition restent théoriques ; aucune religion n’est assimilée à une emprise.
- **Santé** : les quatre paires de dysfonction familiale → santé physique fondées seulement sur le cumul ACE sont retirées. Les deux paires de négligence utilisent la cohorte cardiovasculaire Soares, sans attribuer un diabète à chaque sous-type. Hotopf ne mesure pas les critères de détresse corporelle : paire diagnostique retirée. Guldin mesure le suicide mortel : aucune conversion automatique vers idées/tentatives/automutilation.

### Exil : annulation documentée des cinq anciens transferts

Le transfert documentaire antérieur `migration → exil` est supprimé. Il ne modifiait pas les réponses enregistrées ; il exposait néanmoins des associations insuffisamment circonscrites au facteur d’enfance. Les cinq paires sont retirées du graphe, sans nier les besoins des personnes déplacées.

| Ancienne cible | Motif final |
| --- | --- |
| Anxiété | Blackmore décrit des prévalences chez 779 enfants réfugiés ; Charlson des populations en zone de conflit à âges variés. Pas de comparaison exploitable de l’exposition infantile seule. |
| Dépression | Blackmore, Charlson et Fazel 2005 décrivent des prévalences en populations exposées ou mixtes. La fréquence dans ces groupes ne démontre pas une association propre à l’exil dans l’enfance. |
| Stress post-traumatique | Blackmore décrit une prévalence infantile ; Fazel 2012 les conditions de vie de jeunes déplacés. Ces résultats ne comparent pas adéquatement le vécu d’exil à son absence ; ils restent documentaires. |
| Stress post-traumatique complexe | Mellor synthétise des prévalences chez réfugiés/déplacés sans exposition infantile identifiable pour cette paire. |
| Deuil prolongé | Groen porte sur des patients adultes et ne mesure pas l’exil pendant l’enfance. |

Les facteurs `exil`, `migration` et `secret_famille` restent des questions valides, avec conservation des réponses et notes. Une sélection qui ne comporte que des facteurs sans lien reçoit un message d’absence de rapprochement documenté. `prot_installation` reste une ressource, avec la population de jeunes réfugiés indiquée ; elle n’annule aucun vécu.

### Notices bibliographiques

Les vingt corrections de source du lot 9 et les notices des lots 4–7 sont appliquées sans duplication d’identifiant. Les formes hétérogènes (`source_override`, `fields`, `source_id`, `source`, champ direct `read`) ont été interprétées séparément des paires. Young pointe vers l’extrait primaire ; les sections effectivement lues par les deux lots sont conservées. McLaughlin devient un modèle théorique, Walker une méta-analyse de prévalence, Mellor/Price/Rossow/Fritz des revues. Les lectures antérieures connues sont distinguées des relectures actuelles, sans déclarer une lecture intégrale à partir d’un résumé. OMS CDDR conserve les sections complémentaires des lots 1–4.

Les trois compléments du lot 6 (parentification–anxiété/dépression, contrôle–imposture) ajoutent les sources sans effacer les anciennes ni relever la force. Les résumés perfectionnistes précisent les dimensions réellement mesurées et la portée des attentes parentales.

## Protection des données déjà enregistrées

`retired_outcomes` contient trois entrées séparées des cartes actives : `att_desorganise`, `icd_6d10`, `pat_migration_adjustment`. Leurs libellés/descriptions historiques restent disponibles pour lire les notes et retours, avec `retired:true` et `contextual_only:true`. Ils n’entrent ni dans les résultats, ni dans les preuves, ni dans les suggestions automatiques. Les nouvelles notes automatiques sur ces cibles sont refusées ; les anciennes notes personnelles restent conservées.

Le navigateur accepte ces identifiants lors de la restauration des brouillons locaux. Le Worker les valide lors d’un autosave et d’une modification : aucun HTTP 400 dû à la seule disparition du graphe. Le générateur refuse désormais la disparition d’un ancien sujet sans entrée historique et la suppression d’un facteur déjà publié. La boulimie exacte et l’attachement infantile n’héritent jamais du retour d’un agrégat ou d’une carte adulte.

## Comptes, budgets et couverture

État final vérifié après ces arbitrages : **59 questions, 109 cartes actives, 3 identifiants historiques, 493 liens, 260 sources actives et 11 ressources**. Six thèmes : 7 violences, 3 manques de soins, 13 famille, 13 climat, 14 ressentis, 9 contexte.

| Nature | Liens |
| --- | ---: |
| Associations empiriques | 363 |
| Modèles cliniques | 88 |
| Définitions | 26 |
| Contextes de classification | 16 |

Forces : 7 fortes, 197 modérées, 160 limitées, 113 théoriques, 16 critères. Les résumés vont jusqu’à 347 caractères, sous le plafond de 350. Le module compact et la page sont contrôlés sous les budgets de 500 000 et 600 000 octets.

Le résumé d’Ambre réserve des places par catégorie et ordonne par familles de preuves puis reconnaissance personnelle. Les preuves répartissent d’abord un lien par sujet ; leur budget de 6 000 caractères porte maintenant sur **tout le JSON**, enveloppes de sujets, libellés, sources et drapeau compris. La relecture indépendante a détecté puis fait corriger le comptage initial incomplet. Le contexte conversation/bilan réduit aussi ses blocs contextuels et exploratoires avant de sacrifier les notes personnelles.

Les **161 repères internes** ont chacun une décision explicite : 69 partiels, 11 à examiner, 65 hors périmètre, 12 références exactes et 4 agrégations. « Référence exacte » décrit une correspondance éditoriale, pas une validation causale. Le registre conserve **1 567 décisions**, avec 798 hors périmètre justifié, 68 recherches non concluantes, 280 regroupements, 410 à examiner et 11 références structurelles exactes. Les cinq statuts sont comptés ; leur somme égale le total. Les décisions des neuf lots et les références non actives restent dans les fragments/registre.

Le chapitre 06 compte toujours 858 codes internes, dont **22 références exactes** actives et **836 sans référence exacte** (dont deux seulement agrégées dans les troubles bipolaires). Les 22 chapitres DSM examinés représentent une présence thématique ; les 273 décisions DSM documentaires ne constituent ni un manuel lu en entier, ni une équivalence clinique CIM/DSM. Les restrictions d’accès et inventaires incomplets restent explicites.

L’audit du 17 novembre est reconstruit par `scripts/audit_cartographie.py` à partir des décisions explicites, après vérification de leur description, de leurs cibles actives et de la validation finale du graphe. Le générateur vérifie l’empreinte du cœur, y compris les identifiants historiques, celles des deux catalogues, des décisions sémantiques et du registre. Changer seulement une empreinte d’audit ne remplace plus la validation finale des décisions.

## Validation locale et limites restantes

Vérifications finales réussies sur les fichiers générés :

- `node --test tests/cartographie.test.cjs journal/tests/cartographie.test.mjs` : **42/42**. La parité couvre les **1 770 combinaisons** de deux facteurs (dont les cas d’un seul facteur), avec intensités et réponse négative distincte, sur les données publiques complètes et le module Worker. Tests réels d’autosave des identifiants historiques et de non-transfert des retours ; budgets de preuves complets à 1 500, 3 000 et 6 000 caractères.
- `PLAYWRIGHT_BROWSERS_PATH=/workspace/work/browsers node tests/cartographie-browser.cjs` : **16/16 scénarios**. Anonyme/compte, autosave sérialisé, erreurs/reprise, conflit de versions, rechargement, données historiques, état sans lien, liste unique et tri, cartes repliées, absence de bandeau3114/bouton de suppression, affichages 1 440/390/320 pixels.
- `python3 scripts/build_cartographie.py --check`, `node scripts/build_cartographie_module.mjs --check`, `python3 scripts/audit_cartographie.py --check` : réussis. Empreinte finale acceptée : `f60aaa891c0c8e884385176e4c61e292123e041c025bc735fe34ff884584146d`.
- `python3 scripts/build_coverage_status.py --check`, `node scripts/build_psychology_catalog.mjs --check`, `git diff --check` : réussis.
- Hugo0.140 : rendu complet réussi (7 931 pages), puis rendu ciblé de Cartographie après le dernier changement de données. Ouverture Playwright du HTML réellement servi en local : 59 questions, six thèmes, cartes fermées, libellé « Repère de santé », aucune erreur JavaScript ni débordement horizontal à390pixels.
- Mesures finales : données publiques **376 116 octets**, module Worker **292 325 octets**, HTML de Cartographie **341 410 octets**. Les budgets de500000 et600000octets sont respectés.

La première passe avait révélé un harness Worker à mettre à jour pour `change-backend`, une assertion figée sur cinq facteurs forts (quatre après relecture) et un dépassement réel du contexte modèle. Ces points ont été corrigés puis vérifiés. La relecture indépendante a en plus fait corriger le comptage complet des preuves et les catégories des quatre repères de santé ; elle a revérifié les fichiers générés finaux. Aucun déploiement n’est effectué par l’intégrateur.

La plupart des données restent observationnelles et nombreuses sont les lectures au résumé ; 316 paires reposent sur une seule source. Les cartes actives ne sont pas l’inventaire de toute la psychologie, et une carte absente n’implique aucune absence de souffrance. Les prévalences, construits voisins et analyses transversales isolées ne sont pas convertis en preuves supplémentaires pour remplir un quota. La revue indépendante finale reste distincte des contrôles de code et ne constitue pas une certification clinique individuelle.
