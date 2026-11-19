# Relecture clinique indépendante « Mes symptômes » (19 novembre 2026)

## Procédure

- **Relecteur distinct** : une instance Claude séparée, lancée en mode non interactif (`claude -p --tools ""`), sans outils, sans accès au dépôt ni à la conversation de rédaction. L'outil « Agent » n'était pas disponible dans la session de rédaction ; cette instance séparée en tient lieu. Le relecteur n'a pas écrit le contenu.
- **Entrée** : un dossier de relecture généré depuis `editorial/symptomes/symptomes.json` (avant corrections) : textes fixes, sécurité, 23 modules (questions, échelles, périodes, codes, règles, sujets croisés, textes de résultat), seuils des instruments. Empreinte SHA-256 du dossier transmis : `4ebd67dbb93c399e242eb6fff0918fbca95be4ae139c543dfdca455de6e210c2`. Les items officiels n'étaient pas soumis à correction.
- **Consigne** : juger fidélité CIM-11 / CDDR, stigmatisation, suggestion, lisibilité, correspondances de sujets, sécurité ; statut `relu` (accepté sans correction), `retenu` (accepté à condition d'appliquer les corrections), `écarté` (non publié).
- **Limite déclarée par le relecteur** : relecture faite de mémoire de la CIM-11 / CDDR, sans consultation de sources ; numéros et codes marqués « à vérifier ». Ce n'est pas la relecture d'un clinicien humain.

## Statut par domaine

| # | Domaine | Statut | Publié | Corrections appliquées |
| --- | --- | --- | --- | --- |
| 1 | `humeur` — Humeur basse et perte d’intérêt | retenu | oui | toutes, voir écarts ci-dessous |
| 2 | `anxiete` — Inquiétude et tension | retenu | oui | toutes, voir écarts ci-dessous |
| 3 | `trauma_stress` — Après un événement très éprouvant ; deuil ; adaptation | retenu | oui | toutes, voir écarts ci-dessous |
| 4 | `panique` — Crises de peur soudaines et évitement de lieux | retenu | oui | toutes, voir écarts ci-dessous |
| 5 | `anxiete_sociale` — Peur du regard des autres | relu | oui | aucune demandée |
| 6 | `separation` — Peur d’être séparé(e) de ses proches | retenu | oui | toutes, voir écarts ci-dessous |
| 7 | `toc_apparentes` — Pensées qui s’imposent, rituels, apparence, santé | retenu | oui | toutes, voir écarts ci-dessous |
| 8 | `bipolaire` — Périodes d’énergie ou d’humeur inhabituellement hautes | retenu | oui | toutes, voir écarts ci-dessous |
| 9 | `sommeil` — Sommeil et cauchemars | retenu | oui | toutes, voir écarts ci-dessous |
| 10 | `somatique` — Symptômes physiques qui pèsent | retenu | oui | toutes, voir écarts ci-dessous |
| 11 | `alimentation` — Alimentation, poids et image du corps | retenu | oui | toutes, voir écarts ci-dessous |
| 12 | `regulation` — Émotions qui débordent, colère, impulsivité | retenu | oui | toutes, voir écarts ci-dessous |
| 13 | `alexithymie` — Reconnaître et dire ses émotions | retenu | oui | toutes, voir écarts ci-dessous |
| 14 | `soi` — Image de soi, exigences, honte et culpabilité | retenu | oui | toutes, voir écarts ci-dessous |
| 15 | `relations` — Liens proches, attachement, conflits | retenu | oui | toutes, voir écarts ci-dessous |
| 16 | `jeu_ecrans` — Jeux d’argent, jeux vidéo, écrans | retenu | oui | toutes, voir écarts ci-dessous |
| 17 | `alcool` — Alcool | retenu | oui | toutes, voir écarts ci-dessous |
| 18 | `cannabis` — Cannabis | retenu | oui | toutes, voir écarts ci-dessous |
| 19 | `autres_substances` — Tabac et autres produits | retenu | oui | toutes, voir écarts ci-dessous |
| 20 | `dissociation` — Sentiment de détachement, trous de mémoire, symptômes du corps | retenu | oui | toutes, voir écarts ci-dessous |
| 21 | `perceptions` — Perceptions et idées inhabituelles | retenu | oui | toutes, voir écarts ci-dessous |
| 22 | `personnalite` — Façons durables d’être avec soi et avec les autres | retenu | oui | toutes, voir écarts ci-dessous |
| 23 | `sexualite` — Vie sexuelle | retenu | oui | toutes, voir écarts ci-dessous |
| — | sécurité | retenu | — | toutes, voir écarts |
| — | instruments (seuils, règles, textes) | retenu | — | toutes |

Aucun domaine écarté : 23 domaines publiables (`publie: true`), 1 `relu`, 22 `retenu`.

## Application des corrections et écarts assumés

Toutes les corrections ont été appliquées dans `editorial/symptomes/symptomes.json` (champ `revised: "relecture"` sur les questions réécrites), puis `python3 scripts/build_symptomes.py --check` a été relancé avec succès. Écarts ou adaptations :

1. **`panique_inquietude`** : texte appliqué ; l'étiquette de période reste « Au cours du dernier mois » (la question ne dit plus « pendant des semaines », ce qui supprime la contradiction relevée).
2. **`humeur_gene`** : période « Au cours des 2 dernières semaines » (période définie du site la plus proche de « ces dernières semaines »).
3. **`reg_automutilation` = Oui** : la v1.1 (qui fait foi) impose l'écran de sécurité ; il reste **bloquant**, avec le message proposé par le relecteur (`safety.display.blocking_screen.messages.automutilation`). « Je préfère ne pas répondre » → bloc doux.
4. **`icd_6b81`** : conservé dans `repere_restriction` et `repere_crises`, car son libellé dans la cartographie est le sujet général « Troubles des conduites alimentaires » (6B80–6B82 agrégés), cas que le relecteur autorise.
5. **`itq_tspt_c`** : le libellé affiché de `icd_6b40` est « Trouble de stress post-traumatique » (non générique) : sujet retiré, comme demandé.
6. **`att_anxieux`, `att_evitant`, `pat_social_isolation`** : conservés avec des libellés descriptifs imposés (`crossing.display_labels`).
7. **Numéros ajoutés à vérifier avant mise en ligne** : 116 006 (France Victimes), 0 980 980 930 (Alcool Info Service), 0 800 23 13 13 (Drogues Info Service), 09 74 75 13 13 (Joueurs Info Service), 39 89 (Tabac Info Service). Le numéro d'Anorexie Boulimie Info Écoute n'est pas écrit (seul le nom figure) ; les horaires du Fil Santé Jeunes ne sont pas écrits.
8. **Code sommeil** : 7B01.1 → 7B01.2 (trouble cauchemars), à vérifier sur icd.who.int avec 7A00, la plage HA00–HA20 et les codes 6C42–6C4E / 6C4A.
9. **Bouton Ambre** de l'écran de sécurité : renommé « Écrire à Ambre (assistant automatique, pas un service d'urgence) » et placé en premier ; à retirer au lot C si Ambre ne suit pas le protocole déterministe existant.
10. **Non appliqué, hors périmètre du lot 0** : AUDIT-C comme entrée alcool (licence OMS non commerciale, instrument « sous réserve » dans le plan) ; repères TDAH / autisme / phobies spécifiques (hors périmètre ; mention ajoutée dans `fixed_texts.methode_absences`). Ajouts généraux retenus : `several_positive`, `privacy_sensitive`, `site_note`, `high_note`, `gene_absente_trouble`.

## Points incertains restant ouverts

- Fidélité vérifiée par un relecteur automatique de mémoire : une relecture par un clinicien humain reste souhaitable avant publication publique.
- Seuils d'entrée PHQ-2 / GAD-2 à 2 (v1.1) : jamais présentés comme « repère publié » (le manuel officiel donne 3).
- PHQ-15, item 1d « non concerné » compté 0 ; score affiché « sur 28 » dans ce cas.
- Coquille officielle de l'item ITQ P1 recopiée telle quelle ; demande de confirmation au Cn2r.
- Croisement : les sujets `icd_*` gardent leur libellé de cartographie (noms de troubles) ; le relecteur demande des formulations descriptives : à trancher au lot B.

## Rapport du relecteur (texte intégral, non modifié)

# Rapport de relecture clinique indépendante : page « Mes symptômes »

Relecture faite de mémoire de la CIM-11 / CDDR (OMS 2024), sans consultation de sources. Les numéros de téléphone et codes marqués « à vérifier » doivent être contrôlés avant publication.

**Bilan : 1 module relu, 22 retenus sous corrections, 0 écarté.**

---

### humeur
Statut : retenu
Corrections :
1. `humeur_perinatal` : « Êtes-vous enceinte, ou avez-vous accouché au cours des 12 derniers mois ? » → « Êtes-vous enceinte, ou un enfant est-il né dans votre foyer au cours des 12 derniers mois ? ». Remplacer l'étiquette « 6E20 » par « contexte (site) ».
2. `contexte_perinatal`, positif : ajouter à la fin « Si des pensées vous effraient, pour vous ou pour votre bébé, parlez-en vite à un médecin ; à toute heure, le 3114 répond. »
3. Ajouter une règle d'orientation si PHQ-9 ≥ 10 : « Si vous avez aussi connu des périodes d'énergie ou d'humeur inhabituellement hautes, dites-le au professionnel : cela compte pour le choix de l'aide. »
4. `humeur_gene`, étiquette de période : « Ces derniers mois » → « Ces dernières semaines ».

Motif : `humeur_duree` est fidèle à 6A72. La CIM-11 limite 6E20 à environ six semaines après l'accouchement, donc la fenêtre de 12 mois est un choix du site, et la question doit aussi convenir au co-parent. Un PHQ-9 élevé ne distingue pas un épisode dépressif d'un trouble bipolaire.

### anxiete
Statut : retenu
Corrections :
1. `gad7_repere`, positif : ajouter « Le café, certains médicaments ou un problème de thyroïde peuvent donner des signes proches ; un médecin peut le vérifier. »

Motif : `anxiete_duree` est fidèle à 6B00 (plusieurs mois, plus d'un jour sur deux, plusieurs sujets). Il manque seulement l'orientation vers une cause physique ou une substance, exclusion essentielle.

### trauma_stress
Statut : retenu
Corrections :
1. `itq_tspt`, négatif : « Vos réponses à l'ITQ ne remplissent pas la règle publiée ce mois-ci. » → « Vos réponses à l'ITQ ne remplissent pas la règle publiée pour le dernier mois. » Ajouter la condition : n'afficher ce négatif que si le résultat n'est ni `tspt` ni `tspt_c`.
2. `itq_tspt_c` : sujet `icd_6b40` à conserver seulement si le libellé affiché est générique (« stress post-traumatique ») ; sinon supprimer le sujet `icd_6b40` de cette règle (6B40 et 6B41 s'excluent dans la CIM-11).
3. `itq_menace` : ajouter un texte positif, sinon un sujet apparaît dans le croisement sans résultat visible : « Vous indiquez être souvent sur vos gardes ou sursauter facilement depuis cet événement. »
4. `deuil_douleur` : « Depuis, son absence vous occupe-t-elle presque chaque jour, avec un manque ou une douleur très intense ? » → « Aujourd'hui encore, pensez-vous à cette personne ou vous manque-t-elle presque chaque jour, avec une douleur très intense ? »
5. `repere_deuil` : ajouter la condition `trauma_gene in [un_peu, nettement]`. Positif : ajouter « La durée d'un deuil varie selon les personnes, les familles et les cultures. »
6. `repere_adaptation` : ajouter la condition « ITQ ni `tspt` ni `tspt_c`, et `repere_deuil` négatif ». Positif : ajouter « Ce repère ne s'applique pas si une autre difficulté (humeur, deuil, stress post-traumatique) décrit mieux ce que vous vivez. »
7. Avant l'ITQ, ajouter : « Les questions suivantes parlent des suites de cet événement. Vous pouvez vous arrêter à tout moment. »

Motif : pour 6B42, la CIM-11 exige une réaction qui persiste aujourd'hui, dépasse les normes du contexte et gêne le fonctionnement ; « Depuis » ne dit pas que cela dure encore. 6B43 est un diagnostic d'exclusion. L'ITQ peut réactiver des souvenirs.

### panique
Statut : retenu
Corrections :
1. `panique_inquietude` : « Depuis, avez-vous craint pendant des semaines qu'elles reviennent, ou changé vos habitudes pour les éviter ? » → « Depuis ces crises, craignez-vous souvent qu'elles reviennent, ou avez-vous changé vos habitudes pour les éviter ? » ; étiquette de période → « Ces dernières semaines ».
2. `repere_panique`, positif : « Des sensations physiques fortes méritent aussi un avis médical, pour vérifier qu'aucune cause physique n'est en jeu. » → « Ces sensations méritent aussi un avis médical : cœur, thyroïde, café, cannabis ou médicaments peuvent donner des crises semblables. Une douleur dans la poitrine ou un malaise inhabituel : appelez le 15. »
3. `agoraphobie_raison` : « …d'en partir ou d'être aidé(e) en cas de malaise ? » → « …d'en partir ou d'être aidé(e) en cas de malaise ou de crise de peur ? »

Motif : les éléments de 6B01 et 6B02 sont bien rendus. « Pendant des semaines » contredit une fenêtre d'un mois, et l'orientation somatique doit nommer l'urgence.

### anxiete_sociale
Statut : relu
Corrections : aucune.
Motif : les trois questions reprennent les éléments essentiels de 6B04 (peur marquée, crainte d'une évaluation négative, évitement ou endurance avec peur intense), sans jugement ni suggestion.

### separation
Statut : retenu
Corrections :
1. `repere_separation`, positif : ajouter « Si cette peur répond à une situation réelle (proche malade, danger), ce repère ne s'applique pas. »

Motif : les questions sont fidèles à 6B05 chez l'adulte, mais le caractère « excessif » exigé par la CIM-11 n'est pas testé. Une inquiétude réaliste serait signalée à tort.

### toc_apparentes
Statut : retenu
Corrections :
1. `toc_pensees` : « …des pensées, des images ou des envies vous sont-elles revenues encore et encore, contre votre gré, en vous angoissant ? » → « …des pensées, des images ou des impulsions non voulues vous sont-elles revenues encore et encore, en vous angoissant ? »
2. `repere_toc`, positif : ajouter « Avoir des pensées qui s'imposent ne veut pas dire qu'on les souhaite, ni qu'on va les réaliser. »
3. `apparence_defaut` : « …à un défaut de votre apparence que les autres remarquent peu ou pas ? » → « …à un défaut de votre apparence (autre que le poids) que les autres remarquent peu ou pas ? »
4. `repere_sante`, positif : « Un médecin de confiance peut faire le point avec vous, sans multiplier les examens. » → « Un médecin de confiance peut vérifier ce qui doit l'être, puis vous aider à apaiser cette peur. Avec une maladie connue ou un risque réel, cette peur peut être compréhensible. »

Motif : « envies » se lit comme des désirs, alors que la CIM-11 parle d'impulsions intrusives. Une préoccupation limitée au poids est exclue de 6B21. « Sans multiplier les examens » peut détourner d'un examen nécessaire une personne qui évite les médecins.

### bipolaire
Statut : retenu
Corrections :
1. `bipolaire_periode` : « …une période d'au moins plusieurs jours où… » → « …une période de plusieurs jours ou plus où… »
2. `bipolaire_signes` : « Pendant cette période, aviez-vous besoin de beaucoup moins dormir, parliez-vous beaucoup plus, ou aviez-vous des pensées très rapides ? » → « Pendant cette période, aviez-vous au moins deux de ces changements : dormir très peu sans fatigue, parler beaucoup plus, pensées très rapides, dépenses ou risques inhabituels ? »
3. `bipolaire_remarque` : « Vos proches ont-ils remarqué que vous n'étiez pas comme d'habitude pendant cette période ? » → « Cette période était-elle nettement différente de votre façon d'être habituelle, au point que des proches l'ont remarqué ou auraient pu le remarquer ? »
4. `repere_bipolaire` : supprimer le sujet `icd_6a60`.
5. `repere_bipolaire`, positif : ajouter « Si une telle période est en cours et que vous dormez très peu, consultez sans attendre. »

Motif : la CIM-11 demande plusieurs symptômes associés, pas un seul. « Moins dormir » sans préciser « sans fatigue » capte l'insomnie. Croiser « bipolaire de type I » avec l'enfance à partir de quatre questions est trop suggestif, car ce type de repérage donne beaucoup de faux positifs.

### sommeil
Statut : retenu
Corrections :
1. En-tête du module : code « 7B01.1 » → « 7B01.2 » (trouble cauchemars ; de mémoire, 7B01.1 désigne la paralysie du sommeil isolée récurrente ; à vérifier).
2. `sommeil_insomnie` : « Depuis au moins trois mois, plusieurs nuits par semaine, avez-vous du mal à vous endormir, à rester endormi(e), ou vous réveillez-vous trop tôt ? » → « Depuis au moins trois mois, plusieurs nuits par semaine, dormez-vous mal (endormissement, réveils, réveil trop tôt) alors que vous avez le temps et un lieu pour dormir ? »
3. `sommeil_journee` : « Ce sommeil difficile vous pèse-t-il dans la journée (fatigue, humeur, concentration), alors que vous avez le temps de dormir ? » → « Ce sommeil difficile vous pèse-t-il dans la journée (fatigue, humeur, concentration) ? »
4. `repere_cauchemars` : ajouter la condition `sommeil_gene in [un_peu, nettement]`. Positif : ajouter « Des cauchemars fréquents peuvent être liés à un événement éprouvant, à l'alcool ou à certains médicaments ; un médecin peut en parler avec vous. »

Motif : `sommeil_journee` porte deux idées, et un « Non » y est ininterprétable. Pour les cauchemars, la CIM-11 exige une détresse ou une gêne, ce qu'une seule question « souvent » ne garantit pas.

### somatique
Statut : retenu
Corrections :
1. `somatique_attention` : « Ces symptômes occupent-ils une grande place dans vos pensées, ou vous font-ils consulter très souvent ? » → « Ces symptômes occupent-ils une grande place dans vos pensées, ou vous font-ils consulter très souvent, même après des examens rassurants ? »
2. `somatique_duree` : « Sont-ils là la plupart des jours depuis plusieurs mois, malgré des examens et des avis médicaux ? » → « Ces symptômes sont-ils présents la plupart des jours depuis plusieurs mois ? »
3. `phq15_repere` : supprimer le sujet `ten_douleurs_persistantes`.
4. `repere_detresse_corporelle`, positif : ajouter « Un symptôme nouveau, qui s'aggrave ou qui vous inquiète doit d'abord être examiné par un médecin. »

Motif : dans 6C20, c'est l'attention excessive qui persiste malgré les examens, pas le symptôme ; la CIM-11 n'exige pas des symptômes « inexpliqués ». Le PHQ-15 mesure une charge de symptômes variés sur quatre semaines, pas une douleur persistante.

### alimentation
Statut : retenu
Corrections :
1. `alim_perte_controle` : « …des moments où vous perdiez le contrôle de ce que vous mangiez ? » → « …des moments où vous mangiez plus ou autrement que d'habitude, sans pouvoir vous arrêter ? »
2. `alim_compensation` : « Après ces moments, vous êtes-vous fait vomir, avez-vous jeûné, pris des laxatifs ou fait beaucoup de sport pour ne pas prendre de poids ? » → « Après ces moments, vous arrive-t-il régulièrement de vous faire vomir, de jeûner, de prendre des laxatifs ou de faire beaucoup de sport pour ne pas grossir ? »
3. `repere_restriction` : supprimer le sujet `icd_6b81`. Ne pas afficher la phrase fixe positive après ce résultat ; la remplacer par « Même si vous vous sentez bien, un avis médical dans les prochains jours est important. »
4. `repere_crises` : supprimer le sujet `icd_6b81`. Si `icd_6b81` désigne en réalité un sujet général « conduites alimentaires », le renommer et le conserver.
5. `repere_crises_compensation`, positif : ajouter « Anorexie Boulimie Info Écoute peut aussi vous répondre. » (numéro 09 69 325 900 et horaires à vérifier).

Motif : 6B81 est la boulimie ; l'attacher à une restriction avec poids bas ou à des crises sans compensation est une correspondance fausse. Dire « beaucoup de personnes n'ont pas de trouble » après un repère de restriction avec poids bas rassure à tort, car la minimisation fait partie du tableau.

### regulation
Statut : retenu
Corrections :
1. `reg_automutilation` : « En ce moment, vous arrive-t-il de vous faire du mal volontairement, par exemple en vous coupant, en vous brûlant ou en vous frappant ? » → « En ce moment, vous arrive-t-il de vous blesser volontairement (coups, coupures…) ? »
2. Ajouter la règle `securite_automutilation` si « Oui », non bloquante, avec les ressources « automutilation » : « Merci de l'avoir dit. Se blesser est souvent une façon de supporter une souffrance très forte. Vous pouvez en parler à un médecin, ou au 3114 à toute heure. Blessure grave : 15. » Si « Je préfère ne pas répondre » : afficher la version douce.
3. `repere_intensite` et `repere_impulsivite` : remplacer la phrase fixe positive par la note « repère du site » (voir bloc securite, correction 8).
4. `repere_colere`, positif : ajouter « L'alcool ou d'autres produits peuvent favoriser ces accès. »

Motif : `reg_colere` est fidèle à 6C73. La réponse « Oui » à la question de sécurité n'a aucune suite décrite. Deux items très banals (émotions fortes, coups de tête) ne justifient pas « un échange avec un professionnel est utile ».

### alexithymie
Statut : retenu
Corrections :
1. `repere_alexithymie` et `repere_retenue` : remplacer la phrase fixe positive par la note « repère du site » (bloc securite, correction 8). La phrase actuelle contredit « pas un trouble ».
2. `repere_retenue`, positif : « Vous indiquez garder souvent vos émotions pour vous. » → « Vous indiquez garder souvent vos émotions pour vous. C'est fréquent et dépend aussi de l'éducation et de la culture. »

Motif : le construit est correctement présenté comme un trait de recherche, et la séparation entre alexithymie et retenue émotionnelle est juste. Seul le texte fixe pathologise.

### soi
Statut : retenu
Corrections :
1. `repere_imposture`, positif : « Vous indiquez un sentiment d'imposture. » → « Vous indiquez attribuer souvent vos réussites à la chance et craindre qu'on découvre vos limites. »
2. `repere_valeur`, positif : « Vous indiquez une valeur de vous-même liée à la réussite. » → « Vous indiquez avoir l'impression de valoir quelque chose surtout quand vous réussissez. »
3. Toutes les règles `repere_site` du module : remplacer la phrase fixe positive par la note « repère du site ».
4. Ajouter une règle d'orientation si au moins deux « Oui » parmi `soi_estime`, `soi_honte`, `soi_culpabilite` : « Ces ressentis accompagnent parfois une humeur basse : le module « Humeur basse et perte d'intérêt » peut compléter ce repère. »

Motif : un résultat doit reprendre les mots de la question, pas poser une étiquette. La dévalorisation et la culpabilité sont aussi des symptômes dépressifs, ce que le module ne dit pas.

### relations
Statut : retenu
Corrections :
1. `repere_oubli_soi` : supprimer le sujet `ten_codependance`. Positif : ajouter « Cela peut aussi tenir à une situation, par exemple aider un proche malade. »
2. `repere_peur_perte` et `repere_distance` : conserver `att_anxieux` et `att_evitant` seulement si le libellé affiché est descriptif (« peur de perdre ses proches », « malaise dans la proximité ») ; ne jamais afficher « attachement anxieux » ou « attachement évitant ».
3. `repere_isolement` : libellé affiché du sujet `pat_social_isolation` → « sentiment de solitude ».
4. Ajouter la règle `securite_violences` si `rel_violences` = « Oui », avec les ressources « violences » : « Merci de l'avoir dit. Personne n'a le droit de vous faire peur, de vous frapper, de vous humilier ou de vous forcer. Vous pouvez en parler au 3919, à toute heure, ou à France Victimes (116 006). En danger : 17 ou 112. » (116 006 à vérifier). Si « Je préfère ne pas répondre » : afficher discrètement 3919 et 17.
5. Ajouter la condition : si `rel_violences` = « Oui », ne croiser aucun sujet du module et afficher « Dans une relation où l'on a peur, ces réactions sont des façons de se protéger. »
6. Toutes les règles `repere_site` : remplacer la phrase fixe positive par la note « repère du site ».

Motif : « codépendance » n'est pas un construit validé et blâme la personne. Un seul item ne fonde pas un « style d'attachement ». Interpréter comme un schéma personnel la peur ou l'effacement d'une personne victime de violences serait une faute clinique.

### jeu_ecrans
Statut : retenu
Corrections :
1. `ecrans_controle` : « …avez-vous eu du mal à contrôler le temps passé aux jeux vidéo, sur internet ou sur les écrans ? » → « …avez-vous eu du mal à contrôler le temps passé aux jeux vidéo, aux réseaux sociaux ou sur d'autres écrans, hors travail et études ? »
2. `repere_ecrans`, positif : « Vos réponses ressemblent à certains éléments que la CIM-11 décrit pour les jeux vidéo ; pour les autres écrans, c'est un repère du site. » → « Vous indiquez un temps d'écran difficile à contrôler, qui passe avant le reste et continue malgré ses conséquences. La CIM-11 ne décrit ce tableau que pour les jeux vidéo. »
3. `repere_jeu`, positif : « Joueurs Info Service et un médecin peuvent vous aider. » → « Joueurs Info Service (09 74 75 13 13) et un médecin peuvent vous aider, y compris pour les dettes. » (numéro à vérifier).
4. `repere_jeu` : ajouter « Des périodes d'humeur très haute peuvent aussi faire jouer davantage ; dites-le au professionnel. »

Motif : le triptyque de 6C50 et 6C51 est fidèle. « Sur les écrans » englobe le travail et la CIM-11 ne reconnaît pas de trouble « écrans ». L'épisode maniaque est une exclusion de 6C50.

### alcool
Statut : retenu
Corrections :
1. `alc_controle` : « Vous est-il arrivé de boire plus, ou plus longtemps, que prévu, ou d'avoir du mal à arrêter ? » → « Vous est-il souvent arrivé de boire plus, ou plus longtemps, que prévu, ou d'avoir du mal à arrêter ? »
2. `alc_priorite` : « L'alcool est-il passé avant… » → « L'alcool est-il souvent passé avant… » (reste inchangé).
3. `alc_dommages` : « L'alcool a-t-il abîmé votre santé, ou conduit à… » → « L'alcool a-t-il abîmé votre santé physique ou mentale, ou conduit à… » (reste inchangé).
4. `repere_alcool_dommages` : ajouter la condition « `repere_alcool_dependance` négatif ».
5. `repere_alcool_dependance`, positif : « Alcool Info Service peut aussi vous aider. » → « Alcool Info Service (0 980 980 930) peut aussi vous aider. » (numéro à vérifier).
6. `repere_alcool_dependance`, négatif : ajouter « Ce repère ne dit rien des quantités. Repères français : pas plus de 10 verres par semaine, pas plus de 2 par jour, des jours sans alcool ; aucun pendant la grossesse. »
7. Vérifier que l'entrée « positive » (`alc_usage` = oui) n'affiche jamais la phrase fixe positive.

Motif : la règle « 2 sur 3 » est fidèle à 6C40.2, mais « Vous est-il arrivé » valide un épisode unique alors que la CIM-11 décrit un mode d'usage. 6C40.1 inclut la santé mentale et ne se pose pas en cas de dépendance. Un négatif sans repère de quantité rassure à tort un gros buveur.

### cannabis
Statut : retenu
Corrections :
1. `can_controle` : « Vous est-il arrivé de consommer plus… » → « Vous est-il souvent arrivé de consommer plus… »
2. `can_priorite` : « Le cannabis est-il passé avant… » → « Le cannabis est-il souvent passé avant… »
3. `can_corps` : « …ou ressenti un manque (tremblements, sueurs, anxiété) en arrêtant ? » → « …ou ressenti un manque (irritabilité, anxiété, sommeil difficile, moins d'appétit) en arrêtant ? »
4. `can_dommages` : « …abîmé votre santé, ou conduit à… » → « …abîmé votre santé physique ou mentale, ou conduit à… »
5. `repere_cannabis_dommages` : ajouter la condition « `repere_cannabis_dependance` négatif ».
6. `repere_cannabis_dependance`, positif : « Drogues Info Service et les consultations jeunes consommateurs peuvent vous aider. » → « Drogues Info Service (0 800 23 13 13), un médecin ou un centre d'addictologie (CSAPA) peuvent vous aider. » (numéro à vérifier).

Motif : les signes de sevrage du cannabis ont été copiés de l'alcool, alors que le tableau typique est irritabilité, anxiété, troubles du sommeil et de l'appétit. Les consultations jeunes consommateurs visent surtout les moins de 25 ans, sur une page pour adultes.

### autres_substances
Statut : retenu
Corrections :
1. En-tête : « 6C4G.1, 6C4G.2 » → « 6C42 à 6C4G selon le produit ; tabac : 6C4A » (6C4G ne couvre que les substances inconnues ou non précisées ; à vérifier).
2. `drog_usage` : « …avez-vous pris d'autres drogues, ou des médicaments sans ordonnance ou plus que prescrit, pour leurs effets ? » → « …avez-vous pris d'autres drogues, ou des médicaments (calmants, somnifères, antidouleurs forts) sans prescription ou plus que prescrit ? »
3. `drog_corps` : « Avez-vous eu besoin de en prendre davantage… » → « Avez-vous eu besoin d'en prendre davantage… »
4. `drog_controle` : « Vous est-il arrivé… » → « Vous est-il souvent arrivé… » ; `drog_priorite` et `drog_dommages` : « Ce produit a-t-il / est-il… » → « Ce ou ces produits ont-ils / sont-ils souvent… », avec « santé physique ou mentale » dans `drog_dommages`.
5. `repere_autres_substances_dommages` : ajouter la condition « `repere_autres_substances_dependance` négatif ».
6. `repere_tabac` : ne pas afficher la phrase fixe positive ; le texte actuel suffit.
7. `repere_autres_substances_dependance`, positif : « Drogues Info Service peut aussi vous aider. » → « Drogues Info Service (0 800 23 13 13) peut aussi vous aider. »

Motif : « médicaments sans ordonnance » inclut le paracétamol en vente libre. Le texte contient une faute de français. La mise en garde sur l'arrêt brutal est juste et doit rester.

### dissociation
Statut : retenu
Corrections :
1. `dissoc_realite` : « Pendant ces moments, saviez-vous que c'était une impression ? » → « Pendant ces moments, saviez-vous que c'était une impression, et que vous et le monde étiez bien réels ? »
2. `repere_detachement`, positif : « Ces expériences sont fréquentes après un stress intense ; un professionnel peut en parler avec vous. » → « Ces expériences sont fréquentes après un stress intense, une crise d'angoisse, un manque de sommeil, du cannabis ou d'autres produits ; un professionnel peut en parler avec vous. »
3. `dissoc_memoire` : « Ces derniers mois, avez-vous oublié des moments importants de votre vie, bien plus que des oublis ordinaires ? » → « Ces derniers mois, avez-vous remarqué des trous de mémoire sur des événements importants de votre vie, bien au-delà des oublis ordinaires, sans alcool ni drogue ? »
4. `repere_memoire` : supprimer le sujet `ten_dissociation`. Positif : ajouter « Ne pas se souvenir de sa petite enfance est normal et ne veut pas dire qu'un événement a été oublié. »
5. `dissoc_corps` : « …une perte de force, de sensation, de vue ou des crises, sans cause trouvée par un médecin ? » → « …une perte de force, de sensation, de vue, ou des crises avec perte de contrôle du corps, sans cause trouvée par un médecin ? »
6. Sous `dissoc_corps` et dans `repere_corps`, ajouter : « Une perte soudaine de force, de sensation, de vue ou de parole est une urgence : appelez le 15. »

Motif : sur un site qui rapproche symptômes et enfance, croiser des « trous de mémoire » avec des vécus d'enfance peut induire la croyance en un traumatisme refoulé. C'est le risque de suggestion le plus sérieux du contenu. Les symptômes de 6B60 sont aussi ceux d'un AVC.

### perceptions
Statut : retenu
Corrections :
1. `orientation_perceptions`, positif : « Elles ont de nombreuses causes (manque de sommeil, stress intense, substances, médicaments, maladie) ; un médecin peut en parler avec vous sans jugement. » → « Elles ont de nombreuses causes (manque de sommeil, stress intense, deuil, substances, médicaments, maladie) ; un médecin peut en parler avec vous sans jugement. Si elles sont récentes, vous font peur ou vous poussent à agir, consultez rapidement ; en urgence, le 15. »
2. `percep_idees` : conserver ; ajouter sous la question « Si une personne vous surveille ou vous menace réellement, voyez plutôt le module « Liens proches ». »

Motif : le choix d'une simple orientation sans sujet croisé est le bon, et je confirme le retrait de `icd_6a20` et `ten_traits_schizotypiques`. Il manque le deuil (entendre un défunt est courant et non pathologique) et un critère d'urgence.

### personnalite
Statut : retenu
Corrections :
1. `repere_personnalite` : ajouter la condition `personnalite_gene in [nettement]`.
2. `repere_personnalite`, positif : après « …d'être en relation. », ajouter « D'autres raisons peuvent expliquer ces difficultés : humeur, anxiété, événements de vie, entourage ou contexte difficile. »
3. `repere_personnalite`, négatif : « Vos réponses ne correspondent pas à ce repère. » → « Vos réponses ne correspondent pas à ce repère de difficultés durables. »

Motif : 6D10 exige une détresse ou une gêne importante et l'absence d'autre explication (autre trouble, contexte social). Sans ces garde-fous, le repère le plus stigmatisant du site se déclenche sur deux « Oui ». L'absence de sujet croisé est justifiée.

### sexualite
Statut : retenu
Corrections :
1. `repere_difficulte_sexuelle`, positif : « …des causes physiques sont fréquentes. » → « …des causes physiques ou des médicaments (dont certains antidépresseurs) sont fréquents. N'arrêtez pas un traitement sans avis médical. »
2. `repere_risques` : supprimer le sujet `ten_risques_sexuels` ; ne pas afficher la phrase fixe positive.
3. `repere_risques`, positif : « Un dépistage gratuit est possible dans un CeGIDD ou chez votre médecin ; un préservatif ou la PrEP protègent des infections. » → « Un dépistage gratuit est possible dans un CeGIDD ou chez votre médecin. Le préservatif protège du VIH et de la plupart des infections ; la PrEP protège du VIH seulement. Après un risque de moins de 48 heures, un traitement d'urgence existe aux urgences. »
4. `repere_sexualite_controle` : ajouter la condition `sexualite_gene in [un_peu, nettement]`.
5. En tête du module, ajouter : « Si un acte sexuel vous a été imposé, vous pouvez en parler au 3919 ou à France Victimes (116 006). »

Motif : la PrEP ne protège pas des autres infections, c'est une erreur factuelle de santé publique. Un seul rapport non protégé en douze mois ne fonde pas un sujet « risques sexuels » rapproché de l'enfance. La réserve sur le jugement moral dans 6C72 est correcte.

---

### securite
Statut : retenu
Corrections :
1. Écran bloquant : « Ce site ne peut pas intervenir en urgence. Si vous êtes en danger maintenant, appelez le 15 ou le 112. » → « Ce site ne peut pas intervenir en urgence. Le 3114 répond à toute heure, gratuitement : des professionnels vous écoutent. Si vous êtes en danger maintenant, appelez le 15 ou le 112. »
2. Écran bloquant, boutons : ordre → « Parler à Ambre », « Arrêter », « Continuer le questionnaire ». Les numéros restent visibles après « Arrêter » et tout au long du questionnaire après « Continuer ».
3. « Parler à Ambre » : si Ambre est un agent automatique, renommer le bouton sur cet écran « Écrire à Ambre (assistant automatique, pas un service d'urgence) ». Si Ambre ne suit pas un protocole de crise, retirer ce bouton de l'écran bloquant.
4. Ajouter la règle : tout item 9 du PHQ-9 ≥ 1 déclenche le même écran, quel que soit le score total.
5. Ajouter les règles `securite_automutilation` et `securite_violences` (textes dans les modules regulation et relations).
6. Ressources « violences » : ajouter « 116 006 — France Victimes, pour toute personne victime, gratuit » (à vérifier). Pour le 3919, préciser « 24 h/24 et 7 j/7 ». Ajouter sur ces écrans un bouton « Quitter vite cette page ».
7. Fil Santé Jeunes : « écoute gratuite et anonyme pour les jeunes » → « écoute gratuite et anonyme pour les 12-25 ans, tous les jours de 9 h à 23 h » (horaires à vérifier).
8. Phrase fixe positive : la réserver aux règles `instrument` et `repere`. Pour les règles `repere_site`, créer `site_note` : « Ce repère décrit une façon de fonctionner, pas un trouble. Si elle vous pèse, en parler à un professionnel peut aider. »
9. Ne pas afficher la phrase fixe positive après : `repere_restriction`, `orientation_perceptions`, `orientation_realite`, `repere_tabac`, `repere_risques`, PHQ-9 ≥ 15, GAD-7 ≥ 15.
10. `gene_absente` : « Vous n'indiquez pas de gêne dans votre vie. » → « Vous n'indiquez pas de gêne dans votre vie : sans gêne ni souffrance, ces éléments ne sont en général pas considérés comme un trouble. » Exceptions où le texte actuel reste : alimentation, bipolaire, perceptions, alcool, cannabis, autres_substances, jeu_ecrans.
11. `crossing_childhood_only` : « Lié à vos vécus, sans difficulté signalée aujourd'hui » → « Souvent associé, dans la recherche, à des vécus comme les vôtres ; non signalé chez vous aujourd'hui ». Afficher cette liste repliée par défaut.
12. `what_next` : « Vous pouvez en parler à votre médecin traitant ou à un psychologue, en parler avec Ambre, ou refaire ce module plus tard. » → « Vous pouvez en parler à votre médecin traitant, à un psychologue (dispositif Mon soutien psy) ou à un centre médico-psychologique (CMP, gratuit), en parler avec Ambre, ou refaire ce module plus tard. »

Motif : l'écran de crise n'énonce pas le 3114 dans son texte et propose de continuer avant de proposer de l'aide. Deux questions de sécurité et l'item 9 du PHQ-9 n'ont aucune suite décrite. Lister des difficultés que la personne n'a pas, sous le titre « Lié à vos vécus », affirme un lien et peut induire des symptômes.

### instruments
Statut : retenu
Corrections :
1. `phq9_repere`, `gad7_repere`, `phq15_repere`, positifs : « sont au-dessus du repère publié de 10 » → « atteignent ou dépassent le repère publié de 10 » (la règle est ≥ 10).
2. PHQ-2 et GAD-2, seuil d'entrée ≥ 2 : acceptable car plus sensible et non bloquant, mais le seuil publié est 3 (Kroenke 2003 ; Kroenke 2007). L'indiquer dans la page méthode et ne jamais écrire « repère publié » pour ce seuil.
3. PHQ-9 ≥ 15 et GAD-7 ≥ 15 : remplacer la phrase fixe positive par « À ce niveau de score, un rendez-vous avec un médecin ou un psychologue dans les prochains jours est conseillé. »
4. PHQ-15 : « score {score} sur 30 » → « score {score} sur {max} », avec max = 28 quand l'item sur les règles ne s'applique pas.
5. ITQ, règle interne : remplacer « TSPT-C = TSPT + 3 grappes POS + atteinte POS » par « TSPT-C = TSPT + 3 grappes de perturbation de l'organisation de soi (régulation des émotions, image négative de soi, relations) + atteinte fonctionnelle liée à ces grappes ». Une grappe est positive si au moins un de ses deux items est ≥ 2.
6. ITQ : n'afficher aucun résultat si `exposition_evenement` ≠ oui.
7. `itq_tspt` et `itq_tspt_c`, positifs : après « C'est un repère de questionnaire », ajouter « , pas un diagnostic ».
8. Référence « Peraud et al. 2022 » : vérifier la citation et la licence de la version française avant publication.

Motif : les paliers PHQ-9 (5/10/15/20), GAD-7 (5/10/15) et PHQ-15 (5/10/15) sont conformes aux publications. Écrire « au-dessus de 10 » pour un score de 10 est inexact. À un score élevé, la probabilité d'un trouble après entretien est forte, donc la phrase fixe rassure à tort.

### Remarques générales
- **Gêne et seuil clinique.** La CIM-11 exige presque partout une détresse ou une gêne significative. Seul `repere_toc` l'intègre. La correction 10 du bloc securite et les conditions ajoutées (deuil, cauchemars, personnalité, sexualité) réduisent le sur-repérage, sans le supprimer.
- **Exclusions non testées.** Aucun module ne vérifie qu'un autre trouble, une substance ou une maladie explique mieux les réponses. Les textes ajoutés compensent en partie. Le risque résiduel est celui de plusieurs repères positifs simultanés lus comme autant de « troubles ». Un message unique serait utile quand trois repères ou plus sont positifs : « Plusieurs repères positifs décrivent souvent une seule et même difficulté. »
- **Cartographie des sujets.** Je n'ai pas accès aux libellés affichés ni aux données derrière chaque lien. Mes avis sur `att_*`, `pat_impostor`, `icd_6b60`, `ten_difficultes_sexuelles` et `icd_6b81` dépendent de ces libellés. Le croisement ne doit afficher que des formulations descriptives, jamais un nom de diagnostic.
- **Suggestion par le croisement.** Même avec la note « ne montre pas une cause », la juxtaposition invite à une lecture causale. Les deux points les plus sensibles sont la mémoire (corrigé) et la sexualité (à surveiller).
- **Alcool.** L'AUDIT-C (OMS, libre) serait un meilleur point d'entrée que la question binaire `alc_usage`, positive chez presque tous les adultes.
- **Absences.** Il n'y a aucun repère pour le TDAH, l'autisme ou les phobies spécifiques. L'impulsivité, les difficultés relationnelles et l'alexithymie peuvent en relever. À signaler dans la page méthode.
- **Réponses sensibles.** Les modules cannabis, autres substances, sexualité et violences recueillent des données de santé ou des faits illicites. Une phrase visible sur la confidentialité et la conservation est nécessaire avant ces modules ; une personne victime doit pouvoir effacer ses réponses.
- **Ambre.** Sa nature (humain ou agent automatique) et son protocole face à une idée suicidaire ne figurent pas dans le contenu relu. C'est le principal risque non évalué.
- **Vérifications avant publication.** Numéros et horaires : 116 006, 0 980 980 930, 0 800 23 13 13, 09 74 75 13 13, 09 69 325 900, Fil Santé Jeunes. Codes CIM-11 : 7B01.2 et 6C4G. Durée exacte de la dérogation « usage continu » dans 6C40.2.
