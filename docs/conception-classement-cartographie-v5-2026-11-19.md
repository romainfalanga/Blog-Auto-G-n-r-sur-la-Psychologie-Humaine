# Cartographie v5 — conception du classement et de « Ce qui en ressort »

Analyse en lecture seule (aucun fichier existant modifié, aucun commit). Rédigée le 2026-11-17 pour le lot daté 2026-11-19.
Scripts de mesure et captures : hors dépôt, dans le répertoire de travail de la session (`scratchpad/carto-v5/`).

## 0. État vérifié

- Données publiées `data/cartographie.json` : 59 questions, 109 sujets (24 `contextual_only`), 496 liens (364 associations empiriques, 88 modèles cliniques, 26 correspondances de définition, 18 contextes de classification).
- 43 questions sur 59 ont au moins une association empirique ; les 16 autres (presque tous les « ressentis », exil, migration, secrets, emprise religieuse) n'ont que des modèles ou définitions, ou rien.
- Fréquence des sujets (nombre de questions reliées par une association empirique) : dépression 32, anxiété 26, conduites suicidaires 14 (contextuel), externalisation 13, substances 13, souffrance dans l'enfance 12, psychoses 9, difficultés scolaires 9 ; 27 sujets n'ont qu'une question reliée, 14 en ont deux.
- Ordre actuel (`compute` et `computeCartography`, identiques) : contextuels en dernier, puis `associated_count`, `evidence_groups`, force maximale, identifiant. Les retours « Je me reconnais » / « Pas moi » et l'intensité « Souvent » n'interviennent pas dans l'ordre (la carte écartée est seulement grisée, `is-dismissed`).
- Règle `contextual_only` (générateur `scripts/build_cartographie.py` l. 255-257) : sujet « jeune enfant », catégorie `health_outcome` ou `categorical_personality_diagnosis`, ou sujet dont tous les liens sont des contextes de classification ; s'y ajoutent les marquages portés directement par les données sources. Les 24 sujets : quatre sujets suicidaires (`note: suicide`), santé (diabète, cardiovasculaire, migraine, fibromyalgie), dépression périnatale, symptômes prémenstruels, récompense immédiate chez les jeunes, parentification (motif), traits narcissiques (×2) et antisociaux, 6B20, 6B41, 6B42, 6B43, 6B44 et 6B45 (jeune enfant), 6B64, 6D11.5, désorganisation de l'attachement (jeune enfant).

Captures de l'état actuel (profil « violences », 5 cases) : 67 cartes de même poids visuel, dont 16 contextuelles ; à 390 px le volet « Ce qui en ressort » mesure 12 278 px ; aucune phrase n'explique l'ordre ; chaque carte répète « Je me reconnais / Pas moi » ; l'en-tête « Tendance » n'apporte rien à la lecture. En moyenne (500 profils aléatoires) : 24,7 cartes pour 3 cases (4,8 contextuelles), 54 pour 10 cases (10,4 contextuelles), 67,6 pour 15 cases (13,4 contextuelles).

## 1. Retrait des cartes contextuelles

### 1.1 Page
- `compute` (navigateur) et `computeCartography` (Worker) excluent les sujets `contextual_only` de `results` (même filtre que `retired`). Conséquences directes : plus de légende « Repère de contexte : Ambre ne vous le suggère pas… » (`cartographie-ui.js`, `card`), plus de branche `Repère de contexte` dans `cardMeta`, plus de note `NOTE_TEXT.jeune_enfant` à afficher (tous les sujets `jeune_enfant` sont contextuels).
- Fenêtre « Liens étudiés » d'une question (`openFactor`) : elle liste aujourd'hui aussi les 80 liens qui mènent à un sujet contextuel (66 associations empiriques, dont les conduites suicidaires, 14 contextes CIM). Recommandation : les masquer aussi, et calculer le compteur du bouton « Liens étudiés (n) » sans eux ; sinon la personne retrouve dans la fenêtre des sujets retirés de la liste. Aucune question ne perd tous ses liens (vérifié) : le message « Aucune association suffisamment documentée… » ne change pour aucune question.
- Données inchangées : les sujets restent dans `data/cartographie.json` et dans le module Worker (identifiants historiques, validation des retours et notes déjà enregistrés).

### 1.2 Retours et notes déjà enregistrés sur un sujet contextuel
Même règle que pour les sujets `retired` (test navigateur « stored retired feedback and notes survive editing without moving to a new card ») : `state.feedback` et `state.notes.outcomes` les conservent et les renvoient au PUT ; la validation Worker les accepte toujours. Limite à signaler au propriétaire : une note ancienne sur un sujet désormais masqué n'est plus visible ni modifiable sur la page, alors qu'Ambre la lit. Option si le propriétaire le souhaite : afficher ces seules notes dans la carte de la question concernée — non recommandé ici car cela réintroduit une zone de contexte.

### 1.3 Ambre — proposition
| Élément du résumé | Aujourd'hui | Proposé |
| --- | --- | --- |
| `display_order`, `display_count` | contextuels en dernier, comptés | exclus (identiques à la page) |
| `contextual` (toutes les cartes contextuelles du profil, sans plafond) | transmis | **supprimé** : la personne ne les voit plus ; Ambre ne doit pas évoquer un sujet que la page ne montre pas. Libère du budget (`chat-backend.mjs` l. 418 et 686 tronquaient d'abord cette liste). |
| `recognized`, `dismissed`, notes de la personne sur un sujet contextuel | transmis avec `contextual_only` | **conservés** (ce sont ses propres réponses et ses mots), toujours marqués `contextual_only` |
| `suicide_recognized` | dépend d'un retour « oui » sur une carte suicidaire | conservé pour les retours anciens ; aucun nouveau retour possible. La sécurité d'Ambre repose sur la conversation (`crisisSignal`, consignes de sécurité), pas sur la carte : rien à retirer de `CHILDHOOD_CARTOGRAPHY_GUIDANCE` § Sécurité. |
| `cartographyEvidence` | contextuel inclus seulement si reconnu | inchangé |
| `cartographyHighlights` | exclut déjà les contextuels | inchangé |
| `ambre-engine.mjs` l. 165 (itère `contextual`) | — | tolère une liste absente ; garder `contextual: []` une version pour les clients en cache, puis retirer |

Consigne à mettre à jour (`psychology-framework.mjs`, `CHILDHOOD_CARTOGRAPHY_GUIDANCE`) : `display_order` = ordre de pertinence affiché (voir § 2), sections `reconnu` puis `principal` puis `autre`, jamais un risque ; retirer toute mention de repères contextuels affichés. Le prompt d'`ambre-engine.mjs` l. 493 peut garder sa phrase sur `contextual_only` (retours anciens).

## 2. Classement par pertinence et spécificité

### 2.1 Ce que le score dit — et ne dit pas
Le classement ne peut pas estimer une probabilité individuelle : les forces viennent de méta-analyses hétérogènes (OR non comparables, populations différentes), sans taux de base ni covariance entre vécus. Il répond à une autre question, défendable : **« parmi les sujets que la recherche relie à vos réponses, lesquels leur sont le plus propres, et avec quelle solidité ? »**. Un sujet relié à presque toutes les adversités (dépression : 32 questions sur 43) apprend peu sur *vos* réponses en particulier ; un sujet relié à 2 questions dont la vôtre en apprend davantage. C'est le raisonnement TF-IDF (spécificité d'un terme rare) appliqué aux liens. Conséquence assumée : dépression et anxiété, les issues **les plus fréquentes en valeur absolue**, ne sont plus systématiquement en tête ; elles restent visibles (§ 3.3, ligne « fréquents »).

### 2.2 Variantes simulées
Script hors dépôt `scratchpad/carto-v5/simulate.cjs` : 2 000 profils aléatoires par taille (3, 6, 10, 15 cases) et 2 000 profils de taille mixte 1–20 ; 30 % des réponses à « Souvent » ; contextuels exclus partout. Variantes (B à L : meilleur lien par famille `overlap_group` ; H à L : sujets à association empirique avant les sujets de modèle seul) :
- **A** actuel (`associated_count`, familles, force max) ; **B** somme des forces 3/2/1 ;
- **C** B × spécificité `ln(1 + N/df)` (N = 43 questions ayant une association empirique, df = questions reliées au sujet) ;
- **D** « lift » lissé : (k + 1) / (n·df/N + 1) ; **E** enrichissement hypergéométrique −log10 P(X ≥ k) ;
- **F/H** C + modèles et définitions à 0,5 + « Souvent » ×1,5 (H : empiriques d'abord) ; **G** F avec spécificité 1/√df ;
- **I** H avec spécificité amortie √ln ; **J** H avec « limitée » à 0,5 et modèles 0,25 ; **K** H sans « Souvent » ; **L** H avec modèles et définitions à **0,25**.

Mesures : part des profils où dépression ou anxiété est 1re / dans le top 3 / dans le top 5 ; sujets distincts apparaissant dans un top 5 ; Jaccard moyen entre top 5 de deux profils différents (plus bas = plus personnalisé) ; stabilité = Jaccard du top 5 après ajout d'une case au hasard, puis après passage d'une réponse « Oui » à « Souvent » ; « rang 1 faible » = sujet de tête reposant sur un seul lien « limitée » (ou aucun lien mesuré).

Profils mixtes 1–20 cases :

| Variante | Dép./anx. 1er | top 3 | top 5 | Sujets distincts top 5 | Jaccard entre profils | Stabilité +1 case | Stabilité « Souvent » | Rang 1 faible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A actuel | 87 % | 95 % | 96 % | 52 | 0,38 | 0,84 | 1,00 | 4 % |
| B somme | 87 % | 95 % | 96 % | 52 | 0,37 | 0,84 | 1,00 | 4 % |
| C TF-IDF | 28 % | 58 % | 71 % | 64 | 0,17 | 0,78 | 1,00 | 7 % |
| D lift | 1 % | 6 % | 13 % | 78 | 0,06 | 0,78 | 1,00 | **25 %** |
| E hypergéom. | 7 % | 17 % | 25 % | 78 | 0,07 | 0,75 | 1,00 | **21 %** |
| G 1/√df | 21 % | 46 % | 62 % | 68 | 0,13 | 0,79 | 0,88 | 21 % |
| H | 23 % | 53 % | 70 % | 66 | 0,15 | 0,80 | 0,91 | 9 % |
| I amortie | 65 % | 83 % | 89 % | 63 | 0,27 | 0,83 | 0,93 | 4 % |
| J limitée ½ | 24 % | 54 % | 71 % | 62 | 0,17 | 0,82 | 0,94 | 4 % |
| K sans « Souvent » | 26 % | 55 % | 69 % | 64 | 0,15 | 0,77 | 1,00 | 10 % |
| **L recommandée** | **26 %** | **55 %** | **71 %** | **65** | **0,16** | **0,81** | **0,92** | **8 %** |

Par taille (L) : 3 cases 7 % / 32 % / 56 % ; 6 cases 11 / 42 / 63 ; 10 cases 20 / 58 / 75 ; 15 cases 35 / 69 / 84 (A : 73/92/93 → 99/100/100). Plus on coche, plus dépression et anxiété remontent légitimement : beaucoup de vécus convergent alors vraiment vers elles.

Profils typés, top 5 (A actuel → L recommandée) :

| Profil (cases ; « Souvent ») | A actuel | L recommandée | Rang dép. / anx. (L) |
| --- | --- | --- | --- |
| Parentification (parentification², parent_trouble, ressenti_responsable², debrouiller_seul, enfant_sage) | Anxiété, Dépression, Codépendance, Psychoses, Bipolaires | Anxiété de séparation, Rôle de sauveur, Codépendance, Bipolaires, Psychoses | 10 / 8 sur 19 |
| Violences (abus_physique², abus_emotionnel², temoin, ressenti_peur, pas_protege) | Violences de couple, Sommeil, Dépression, Anxiété, Attachement anxieux | Violences de couple, Attachement anxieux, Attachement évitant, Colère, Régulation émotionnelle | 30 / 32 sur 51 |
| Négligence (negligence_emotionnelle², physique, pas_compter, solitude, debrouiller_seul) | Dépression, Bipolaires, Alexithymie, Scolarité, Estime de soi | Honte de soi, Névrosisme, Bipolaires, Alexithymie, Carence affective | 18 / 32 sur 36 |
| Harcèlement (harcelement², moqueries_corps, isolement, se_sentir_different, ressenti_honte) | Dépression, Anxiété, Isolement, TCA, Scolarité | Agoraphobie, Sommeil, Scolarité, Isolement, Anxiété sociale | 13 / 11 sur 17 |
| Deuil (deces_proche², ressenti_abandon, ressenti_solitude) | Dépression, Anxiété, Psychoses, Substances, Attachement anxieux | Dépression, Psychoses, Anxiété, Substances, Schéma d'abandon | 1 / 3 sur 11 |
| Séparation (separation_conflit², triangulation, conflits, recomposée, ressenti_abandon) | Dépression, Anxiété, Externalisation, Souffrance enfance, Substances | Souffrance enfance, Substances, Culpabilité, Externalisation, Dépression | 5 / 7 sur 14 |
| Perfectionnisme (critique², amour_conditionnel, pression_scolaire², comparaison, ressenti_perfection²) | Dépression, Perfectionnisme, Anxiété, Externalisation, Valeur liée à la réussite | Perfectionnisme coûteux, Dépression, Anxiété, Autocritique, Valeur liée à la réussite | 2 / 3 sur 17 |
| Tout coché (58 cases) | Dépression, Anxiété, Externalisation, Substances, Souffrance enfance | Dépression, Anxiété, Estime de soi, Bipolaires, Scolarité | 1 / 2 sur 85 |

(Rangs dép./anx. mesurés sur H ; L identique à ± 1 rang sur ces profils.)

### 2.3 Lecture
- **A et B** ne changent rien : la somme des forces reproduit le comptage.
- **D (lift) et E (hypergéométrique)** maximisent la diversité mais placent en tête, pour 21–50 % des profils, un sujet tenu par un seul lien « limitée » ; D sort « Tabagisme », « Cannabis », « Boulimie » en tête pour le profil violences. Indéfendable face à la charte (force affichée faible en 1re position).
- **I** (spécificité amortie) laisse dépression/anxiété en tête dans 65 % des cas : ne répond pas à la demande.
- **C/H/L** divisent par 3 à 4 la présence de dépression/anxiété en tête, doublent la personnalisation (Jaccard 0,38 → 0,16), gardent une stabilité proche (0,84 → 0,81 à l'ajout d'une case) et limitent les têtes faibles à 8 %. L diffère de H par un poids plus faible des modèles cliniques (0,25 au lieu de 0,5), ce qui réduit les têtes faibles (9 → 8 %, 15 cases 2 → 1 %) sans perdre « Rôle de sauveur » ni « Perfectionnisme coûteux » dans les profils où ils sont attendus.
- **J** (limitée ½) réduit encore les têtes faibles (4 %) mais écrase l'écart entre « limitée » et modèle clinique (0,5 contre 0,25) et fait disparaître l'agoraphobie du profil harcèlement (lien relu en v4.1) : à garder comme repli si le propriétaire veut zéro tête « limitée ».
- « Souvent » ×1,5 (L vs K) : modifie le top 5 dans 8 % des cas (stabilité 0,92), effet visible mais modéré ; il ne change ni force affichée, ni niveau, ni section de preuve.
- Points d'attention à faire valider : pour « violences », dépression (lien « forte ») descend au 30e rang ; pour « parentification » et « deuil », les psychoses/bipolaires montent (liens `parent_trouble` et `deces_proche`, modérés, reflétant aussi l'hérédité). C'est l'effet voulu de la spécificité ; la ligne « fréquents » (§ 3.3) et le niveau de preuve sur chaque carte évitent qu'il soit lu comme un risque.

### 2.4 Variante recommandée (L) — pseudo-code commun
À implémenter **à l'identique** dans `static/js/cartographie.js` (`compute`) et `journal/src/cartographie.mjs` (`computeCartography`). Tout en entiers : aucun écart d'arrondi entre moteurs JavaScript (navigateur et Worker), donc parité testable par égalité stricte. Implémentation de référence exécutable : `scratchpad/carto-v5/ranking-v5.js` (1 976/2 000 ordres identiques à la variante L flottante ; les 24 écarts sont des quasi-égalités départagées autrement).

```
# Construction (scripts/build_cartographie.py, puis copie dans le module Worker)
EMPIRIQUE(l) = relation(l) == "empirical_association" et non l.definitional et l.strength ∈ {forte, moderee, limitee}
N = nombre de questions ayant au moins un lien EMPIRIQUE                      # 43 aujourd'hui
pour chaque sujet o :
    df(o) = max(1, nombre de questions distinctes reliées à o par un lien EMPIRIQUE)
    o.specificity = arrondi(1000 × ln(1 + N / df(o)))                          # entier ; dépression 852, df=1 : 3784
    o.frequent    = (4 × df(o) ≥ N)                                            # dépression, anxiété, externalisation, substances, souffrance enfance (+ suicidaire, contextuel)

# Calcul (page et Worker)
POINTS = {forte: 12, moderee: 8, limitee: 4}; POINTS_AUTRE = 1                # modèle, définition, contexte CIM = ¼ d'un lien limité
FACTEUR = {1: 2, 2: 3}                                                          # « Oui » ×2, « Souvent » ×3 (= ×1,5)
SECTION_PRINCIPALE = 5

entrée : answers {question: 0|1|2}, feedback {sujet: "oui"|"non"} (facultatif)
niveau = {q: v pour q, v dans answers si q connu et v ∈ {1, 2}}
pour chaque lien l avec niveau[l.factor] défini :
    o = sujet(l.outcome) ; ignorer si absent, o.retired ou o.contextual_only
    ajouter l à liens[o]
pour chaque sujet o de liens :
    familles = {} ; familles_empiriques = {} ; force_max = 0
    pour l dans liens[o] :
        p = (EMPIRIQUE(l) ? POINTS[l.strength] : POINTS_AUTRE) × FACTEUR[niveau[l.factor]]
        f = famille(l.factor) = overlap_group ou identifiant de la question
        familles[f] = max(familles[f] ou 0, p)
        si EMPIRIQUE(l) : familles_empiriques[f] = vrai ; force_max = max(force_max, POINTS[l.strength])
    relevance(o) = somme(familles) × o.specificity                              # entier, jamais affiché
    has_empirical(o) = force_max > 0
trier par : has_empirical décroissant, relevance décroissante, nombre(familles_empiriques) décroissant,
            force_max décroissante, identifiant croissant (ordre binaire, pas localeCompare)
sections, dans l'ordre trié : feedback "oui" → "reconnu" ; "non" → "ecarte" ;
            sinon les SECTION_PRINCIPALE premiers → "principal", les suivants → "autre"
ordre final = reconnu, principal, autre, ecarte (tri stable : le rang est conservé dans chaque section)
sortie par sujet : outcome, links (empiriques d'abord par force, puis les autres), associated_factors,
            associated_count, evidence_groups (inchangés), tier (inchangé, pour Ambre), relevance, section, frequent
```

Remarques : `tier` (convergence / piste / théorique) reste calculé comme aujourd'hui pour les listes plafonnées d'Ambre, mais ne sert plus à l'ordre. Le score n'est jamais affiché ni transmis en chiffre à Ambre (seulement l'ordre et la section). L'identifiant final départage en ordre binaire pour éviter toute différence de `localeCompare` entre moteurs.

Texte de `interpretation.ranking` proposé : « Liste unique, même ordre pour la page et pour Ambre. Chaque vécu coché relié à un sujet compte selon le niveau de preuve du lien (solide, modéré, limité ; modèles cliniques et définitions comptent peu), un peu plus si vous avez répondu « Souvent », une seule fois par famille de vécus proches ; ce total est pondéré par la spécificité du sujet, moindre quand la recherche le relie à presque tous les vécus difficiles. Les sujets dans lesquels vous vous reconnaissez viennent d'abord, ceux que vous avez écartés à la fin. Ce n'est ni une probabilité ni un risque personnel. »

