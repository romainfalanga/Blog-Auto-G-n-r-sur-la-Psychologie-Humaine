# Cartographie — lot 4 : personnalité, traits et défenses

Recherche réalisée le 16 novembre 2026 UTC pour le chantier du 17 novembre. Fichiers produits exclusivement : ce rapport et `editorial/cartographie/liens-personnalite-traits-defenses-2026-11-17.json`. Aucune génération, publication, modification d’un fragment existant ou opération Git d’écriture.

## Résultat et règle d’admission

Le fragment retient **4 nouveaux sujets, 12 liens actifs, 10 nouvelles références documentaires, 83 décisions hors ajout et 4 corrections proposées**. Il conserve séparément **6 sujets et 6 liens proposés**, ignorés par le générateur. Aucun nouveau diagnostic catégoriel DSM n’est activé.

Les sujets actifs sont le névrosisme, les dimensions narcissiques vulnérable et grandiose, et les traits de personnalité antisociale mesurés dans une cohorte. Le douzième lien enrichit le motif borderline existant pour la négligence physique. Les trois dimensions sensibles (narcissisme et traits antisociaux) sont `contextual_only:true` ; tous les diagnostics proposés le sont aussi. Nouvelle catégorie à intégrer : `personality_trait`, libellé **« Trait de personnalité (dimension) »**.

La liste stricte du §0 prime sur la simple grille de force : une étude transversale isolée n’admet pas un lien actif. Les quatre paires initialement documentées par Waxman seul ont donc été reportées dans `proposed_links`, leurs sujets dans `proposed_outcomes`, sur arbitrage du pilote. Une étude incluse dans une cohorte n’est pas présentée comme longitudinale lorsque son analyse est transversale. Les synthèses d’études transversales restent explicitement limitées ; aucune causalité personnelle n’est inférée.

## Lectures vérifiables

Les références complètes et le niveau de lecture figurent dans `sources`. Les pages ont été recherchées et ouvertes avec l’outil web. Pour plusieurs pages PMC et le PDF Rosenek, le rendu web a ensuite renvoyé un contrôle de navigateur ; les sections ont été effectivement lues via téléchargement HTTP direct et extraction du texte, sans écrire de copie dans le dépôt.

| Source | Lecture effective et décision |
| --- | --- |
| [OMS, CDDR 2024](https://www.who.int/publications/i/item/9789240077263) | Page officielle ouverte ; texte local fourni, pp. 553–564 : diagnostic général, sévérité, cinq domaines et motif borderline. Réutilisation de `oms_cddr`, sans redéclaration. |
| [APA, page publique des personnalités](https://www.psychiatry.org/patients-families/personality-disorders/what-are-personality-disorders) | Définition et liste des dix diagnostics, revue en novembre 2024. **Pas de lecture du manuel DSM-5-TR**. |
| [Johnson et al., 1999](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/205066) | Méthodes, résultats, tableaux 2–4 et commentaire. Cohorte communautaire de 639 personnes ; seulement 15 violences physiques documentées. La paire spécifique retenue porte sur les symptômes antisociaux, pas un diagnostic. |
| [Waxman et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3927226/) | Méthode, résultats, tableau 1, limites. Analyse transversale de NESARC, avec souvenirs d’enfance et diagnostics DSM-IV. Les ajustements changent certaines associations : preuve documentaire et propositions, aucune admission isolée. |
| [Solmi et al., 2021](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2021.679379/full) | Méthodes, résultats et tableaux 2–4. La synthèse soutient le lien négligence physique–borderline ; biais et hétérogénéité excluent une force maximale. Les données de dépendance demandent une vérification du construit primaire. |
| [Rosenek et al., 2026](https://centaur.reading.ac.uk/128410/4/1-s2.0-S0272735826000097-main.pdf) | PDF publié, méthodes 2.1–2.3 et résultats 3.1–3.5, extraits lus avec `pdftotext`. Méta-analyse de 136 études ; cinq sous-types de maltraitance retenus. Effets modestes ou faibles, hétérogénéité élevée, corrections de biais pour plusieurs sous-groupes. |
| [Gao et al., 2024, en ligne 2023](https://pure.uva.nl/ws/files/150588946/gao-et-al-2023-the-association-between-child-maltreatment-and-pathological-narcissism-a-three-level-meta-analytic-review.pdf) | Méthodes, résultats, tableaux 2–4 et limites. Toutes les études sont transversales, surtout communautaires ; effets faibles. Aucun lien grandiosité–violences psychologiques, ni scission des négligences agrégées. Niveau `limitee`, contexte uniquement. |
| [Di Giuseppe et Perry, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8555762/) | Hiérarchie et méthode DMRS-Q ; titres des tableaux 2–8 et items sélectionnés. Source théorique/méthodologique, pas preuve qu’un vécu particulier produit une défense. |
| [Krueger et Hobbs, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7529724/) | Structure AMPD, littérature de validation et comparaison CIM. Revue des auteurs du domaine, pas consultation du manuel. |
| [Tracchegiani et al., 2025](https://link.springer.com/article/10.1007/s40653-025-00710-4) | Introduction, méthode, résultats et limites. Profils transversaux de 1 315 jeunes adultes : les expositions et défenses participent ensemble à la formation des groupes. Cela ne teste pas chaque paire vécu–défense. |
| [Consortium HiTOP](https://www.hitop-system.org/) | Page officielle du cadre, du consortium et du réseau clinique. Documentation d’une classification dimensionnelle, pas source étiologique pour créer un lien. |

Rosenek présente des incohérences de rédaction dans certains effectifs de sous-groupes, particulièrement la négligence physique, et exclut la rubrique « confounders » de son évaluation de qualité. Aucun effectif détaillé litigieux n’est repris dans les cartes. Les cinq liens restent au plus `moderee`, avec effet moyen et limites exprimés ; ils ne reçoivent aucun code CIM. Les échelles de névrosisme et d’émotionnalité négative sont diverses : cette carte n’est pas un résultat de test Big Five individuel.

La référence [Kane et Bornstein](https://onlinelibrary.wiley.com/doi/abs/10.1002/jclp.22550) a été ouverte au résumé : elle porte sur la dépendance interpersonnelle et le trouble dépendant. Sans lecture des mesures primaires, l’intitulé diagnostique adopté par l’umbrella ne suffit pas. Le lien et le diagnostic dépendants sont donc proposés seulement, sans source primaire prétendument lue intégralement.

## DSM : décision diagnostic par diagnostic

Les noms sont vérifiés sur la page publique APA. Les études anciennes utilisent DSM-IV ou des mesures symptomatiques : elles ne constituent pas une validation directe des critères DSM-5-TR.

| Diagnostic DSM | Décision du lot |
| --- | --- |
| Paranoïaque | Proposition contextuelle ; association transversale de négligence affective, sensible aux ajustements. |
| Schizoïde | Proposition contextuelle ; résultat transversal spécifique, sans admission isolée. |
| Schizotypique | `a_examiner` ; données diagnostiques chez Waxman. `ten_traits_schizotypiques` du lot 2 ne couvre ni ce diagnostic ni automatiquement 6A22. |
| Antisocial | Diagnostic proposé seulement. Carte active **de traits** fondée sur la mesure symptomatique de Johnson, contextuelle. |
| Borderline | Sujet existant `icd_6d11_5` enrichi ; rapprochement documentaire, pas identité de statut DSM/CIM. |
| Histrionique | `a_examiner` ; signal symptomatique ancien insuffisant pour une carte diagnostique. |
| Narcissique | Diagnostic proposé seulement ; les deux dimensions de recherche actives ne le remplacent pas. |
| Évitante | Proposition contextuelle ; pas d’admission de la seule analyse transversale. |
| Dépendante | Proposition seulement ; séparation préalable nécessaire entre dépendance dimensionnelle et diagnostic. |
| Obsessionnelle-compulsive | `a_examiner` ; aucune paire diagnostique suffisamment étayée dans les textes retenus. Distincte du TOC. |

Le registre maintient une décision explicite pour chacun des dix diagnostics, même lorsqu’aucune carte n’est créée.

## CIM, AMPD, Big Five et HiTOP

La CIM sépare le fonctionnement et la sévérité (6D10.0/.1/.2/.Z), les cinq domaines (6D11.0–.4) et le motif borderline (6D11.5). Ces domaines ne sont pas cinq diagnostics autonomes. QE50.7 n’est pas un trouble mental. Le tempérament et l’environnement ne se réduisent pas l’un à l’autre ; l’adversité n’est pas déterminante chez une personne. Aucun niveau de sévérité ne se déduit du nombre de vécus cochés.

L’AMPD articule fonctionnement et traits pathologiques, avec affectivité négative, détachement, antagonisme, désinhibition et psychoticisme. Il conserve six configurations nommées et une description par traits ; il n’est pas une simple copie des dix catégories. Le psychoticisme AMPD et l’anankastie CIM n’occupent pas la même place. Ces différences empêchent des alias automatiques entre instruments.

Pour les Big Five, seule la dimension névrosisme bénéficie ici d’une synthèse accessible lue au-delà du résumé. La méta-analyse [Credé et al., 2023](https://www.sciencedirect.com/science/article/pii/S0191886922003737), pertinente pour les cinq traits, a été repérée mais son texte n’a pas été obtenu : extraversion, agréabilité, conscienciosité et ouverture restent `a_examiner`, sans conclure à une absence de lien.

HiTOP demeure un cadre de regroupement des dimensions. Aucun « sujet HiTOP » n’est créé en doublon des troubles ou tendances existants. Les champs `out_of_scope` distinguent le cadre et ses principaux spectres.

## Défenses et dix décisions du catalogue de 161 repères

Les 30 mécanismes de la hiérarchie DMRS et le score global disposent d’entrées documentaires. Les entrées théoriques sont explicitement `clinical_model` / `theorique`, mais **aucun lien de défense n’est activé**. Déni, clivage et dévalorisation sont illustrés par un modèle face à des abus agrégés : l’absence de ventilation ne permet pas de choisir arbitrairement un des 58 facteurs.

La méta-analyse *Childhood maltreatment and defenses*, [PMID 42987632](https://pubmed.ncbi.nlm.nih.gov/42987632/), publiée en novembre 2026, a été identifiée ; son texte complet n’a pas été lu. Elle reste une piste explicite, pas une source active. « Aucune preuve trouvée » signifie ici « aucune paire précise admissible confirmée dans les textes effectivement lus », jamais preuve universelle d’absence.

| Repère du catalogue | Nature et décision |
| --- | --- |
| `intellectualization` | Analyse ordinaire ou mécanisme possible selon le contexte ; aucune attribution automatique. Pas de carte nouvelle. |
| `rationalization` | Processus possible ; une explication sincère n’est pas un mensonge ni la preuve d’une défense. |
| `minimization` | Pas de mécanisme autonome parmi les 30 DMRS ; aucun alias forcé au déni ou à la rationalisation. |
| `emotional_detachment` | État momentané ; distinct du trait de détachement et d’un trouble dissociatif. |
| `emotional_suppression` | Recouvrement éditorial conservé avec `pat_emotional_suppression` ; distinct de la suppression volontaire adaptative du DMRS. |
| `all_or_nothing` | Processus cognitif, pas équivalent au clivage ; compétence du lot 6. |
| `overcontrol` | Processus contextuel, insuffisant pour un qualificatif d’anankastie ou un diagnostic obsessionnel-compulsif. |
| `withdrawal` | Comportement contextuel, non synonyme de personnalité schizoïde/évitante. |
| `impulsive_reaction` | Réaction possible, non équivalente à désinhibition, passage à l’acte défensif ou trouble antisocial. |
| `humor_coping` | Ressource ordinaire ; proximité avec une défense mature sans pathologisation ni nouveau calcul. |

Ces décisions sont des choix sémantiques de couverture, pas des évaluations cliniques. Les ressources et compétences normales restent hors du calcul d’effets défavorables.

## Corrections réservées à l’intégrateur

1. Retirer l’alias existant `icd_6d10 → icd_6d11_5`, qui assimile toute la personnalité au seul motif borderline.
2. Ajouter `contextual_only:true` au motif borderline existant, sans redéclarer le sujet.
3. Étendre la portée de lecture de `oms_cddr` aux pp. 553–564 effectivement lues.
4. Ajouter le libellé d’interface `personality_trait` ; les diagnostics proposés ne doivent pas être affichés comme diagnostics CIM.

Les six propositions ne doivent pas être activées par simple copie : elles nécessitent une synthèse ou une cohorte portant sur le construit exact. `proposed_outcomes` a été ajouté sur instruction du pilote et reste ignoré par le générateur.

## Contrôles

Contrôles locaux non écrivants : collisions d’IDs et de paires contre les données publiées et les autres fragments, références résolues, chaque sujet actif relié, domaines explicites, catégories autorisées et garde-fous contextuels. Résumés tous sous 350 caractères, sans pourcentage ni terme causal interdit. Aucun identifiant publié redéclaré.

La commande `b.build()` a été exécutée sans génération ; l’empreinte d’audit périmée est attendue. Le dernier résultat chiffré après arbitrages est ajouté ci-dessous. Les totaux globaux reflètent aussi les écritures des autres lots et ne constituent pas un décompte attribuable au lot 4.

Résultat final : 4 sujets actifs, 12 liens (6 preuves modérées, 6 limitées), 10 sources nouvelles ; 83 décisions de registre, 6 sujets et 6 liens proposés, 4 corrections proposées. Longueur maximale des résumés : 243 caractères. Les contrôles de collision, de référence et de contextualisation passent. `b.build()` donne à cet instant 59 facteurs, 106 sujets, 520 liens et 271 sources ; sa seule erreur est `audit des correspondances périmé : les facteurs, sujets, liens ou ressources ont changé`. Aucun fichier généré, aucune publication ni opération Git d’écriture n’a été exécuté.
