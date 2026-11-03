# Lots C, D et F — livraison et preuves

Date : 3 novembre 2026. Implémentation locale ; aucun appel IA réel, achat, publication ou modification des récits publics effectué par ce lot.

## Ce qui change

- Le portrait V2 utilise des sources typées et leurs révisions/hash. Extraction incrémentale des seules sources nouvelles ou modifiées ; dérivés réutilisés après vérification des citations originales. Synthèse et contrôle sémantique distincts. Les assertions, forces, états et préférences ont des preuves ; la fiction n'entre pas dans le portrait. Un like nu reste un intérêt de confiance ≤0,25, même si une citation du titre pourrait sembler soutenir une biographie.
- Une source retirée ne revient pas via une ancienne synthèse. Les champs utilisateurs ne sont plus coupés à 120 caractères dans le pipeline V2. La couverture non traitée reste dans `coverage.pending` ; le Worker poursuit l'analyse avec les dérivés déjà stockés, en respectant les limites gratuites et techniques.
- Les trois identités sont préparées ensemble pour les slots manquants. Les identités existantes sont conservées. Un slot migré dont l'identité manque doit citer ses archives fictives pour retrouver son nom. Lorsque toutes les sources sont `understanding_only`, les trois propositions sont explicitement exploratoires, sans fausses preuves ni détail protégé.
- Le brief filtre les anciennes facettes dont une source a disparu ou est devenue protégée, conserve l'identité fictive, les corrections complètes, les sources pertinentes, le parent, cinq épisodes récents et des épisodes anciens retrouvés parmi tout l'historique. Il récupère aussi les résolutions ultérieures d'anciens fils, même au-delà de vingt épisodes.
- La chaîne récit V2 réalise cadrage → rédaction → contrôle structuré → remplacement ciblé et second contrôle si nécessaire → résumé/delta du corps final complet → questions. La validation locale exige 1000–2200 mots, au moins trois H2, un titre, des preuves exactes dans chaque delta et une résolution sourcée dans le dernier tiers. Le chemin nominal synthétique utilise cinq appels, six avec correction ; ce n'est pas une mesure du fournisseur.
- Le transport garde OpenRouter et les modèles configurés côté Worker. Il transmet la température par étape et les versions, reprend le compteur d'appels cumulé, reçoit les métriques sans texte privé et rejoue une completion perdue avec le même résultat. Aucun nouveau récit n'est rédigé lors de ce replay.
- Le petit référentiel pédagogique `scripts/personal_concepts.json` borne les formulations à trois repères vérifiés sur les pages officielles WHO et Centre for Clinical Interventions : observation/nommage, retour au présent et expression respectueuse d'une limite. Il n'autorise ni diagnostic ni promesse de progrès.

## Classement et catalogue

Les canaux demande, déclaration, résonance, like, différence et évitement sont distincts. Une différence nominale (« le deuil familial ») ne devient pas positive. Une double négation (« je ne veux pas éviter la colère ») ne filtre pas la colère. Seul le champ d'évitement séparé ou un évitement confirmé produit un filtre. Les sources sont plafonnées avant la saturation finale ; le portrait ne double pas leurs contributions. Les raisons indiquent la catégorie et l'ID de la source réelle. Les sujets sans rapport ne remplissent pas artificiellement douze positions. Les expositions récentes peuvent réduire légèrement la répétition sans devenir une preuve personnelle.

Le classement V2 fonctionne avec une relation décrite, un événement ou une mémoire et zéro like. Un nom/rôle de proche seul ne déclenche rien. La baseline reste appelable en recette avec `recommend(context, {algorithm:'baseline'})`. L'option interne `includeLiked:true` sélectionne les inspirations publiques par pertinence et diversité, y compris quand aucune lecture n'est aimée.

`build_examples.py` fabrique un manifeste unique pour les 231 articles, avec concept canonique, contexte, ressources, version, hash et extraits exacts d'ouverture/dernière scène. Les offsets sont vérifiés sur les corps originaux. Les résumés de matrice non vérifiés ne sont plus repris. La méthode est nommée `extractive_opening_and_final_scene` : elle n'est pas présentée comme une synthèse exhaustive de chaque péripétie. `examples.mjs`, les données Hugo et le payload `--seed` viennent de ce même constructeur. Le renouvellement du catalogue D1 fait partie de l'intégration/déploiement.

## Mesure de pertinence

Commandes reproductibles :

```sh
node --test journal/tests/recommendations*.test.mjs journal/tests/portrait-contract.test.mjs
python -m unittest discover -s journal/tests -p 'test_personal*.py'
node journal/tests/evaluate-recommendations.mjs
```

Résultat local : 34 tests JS ciblés et 35 tests Python passent. Douze fixtures de provenance sont communes aux validateurs Python/Worker. Les tests incluent dix profils synthétiques de contexte, un récit final de plus de mille mots, corrections ciblées, trente-cinq/quarante épisodes, sources retirées, budget cumulé, scène protégée seule, identité migrée et replay de completion.

Le fichier `journal/tests/fixtures/recommendation-scenarios.mjs` contient trente scénarios français et leurs jugements explicites. Les coefficients ont été gelés avant la création de ces scénarios ; aucune retouche n'a été faite à partir du score observé. `docs/evaluation-recommandations-v2.json` conserve les résultats par scénario et par cohorte.

| Cohorte | Version | NDCG@6 | NDCG@12 | Précision@6 | Nombre retourné moyen |
|---|---|---:|---:|---:|---:|
| 15 avec likes | Baseline | 0,767 | 0,794 | 0,233 | 12,00 |
| 15 avec likes | V2 | 0,771 | 0,786 | 0,200 | 6,07 |
| 15 sans like | Baseline | 0 | 0 | 0 | 0 |
| 15 sans like | V2 | 0,806 | 0,831 | 0,278 | 6,73 |

La baseline renvoie zéro sans like par construction. Le résultat global ne démontre donc pas un meilleur classement pour les personnes ayant déjà des likes. Sur cette cohorte, NDCG@6 est pratiquement inchangé, NDCG@12 et précision@6 diminuent. La V2 renvoie moins de propositions et évite le remplissage ; il faut une revue éditoriale indépendante et un jeu réservé avant tout réglage ou affirmation de gain.

## Limites à conserver dans la communication

Les annotations sont synthétiques, peu denses (articles non annotés comptés zéro) et non indépendantes. Les dix profils et récits simulés valident les contrats et frontières, pas la qualité littéraire produite par le modèle. Aucun débit réel, latence fournisseur, coût réel ou taux de réussite de génération n'a été mesuré. Le contrôle sémantique améliore la provenance mais n'est pas une preuve de vérité psychologique. La récupération des anciennes intrigues est lexicale et bornée ; elle reste à évaluer sur de longues sagas générées réellement. Les sources très volumineuses peuvent nécessiter plusieurs démarrages et sont affichées comme partiellement traitées plutôt que tronquées.
