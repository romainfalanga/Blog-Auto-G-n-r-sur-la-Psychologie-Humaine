# Recherche des personnages et reprise du 3 novembre 2026

Demande récupérée depuis la conversation Décode ton esprit interrompue. Dépôt vérifié : `romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine`. Le site public est compilé avec Hugo et publié par Netlify sur https://decodetonesprit.fr ; les pages personnelles sont servies par le Worker Cloudflare `decode-journal`, derrière les routes du même domaine. GitHub Actions orchestre le pipeline des portraits et histoires personnelles.

## Exigences et réalisation

- R1 : titre d’onglet uniquement « Décode ton esprit », dans Hugo et les pages personnelles, y compris lors des navigations internes.
- R2 : réduire l’espace entre l’introduction et les personnages, en conservant l’identité visuelle et le contenu de l’accueil.
- R3 : remplacer les deux filtres par une recherche unique. Les suggestions et repères se trouvent dans ce champ. La recherche combine les mots, classe les résultats par pertinence, normalise accents et mots composés, reconnaît des formulations courantes et tolère une faute légère. L’index couvre présentations, professions, situations, traits, tendances, affinités thématiques, sujets, contextes et mots-clés des articles. Aucun appel IA ni service externe n’est nécessaire pour la recherche.
- R4 : retirer les blocs « Évolution » des parcours publics ; conserver les données d’évolution dans les fichiers et le pipeline internes. Exclure ces blocs de l’index public. Le générateur ne produit plus ces blocs.
- R5 : remplacer les anciens liens de parcours par « Lire » ; appliquer aussi ce libellé dans le générateur pour les futures publications.

Les sources existantes et les 231 articles / 20 personnages sont conservés. Les anciennes modifications non enregistrées ont été récupérées et comparées à l’état GitHub `7411ad2` avant intégration. La copie de l’ancienne session est conservée sans modification.

## Vérifications locales

- Compilation Hugo 0.140.0 réussie.
- 11 tests Node du moteur de recherche réussis.
- Contrôle sur index réel : 291 associations thématiques et 693 mots-clés d’articles retrouvent leur personnage, aucune omission.
- 20 pages personnages : titre conforme, aucun bloc évolution, 231 liens « Lire » et aucune cible locale manquante.
- Les tests backend existants sont adaptés au nouveau titre unique ; les contrôles de comptes, de conservation des notes et d’isolation restent présents.

Commandes : `hugo --minify`, `node --test tests/character-search.test.cjs journal/tests/*.test.mjs`, `python -m unittest discover -s journal/tests -p 'test_personal_pipeline.py'`.

## Publication et vérification en ligne

- Correctif enregistré dans le commit `f32afbd0db231c308b6e8272f59b061dc41b9d9d` sur `main`.
- Déploiement Netlify `6ae9b59022dbd5ba80842453` publié avec l’état `ready` sur le domaine existant.
- Worker mis à jour par remplacement de contenu uniquement, en conservant configuration, liaisons et secrets ; déploiement `88753b331789470f91ab17087cf15c87`.
- Dans le navigateur en ligne : « je pense trop » retrouve Sophie en tête ; les suggestions se sélectionnent ; « Sophie restructuration cognitive » retrouve Sophie ; une recherche inconnue affiche un état vide ; Effacer rétablit les 20 personnages.
- Parcours de Sophie contrôlé en ligne : aucun bloc Évolution, liens Lire et titre exact.
- Mon portrait, Mes histoires et Mon compte chargent avec le titre unique, y compris après navigation interne. Vérification publique sans connexion ; aucune donnée de compte modifiée.
- 60 tests automatisés réussis : 11 moteur de recherche, 40 backend/recommandations et 9 pipeline Python. Contrôle visuel effectué sur navigateur de bureau ; appareil mobile réel non testé.

Les cinq exigences de cette reprise sont réalisées et vérifiées. Les étapes historiques d’activation de la facturation et de génération réelle, documentées ailleurs, ne font pas partie de cette correction d’interface.
