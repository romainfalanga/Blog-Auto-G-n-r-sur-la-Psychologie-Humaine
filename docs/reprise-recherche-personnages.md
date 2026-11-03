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

La validation en ligne doit confirmer le déploiement Netlify du commit de correction, le titre des pages personnelles après mise à jour du Worker, ainsi que la recherche, les suggestions et le bouton Effacer dans le navigateur.
