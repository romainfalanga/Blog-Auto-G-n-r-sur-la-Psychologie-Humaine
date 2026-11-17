# Correction de Mon portrait et affichage direct — 17 novembre 2026

## Consignes finales

La demande de simplification concernait les paragraphes, pas les informations d'identité. Cette correction remplace l'interprétation du lot précédent.

- Restaurer prénom/pseudonyme, nom, date de naissance, genre et sa précision facultative, activité. Conserver un seul paragraphe libre « Ma présentation », sans titres de catégories, émotions, contexte, schémas ou autres blocs narratifs.
- Ouvrir Mon portrait, Mes proches, Ma ligne de vie et Mon enfance avec leurs données prêtes, sans écran de chargement intermédiaire en fonctionnement normal.
- Proposer uniquement Non / Oui / Je ne sais pas à toutes les questions de Mon enfance. Les anciennes fréquences restent enregistrées sans conversion silencieuse et s'affichent comme Oui.
- Retirer les niveaux de preuve des cartes, des détails et des fenêtres d'explication, ainsi que les correspondances de définition, références CIM/DSM et formulations techniques de nomenclature. Conserver les explications utiles et l'intégralité des données scientifiques internes.

## Réalisation

Le formulaire envoie uniquement les champs visibles effectivement modifiés, avec leur révision. Les autres paragraphes historiques ne sont jamais renvoyés ni effacés. La comparaison d'un conflit conserve le brouillon et reprend les valeurs distantes pour les champs non modifiés. Les saisies pendant une sauvegarde et les brouillons entre pages restent protégés.

Le Worker prépare les premières réponses GET avec les routes authentifiées existantes, sans requête HTTP interne ni appel de modèle. Les trois pages personnelles intègrent ces réponses dans leur script protégé par nonce ; le script est parsé dans le head avant affichage du corps. Une navigation interne attend les données avant de changer de destination, sans remplacer la page courante par un chargement. Une réponse tardive ne gagne pas contre une navigation plus récente ou une déconnexion.

Mon enfance reçoit ses réponses initiales via une fonction Netlify Edge limitée aux deux chemins de la page. Elle transmet uniquement le cookie au Worker de cette application. Toutes les réponses personnalisées et de fallback portent des directives de cache privées/no-store, y compris CDN, et Vary Cookie. JSON échappé et consommé une seule fois ; erreurs et nouvelles requêtes de conflit/reprise utilisent les API courantes. Les scripts et le catalogue de Mon enfance sont intégrés dans le head pour supprimer les requêtes de scripts avant le premier rendu.

Les contrôles à trois réponses gardent leurs séparations : Non/Oui sur la première ligne et Je ne sais pas sur toute la seconde sur mobile, trois colonnes sur ordinateur. Aucun changement du moteur scientifique, du schéma ou des réponses enregistrées.

## Validation locale et indépendante

- 800 tests journal réussis ; après le dernier déplacement du script en head, 110 tests UI/bootstrap ciblés relancés avec succès.
- 46 tests site, dont 3 tests de la fonction Edge ; contrôles verify.yml et génération Hugo.
- Lecture seule, isolation de deux comptes, aucune mutation ni génération, échappement de texte `</script>` et Unicode, nonce CSP, en-têtes privés et fallback vérifiés.
- Parcours navigateur : identité et conservation des anciens champs, conflits, chargement au premier affichage, navigation rapide, erreurs, trois réponses, fréquences historiques et absence de nomenclature dans tous les détails.
- Une preuve de transfert HTML interrompu avant le script avait révélé un affichage prématuré du chargement : le script dans le head et le retrait du scaffold l'ont corrigé. Contrôle du premier requestAnimationFrame et de la fin DOMContentLoaded, en plus des parcours fonctionnels.
- Relecture applicative indépendante par l'agent qui n'a pas écrit le Worker, l'Edge ou l'interface ; vérifications finales à consigner dans la livraison.

## Publication

À compléter après publication et recette réelle. Aucune migration, aucun secret changé, aucune génération IA demandée.
