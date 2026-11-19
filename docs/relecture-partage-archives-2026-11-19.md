# Relecture indépendante du partage des archives

Relecture du 19 novembre 2026 par l’agent ambre_v2, sur les changements de care-backend, care-context, care-interface, shared-review-snapshot, migration 0032 et care-history.test. Aucun fichier de production modifié pendant cette relecture.

## Résultat provisoire : corrections nécessaires

1. **Retrait du consentement pendant une lecture d’archive professionnelle.** La route lit le bilan et l’instantané, puis ne revérifie que le lien de partage. La suppression de l’instantané et l’invalidation du bilan entre ces opérations ne bloquent pas la remise du contenu déjà lu. Contrepreuve exécutée avec le vrai harness SQLite : hook après `SELECT * FROM review_snapshots WHERE review_id=? AND user_id=?`, suppression de l’instantané et passage du bilan à `failed` ; réponse HTTP 200 contenant « Secret retiré » alors que la table ne contient plus aucun instantané. Ajouter à la vérification finale la présence de la même archive et un bilan toujours disponible. Ajouter le test de course correspondant.

2. **Archive ouverte conservée après révocation dans l’interface professionnelle.** `unavailable()` retire le dossier mais le dialogue d’archive est monté dans `document.body`, hors de ce dossier. Le texte déjà affiché reste donc visible après que la surveillance a détecté la révocation. Fermer les dialogues d’archives lors de cette invalidation et d’un changement de patient.

3. **Écriture d’une conclusion : contrat de rubriques incohérent.** L’interface exige `diagnoses` et `reviews`, mais l’autorisation et l’INSERT serveur vérifient seulement `diagnoses`. Si les deux rubriques sont requises, ajouter `reviews` aux contrôles initiaux, au rejeu idempotent et à la condition atomique de l’INSERT.

## Points vérifiés

- Les sections d’archives professionnelles suivent une liste fermée de correspondances avec les autorisations actuelles. Une section future inconnue est exclue. Les associations événements/proches nécessitent les deux rubriques.
- L’évaluation automatique n’est transmise qu’avec `diagnoses`. La notice indique que cette rubrique peut résumer les autres informations ; ce choix doit rester explicite.
- Les plans historiques passent par `serializeSharedPlan` : seules les propriétés autorisées et les éléments adoptés liés entre eux survivent ; consignes privées, propositions et propriétés supplémentaires imbriquées sont exclues.
- Les lectures revérifient la révision du partage. Le test de révocation du lien pendant la lecture réussit ; il ne couvrait pas la suppression de l’archive signalée ci-dessus.
- Les conclusions sont attribuées au compte professionnel, nom et date figés. L’INSERT est conditionné au propriétaire du bilan et à un lien actif de même révision. Le rejeu conserve la même conclusion et rejette un contenu différent.
- L’export du propriétaire inclut les conclusions. Suppression du bilan ou du compte propriétaire : cascade SQL. Révocation du lien : la conclusion attribuée reste dans l’historique du propriétaire. Suppression du professionnel : identité référentielle détachée, nom figé conservé.
- Le contexte Ambre distingue explicitement les conclusions humaines des hypothèses du modèle, identifie les extraits et interdit de traiter ces textes comme des instructions.
- Rendez-vous : formulaire natif soumis au clavier, champ requis, date future contrôlée, mise à jour avec révision, rechargement après succès et bouton de rechargement après conflit 409. Les réponses tardives sont protégées par `renderGuard` et la connexion du conteneur. Les tests serveur couvrent propriété, sélection de la prochaine date et conflit de révision. La vérification visuelle et au clavier réelle reste à la charge de l’intégration root.

## Exécution

`node --test journal/tests/care-history.test.mjs journal/tests/care.test.mjs journal/tests/care-change.test.mjs` : **47 tests réussis, 0 échec**. Journal : `work/reprise-nov19/care-independent-review.log`. Ces tests verts ne lèvent pas les trois constats ci-dessus.

## Contre-vérification des corrections

Les trois constats sont corrigés : vérification finale conjointe du lien, du bilan, de l’archive et du consentement ; fermeture des dialogues lors de la révocation ou d’un changement de patient ; double périmètre Bilans et Évaluations exigé jusque dans l’INSERT. Huit tests de suivi passent, dont trois courses de suppression/invalidation et le retrait d’autorisation pendant la mutation. Les archives et le rendez-vous ont ensuite été vérifiés dans le navigateur à 390 et 1440 px, sans débordement ni erreur JavaScript.
