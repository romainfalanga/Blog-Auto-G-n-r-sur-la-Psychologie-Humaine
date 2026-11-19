# Recette réelle après recharge OpenRouter — 19 novembre 2026

Le propriétaire a rechargé 10 euros et autorisé les tests. Les appels réels au fournisseur fonctionnent de nouveau. **Aucun bilan complet n’a été publié pendant cette recette : la validation de bout en bout reste à obtenir.** Aucun code applicatif, modèle ou critère de relecture n’a été modifié pour forcer un succès.

## Résultats

1. Première tentative : refus local `provider_cooldown`. La ligne `provider_state` conservait la pause de 900 secondes issue du précédent 402. Son expiration a été avancée à l’heure courante, avec comparaison exacte du motif, de la portée et des deux horodatages. Aucune autre limite n’a été levée.
2. Deux demandes initiales ont reçu de vraies réponses IA, mais ont fini en `context_changed`. L’examen du second essai a établi une modification de la seule section `portrait` : vide lors de la capture, elle avait été remplie par l’analyse automatique exécutée en arrière-plan. Les données saisies n’avaient pas changé. Le contrôle de cohérence a empêché la publication d’un bilan mélangeant deux états. Le compte a été conservé temporairement pour retenter après stabilisation.
3. Sur ce même compte, avec portrait déjà calculé, demande `845808de-4a8e-4229-92dc-13ee42da0b18` : bloc analysé, rédaction, relecture, réécriture, puis refus final `review_not_approved`. Début 16:28:20 UTC, fin de recette 16:30:38 UTC. Le premier appel court avait expiré, mais l’analyse du bloc était conservée et la tâche planifiée a repris les étapes suivantes.

La première relecture de ce dernier essai signalait `invented_quote`, `no_diagnosis` et `quotes_faithful`. L’expression d’interface « dans le passé » était considérée comme citation non vérifiable par le contrôle déterministe. Les raisons de la relecture étaient partiellement contradictoires : elle constatait l’absence de diagnostic explicite tout en refusant ce critère. Le détail du verdict final n’est pas conservé après l’effacement du brouillon ; ne pas affirmer qu’il est identique au premier verdict.

Ces constats demandent une correction ciblée : gérer la concurrence avec l’analyse du portrait sans affaiblir la cohérence ni les contrôles de consentement ; distinguer les libellés de l’interface des citations attribuées à la personne ; vérifier la cohérence du contrat de relecture. Ne pas désactiver les critères pour obtenir un test vert. Aucun nouvel essai payant en boucle n’a été lancé après ce refus final.

## Limites et nettoyage

Les profils et récits étaient explicitement synthétiques. Cette recette vérifie un fonctionnement technique, pas une exactitude diagnostique. Les contrôles en aval d’un bilan réel prêt (archive, export, partage et navigateur) n’ont pas été atteints ; la recette synthétique de la livraison précédente ne les remplace pas.

Trois nouveaux comptes membres ont été utilisés, puis supprimés via l’API ; aucun compte professionnel n’a été créé pendant ce lot. Contrôle D1 final : aucun des trois comptes restant et aucune donnée liée dans les 47 tables à `user_id` vérifiées. Les fichiers de récupération des accès ont été supprimés. Journaux locaux : `work/bilans-historique-nov19/production-recharged/` et `production-stable/`.

Le coût exact total n’a pas été mesuré : ne pas annoncer un montant ni un solde restant. Aucun achat, changement de clé, changement d’abonnement ou modification du plafond quotidien n’a été effectué.
