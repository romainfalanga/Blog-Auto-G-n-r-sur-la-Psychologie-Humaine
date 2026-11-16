# Livraison Ambre et Cartographie — 17 novembre 2026

Base : PR83, `1b86169`. Chantier éditorial des personnages inchangé. Les décisions du propriétaire concernant l’encadré3114 et le bouton de suppression de la page Cartographie sont appliquées ; les voies serveur de sécurité/suppression restent disponibles.

## Résultat intégré

- Corrections de l’audit Ambre §14 sauf le code mort15 explicitement exclu ; date, mémoire récente, question finale conservée, faux positifs de sécurité, bilans et observations, budget de réécriture.
- 36 stratégies sourcées, planV3 retenu après deux cycles de critique, suivi corrigible et relances choisies ; Mon suivi/Paramètres/coopération professionnelle raccordés.
- Neuf lots de recherche,161 correspondances relues, registre1567décisions. Graphe final59questions/109cartes/493liens/260sources/11ressources et3IDs historiques sans réinterprétation. Aucun score diagnostique individuel ni fausse affirmation de lecture du manuel DSM complet.
- Relectures indépendantes finales favorables après correction de courses de trace, provenance, mémoire et export. Les restrictions des méthodes restent entières ; le contexte saturé peut interdire une nouvelle méthode plutôt que couper ses limites.

## Vérifications avant publication

Les commandes de `.github/workflows/verify.yml` ont été exécutées. Site :32tests ; Python :43personnels,13pipeline,21contrats. Générateurs/navigation/catalogues/registre/audit/editorial contrôlés ; Hugo complet7931pages. Cartographie :42tests,1770comparaisons public/Worker,16scénarios navigateur. Recette `tests/change-plan-browser.cjs` avec vrai Worker et SQLite :1440/390/320px, objectifs/étapes/essais, dates inconnues, conflits avec saisie conservée, préparation privée, catalogue, portrait conservé, suspension et purge du chat gardant le plan. Aucune erreurJavaScript ni débordement horizontal dans ces parcours.

Le nombre final de tests journal et les identifiants de publication sont complétés après la dernière passe. Les fixtures du moteur simulent le fournisseur ; elles mesurent l’estimation utilisée par les réservations, pas une facture réelle ou une efficacité clinique. Profil rempli avec deux méthodes : environ11742jetons d’entrée et29098réservés avec réécriture, sous le plafond nominal30000.

## Publication et recette distante

En attente à ce point du document. Une seule migration nouvelle :`0021_change_plan.sql`. La base distante ne possède pas de table d’historique `d1_migrations` ; `sqlite_master` confirme les structures0020 et l’absence de0021 avant intervention. Les32modules du Worker publié correspondent exactement par SHA256 à la basePR83 ; ses7bindings, son secret et son cron minute ont été vérifiés avant intervention. La version complète à livrer comprend38modules.

Les preuves de migration, Worker, fusion, Netlify, parcours avec fournisseur réel et nettoyage seront ajoutées après constat, sans présenter les tests locaux comme une vérification de production.
