# Livraison Mon compte, chronologies et bulle Ambre — 17 novembre 2026

Cet état remplace les anciennes descriptions de Mon portrait, de la navigation, des formulaires d’événements et des crédits hebdomadaires.

## Résultat

- **Mon compte** remplace Mon portrait et ferme le menu. L’identité affiche prénom, nom, date de naissance et genre, avec précision facultative pour un genre exprimé librement. Présentation et activité sont masquées, sans effacer les anciennes valeurs. Abonnement, export, déconnexion et suppression du compte sont réunis ici. `/mon-portrait/` redirige vers `/mon-compte/`.
- **Accès** : l’accès gratuit est maintenu selon la réponse du propriétaire. Le paiement Stripe reste non configuré ; ni tarif ni essai existants ne sont modifiés. Le serveur applique 100 messages et un bilan Ambre par mois civil Europe/Paris. Initial et point partagent ce crédit bilan, sans débiter de message. Réservations, échecs, reprises, concurrence et suppression sont couverts ; effacer un contenu ne restitue pas un crédit consommé. Les budgets techniques hebdomadaires restent distincts. La limite quotidienne de dépense IA reste désactivée.
- **Ma ligne de vie** : formulaire réduit et dates adaptées à chaque précision, sans jour inventé. Les anciens ressentis/soutiens sont conservés en données mais retirés du formulaire. Chaque événement peut recevoir plusieurs notes horodatées, modifiables avec contrôle de révision.
- **Mes proches** : chronologie commençant par la rencontre, même sans date, puis événements communs, notes personnelles et observations proposées par Ambre puis acceptées. Les événements ne sont pas copiés ; leurs notes restent communes aux deux vues. Les anciens repères automatiques restent consultables et distingués des observations validées.
- **Ambre** : bulle sur les pages publiques et privées, panneau sur ordinateur et plein écran sur mobile ; conversation directe toujours disponible. Origine et fenêtre vérifiées pour chaque message entre fenêtres. Brouillon conservé entre ouvertures dans le même document, dictée arrêtée à la fermeture, focus bouclé, session vérifiée avant réaffichage. L’accueil reste public et le chargement asynchrone du lanceur ne bloque pas celui des pages.
- **Prérequis** : identité complète, au moins trois proches et trois événements, et 60 questions actives de Mon enfance renseignées, enregistrées et consenties. Non, Je ne sais pas et anciennes valeurs 2 comptent comme réponses. Ces règles s’appliquent aussi aux comptes ayant déjà discuté, aux reprises et avant appel fournisseur/publication. Une réponse statique de sécurité reste disponible sans génération.
- **Mon enfance** : description visible dès la carte repliée, détails distincts, espace accru au-dessus des réponses, retrait de la mention clinique répétée demandée. Précisions et manifestations s’éditent dans une fenêtre titrée et préremplie avec Enregistrer/Annuler, Échap et retour du focus. Les choix Non/Oui/Je ne sais pas, le tri et les réponses historiques restent préservés.

## Validation

838 tests serveur et 46 tests site réussis, ainsi que toutes les autres commandes de `.github/workflows/verify.yml` : génération des catalogues/navigation, audits et 43 + 13 + 21 tests Python. Recettes locales : 19 parcours de premier affichage/navigation, 12 parcours formulaires/timelines aux quatre largeurs, 55 parcours Mon enfance, tests de dialogue, dates, conflits, session et boucle de focus. Les tests de budget conservent les plafonds antérieurs.

Relectures indépendantes : [données et accès](relecture-compte-timelines-donnees-2026-11-17.md), [bulle et session](relecture-compte-timelines-widget-2026-11-17.md). Deux courses avant fournisseur, ainsi que les problèmes de focus/session, ont été corrigés et contre-vérifiés. Notes seules et notes abondantes ont été vérifiées dans conversation et bilans avec des modèles simulés.

Recette en production : 16 premiers affichages prêts à 320/390/768/1440 px, sauvegarde d’identité et conservation des anciens paragraphes, notes enfance, dates natives, notes communes et chronologie proche, blocage serveur/visuel avant complétude. Le premier passage a révélé l’ancienne redirection explicite de l’accueil vers Ambre ; la PR 98 la retire et protège également le périmètre de la fonction Edge. Le complément après correction valide l’accueil 200, sa bulle, les prérequis complets, les en-têtes d’iframe et l’export.

Les deux comptes synthétiques sont supprimés ; contrôles D1 sur 45 tables par compte, aucun reliquat, aucune violation de clé étrangère. Les fichiers temporaires d’identifiants ont été supprimés. Aucune conversation n’a été lancée par la recette ; les éventuelles tâches de fond du service restent distinctes des appels de test simulés.

## Publications et restauration

- PR 97 : source `975460efaa6f2792bb055259edd5db97ce8d49d8`, fusion `3e023bd480da6043dadefe34f3497d9990720de7`, arbre `6c4f8c1e8df490aae841da861eebf6b11b1ac0f6`. CI `44154966010` réussie.
- D1 : migrations 0025 et 0026 appliquées une seule fois et inscrites dans `d1_migrations` (26 au total). Point de restauration précédent : `000000a6-00000808-00005126-715db9831c85146f94fda563df4278ab`. Pause de génération temporaire, retirée à la publication du Worker.
- Worker : `83f707d5a0fb4860b87632bac92bef2e`, 17 novembre à 20:08:56 Paris. Les 48 modules ont été relus et comparés SHA-256 sans écart. Les 7 liaisons, `FREE_ACCESS_ENABLED=true`, le secret OpenRouter, la date de compatibilité 2026-10-01 et le cron chaque minute sont conservés.
- Netlify initial : `6afca65da26e369b1e217b72`, ready, fusion PR 97, publié à 20:09:57 Paris, Edge présent.
- PR 98 : source `41b905a57ddf51651ee5cff265a5e9641188ec5b`, arbre `2ee9ce8790125f6069c8b3a9e869629873c1d3bc`, CI `44157562080` réussie ; fusion **`9fa668063912bad7810e6cf0721e600aeb4571ba`**. Worker inchangé.
- Netlify final : **`6afca97789f988d5031c06a3`**, ready, fusion PR 98, publié **le 17 novembre 2026 à 20:23:05 Paris**.
- Comptes de recette supprimés : `503b60f0-b68c-4182-b154-4c11854df303` et **`6bfafa50-488a-4d2f-803d-9cab5d456ef5`**.

Preuves de travail locales : `work/compte-timelines/` (rapports, journaux, captures, manifestes et reçus de publication). Le chantier éditorial des personnages reste en pause.
