# Portrait personnel et personnage unique — V3

## Objectif et décisions

Cette évolution remplace le parcours à trois personnages. Elle conserve les données et les histoires existantes ainsi que le quota de sept histoires par semaine.

| Demande | Mise en œuvre |
|---|---|
| Mes proches séparé | Page `/mes-proches/`, avec les relations et leurs liens vers les événements |
| Ma ligne de vie séparée | Page `/ma-ligne-de-vie/`, événements positifs, difficiles et prises de conscience |
| Mon portrait enrichi | Identité facultative, présentation, perception de soi, schémas répétitifs, émotions, contexte actuel et activité |
| Mon personnage | Page `/mon-personnage/`, un personnage principal stable ; les autres restent en archives |
| Début de l'histoire | Abonnement actif, au moins dix articles aimés, une présentation personnelle et un portrait analysé à jour couvrant toutes les sources |

Les dix articles correspondent aux articles aimés, comme dans le parcours existant. Une réponse dans l'un des cinq champs de présentation suffit à fournir la source personnelle ; aucun seuil arbitraire de longueur n'est imposé. Les informations civiles sont facultatives.

## Plan d'implémentation

1. Ajouter un profil déclaré versionné, distinct des anciens paramètres et du portrait interprété.
2. Brancher cette nouvelle source sur les recommandations, la synthèse du portrait et la préparation narrative.
3. Migrer vers un personnage principal unique sans supprimer les autres identités, histoires ou crédits consommés.
4. Séparer les pages et adapter les menus, les formulaires, les états de progression et les anciennes URL.
5. Tester les conflits d'enregistrement, les migrations, l'isolation des comptes et les contrôles de génération.
6. Valider en environnement isolé, puis migrer et publier la production en préservant ses bindings et ses données.

## Contrat et personnalisation

`GET/PUT /api/self-profile` utilise `expected_revision` ; une modification concurrente retourne 409 et l'interface conserve le brouillon. `profiles.self_data` et `self_revision` ne remplacent pas l'ancien `profiles.data`.

La source typée `profile:<user_id>` rejoint les articles aimés, proches, événements et retours de lecture. Ses citations permettent de distinguer une déclaration personnelle d'une hypothèse. Le portrait devient périmé à chaque changement pertinent ; une génération ne peut pas être publiée avec un contexte obsolète.

Le nom, le prénom et la date de naissance exacte ne sont pas envoyés aux modèles. La génération peut recevoir une tranche d'âge et un genre déclaré contrôlé. Ces données et la profession ne sont pas des preuves de traits psychologiques. Les scènes personnelles restent dans l'analyse : le planificateur et le rédacteur utilisent une abstraction contrôlée, sans recopier la vie de l'utilisateur dans la fiction.

Le personnage principal existant est celui du premier emplacement. Son identité reste stable ; les personnages secondaires sont archivés, leurs histoires restent lisibles, leurs nouvelles générations sont annulées et les crédits réservés sont libérés. Les crédits déjà consommés restent consommés. Les conditions d'accès sont revérifiées à la réservation, à la prise en charge et à la publication.

## Vérification

- Tests JavaScript : profil, CAS, dates, isolation, sources, recommandations, archives, quota, invalidation et navigation.
- Tests Python : citations de présentation, contexte démographique minimal, personnage unique, séparation entre analyse et écriture et protection contre la reprise de scènes privées.
- Tests de recherche et construction Hugo.
- Relecture indépendante : un conflit entre migration et réactivation d’anciens jobs a été identifié, corrigé et vérifié indépendamment avec deux anciens jobs échoués.
- Recette isolée : sept contrôles API réussis (pages, redirections, sauvegarde et conflit 409, date invalide, personnage unique et portrait périmé, conservation des histoires, recommandations). Aucun appel IA payant.
- Validation locale : 105 tests JavaScript du journal, 16 tests de recherche, 70 tests Python, construction Hugo réussie.
- Migration de production 0007 appliquée le 3 novembre 2026, générations suspendues pendant la publication. Le contrôle final de publication est consigné dans le compte rendu de livraison.

Le paiement Stripe reste une configuration externe distincte : cette évolution ne prétend pas activer l'encaissement. Aucun appel IA payant n'est nécessaire pour ces tests de migration et de parcours.
