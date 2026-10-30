# Journal intégré à Décode ton esprit

URL utilisateur : https://decodetonesprit.fr/journal/

Le site original reste sur Netlify. Ses contenus Markdown ne sont pas modifiés. Le Worker Cloudflare sert le journal et son API derrière les réécritures du même domaine; son ancienne page d’accueil workers.dev redirige vers le site original.

## Configuration

- `PUBLIC_ORIGIN=https://decodetonesprit.fr`
- Secret Cloudflare **OPENROUTER_API_KEY** : unique emplacement de la clé du journal.
- `OPENROUTER_MODEL=openai/gpt-5-mini` : variable non secrète.
- Stripe : secrets `STRIPE_SECRET_KEY` et `STRIPE_WEBHOOK_SECRET`, variable `STRIPE_PRICE_ID` correspondant à un prix actif 1000 centimes EUR / mois. Webhook https://decodetonesprit.fr/api/billing/webhook ; événements customer.subscription.*, checkout.session.completed, checkout.session.async_payment_succeeded, invoice.paid et invoice.payment_failed. Activer le portail client pour gérer/résilier.

L’absence de clé IA ou de Stripe est affichée honnêtement dans le journal. Aucun paiement réel ni génération réelle n’est validé sans ces configurations.

## Données et génération

`0001.sql` crée les comptes, sessions, abonnements, profils, mémoire, récits et retours. `0002_inspirations.sql` ajoute les choix d’articles privés et `public_examples`, le corpus public canonique.

`python scripts/build_examples.py --seed /tmp/decode-examples.json` reconstruit le module de métadonnées `src/examples.mjs` et exporte les textes publics à charger dans D1 par INSERT ... ON CONFLICT UPDATE. Le JSON d’export est temporaire; les sources de vérité restent les articles du dépôt. Ne jamais reconstruire les pages publiques à partir des données privées.

Le lecteur peut retenir jusqu’à trois articles. Les raisons et différences sont enregistrées et envoyées au générateur avec les extraits canoniques. Une sélection ne vaut ni diagnostic ni déclaration que l’utilisateur a vécu les événements du personnage. Les corrections utilisateur priment sur les anciennes hypothèses.

Quatre étapes OpenRouter : cadrage à partir du personnage et des inspirations, rédaction, relecture indépendante, synthèse provisoire de la compréhension de l’utilisateur. La synthèse ne reçoit pas les récits fictifs et reste distincte de la mémoire validée/modifiable. Les textes et clés ne sont pas journalisés.

Le Cron Cloudflare `*/5 * * * *` réserve jusqu’à trois utilisateurs, puis traite leurs récits en parallèle. Une histoire maximum par jour et utilisateur, trois essais maximum, contrôle des versions du contexte, de l’abonnement et du consentement à la publication. Une inspiration sauvegardée est obligatoire; catalogue manquant bloque la génération plutôt que produire un texte générique. Les appels IA ont des délais bornés pour rester dans la durée autorisée d’un Cron Cloudflare.

## Publication

1. Appliquer les migrations D1 sans supprimer les données.
2. Charger les 231 corps publics canoniques dans public_examples.
3. Déployer les modules worker.mjs, generator.mjs, ui.mjs et examples.mjs avec DB et variables déclarées dans wrangler.jsonc; préserver les secrets existants.
4. Activer le Cron Cloudflare.
5. Publier les modifications Hugo/netlify.toml sur main du dépôt existant, attendre Netlify, vérifier les routes du domaine.

## Vérification et limites

17 tests automatisés backend/génération passent : isolation des comptes et inspirations, validation du catalogue, mémoire sans fiction, baux et invalidation du contexte, conditions d’éligibilité, erreurs fournisseurs, Checkout et suppression. Hugo compile le site original avec le journal. Aucun article ni personnage Markdown n’est supprimé ou réécrit.

Stripe et la génération OpenRouter réelle restent à valider après configuration. Les frais fournisseur ne sont pas garantis par le tarif client de 10 EUR. La base D1 est européenne; cela ne garantit pas le lieu de traitement du fournisseur IA. La récupération de mot de passe par e-mail reste à ajouter avant une ouverture commerciale complète.
