# Décode ton esprit

Blog de psychologie vulgarisée auto-généré, hébergé sur Netlify.

## Stack technique

- **Hugo** : générateur de site statique
- **Netlify** : hébergement avec rebuild automatique
- **Python + GitHub Actions** : génération quotidienne de 3 articles via l'API Mammouth
- **API Mammouth** : LLM compatible OpenAI pour la rédaction

## Installation locale

### Prérequis

- [Hugo](https://gohugo.io/installation/) (v0.140.0+)
- Python 3.11+
- Git

### Lancer le site en local

```bash
git clone https://github.com/romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine.git
cd Blog-Auto-G-n-r-sur-la-Psychologie-Humaine
hugo serve
```

Le site sera disponible sur `http://localhost:1313/`.

### Générer des articles manuellement

```bash
cd scripts
pip install -r requirements.txt
export MAMMOUTH_API_KEY="votre-clé-api"
python generate_articles.py
```

## Déploiement sur Netlify

1. Connecter le repo GitHub à Netlify
2. Configurer les paramètres de build :
   - **Build command** : `hugo --minify`
   - **Publish directory** : `public`
3. Netlify rebuildera automatiquement le site à chaque push sur `main`

## Configuration de la génération automatique

1. Aller dans les **Settings** du repo GitHub
2. Cliquer sur **Secrets and variables** > **Actions**
3. Ajouter un secret `MAMMOUTH_API_KEY` avec votre clé API Mammouth
4. Le workflow GitHub Actions génère automatiquement 3 articles chaque jour à 12h (heure de Paris)

### Déclenchement manuel

1. Aller dans l'onglet **Actions** du repo GitHub
2. Sélectionner le workflow **Génération quotidienne d'articles**
3. Cliquer sur **Run workflow**

## Structure du projet

```
├── config.toml              # Configuration Hugo
├── netlify.toml              # Configuration Netlify
├── content/                  # Contenu du site (articles, pages)
├── layouts/                  # Templates Hugo
├── static/                   # Fichiers statiques (CSS, images, llms.txt)
├── scripts/                  # Script Python de génération
│   ├── generate_articles.py  # Script principal
│   ├── config.py             # Listes de sujets/contextes/angles
│   └── generated_topics.json # Tracking anti-doublons
└── .github/workflows/        # GitHub Actions
```

## Les 3 catégories

1. **Reprendre le contrôle de ses pensées** — Biais cognitifs, distorsions cognitives, rumination
2. **Comprendre et maîtriser ses émotions** — Émotions, régulation émotionnelle, intelligence émotionnelle
3. **Sortir de ses schémas répétitifs** — Schémas de Young, styles d'attachement, mécanismes de défense
