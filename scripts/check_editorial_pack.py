"""Check the public story structure and the manually authored V10 pack.

No provider, generation, text repair or publication is performed here.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import json
import re
import sys
import yaml
from build_article_intros import normalized, plain_text

ROOT=Path(__file__).resolve().parents[1]
PACK='psychology-v10'
CATEGORIES={
    'cat1_pensees':('Reprendre le contrôle de ses pensées','reprendre-le-controle-de-ses-pensees'),
    'cat2_emotions':('Comprendre et maîtriser ses émotions','comprendre-et-maitriser-ses-emotions'),
    'cat3_schemas':('Sortir de ses schémas répétitifs','sortir-de-ses-schemas-repetitifs'),
}
SOURCE_BLOCK = re.compile(
    r'^[ \t]*(?:#{1,6}[ \t]+|[-*][ \t]+)?(?:\*{1,2}|_+)?'
    r'(?:sources?|références?|bibliographie)(?:\*{1,2}|_+)?[ \t]*(?::|$)',
    re.IGNORECASE | re.MULTILINE,
)
STAGE_HEADING = re.compile(r'^##[ \t]+(.+)$', re.MULTILINE)
SETEXT_HEADING = re.compile(r'(?<=\S)\n(?:-+|=+)[ \t]*(?=\n|$)')


def title_starts_with_name(title, name):
    if not isinstance(title, str) or not isinstance(name, str) or not name.strip():
        return False
    prefix = re.escape(normalized(name.strip()))
    return bool(re.match(r'^\W*' + prefix + r'(?!\w)', normalized(plain_text(title))))


def structure_issues(meta, body):
    issues = []
    if title_starts_with_name(meta.get('title'), meta.get('personnage')):
        issues.append('le titre commence par le prénom du personnage')
    if SOURCE_BLOCK.search(body):
        issues.append('conserver les références dans les données éditoriales')
    headings = list(STAGE_HEADING.finditer(body))
    if not headings or body.startswith('#') or not body[:headings[0].start()].strip():
        issues.append('ouvrir sur une scène sans intertitre')
    if len(headings) != 4:
        issues.append('quatre étapes après la scène sont requises')
    if SETEXT_HEADING.search(body):
        issues.append('séparer le trait horizontal du paragraphe précédent')
    return issues

def word_count(body):
    text=re.sub(r'https?://\S+','',body)
    text=re.sub(r'<[^>]+>','',text)
    return len(re.findall(r"\b[\w]+(?:[’'\-][\w]+)*\b",text,re.UNICODE))

def read_article(path):
    parts=path.read_text().split('---',2)
    if len(parts)!=3 or parts[0].strip():raise ValueError('frontmatter requis')
    meta=yaml.safe_load(parts[1])
    if not isinstance(meta,dict):raise ValueError('frontmatter invalide')
    return meta,parts[2].strip()

def check_article(path,meta,body):
    errors=[]
    for key in ('title','description','slug','personnage','sujet','contexte'):
        if not isinstance(meta.get(key),str) or not meta[key].strip():errors.append(f'{key}: requis')
    if len(meta.get('title',''))>65:errors.append('titre au-delà de 65 caractères')
    if len(meta.get('description',''))>155:errors.append('description au-delà de 155 caractères')
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+){0,5}',meta.get('slug','')):errors.append('slug ASCII, six mots maximum')
    if meta.get('draft') is not False:errors.append('draft doit être false')
    if not isinstance(meta.get('tags'),list) or len(meta['tags'])<5 or len(set(meta['tags']))!=len(meta['tags']):errors.append('au moins cinq tags distincts')
    episode=meta.get('episode')
    if not isinstance(episode,int) or isinstance(episode,bool) or not 1<=episode<=12:errors.append('episode entre1et12 requis')
    categories=meta.get('categories',[])
    valid_categories=[name for name,slug in CATEGORIES.values() if path.parent.name==slug]
    if len(valid_categories)!=1 or categories!=valid_categories:errors.append('catégorie et répertoire incohérents')
    count=word_count(body)
    if not 1800<=count<=2500:errors.append(f'{count} mots, intervalle1800–2500 requis')
    if len(re.findall(r'^### ',body,re.M))<3:errors.append('trois H3 minimum')
    if not re.search(r'^##\s+Qu[’\']est-ce',body,re.M|re.I):errors.append('définition dédiée requise')
    if not 2<=len(re.findall(r'^### [123]\. ',body,re.M))<=3:errors.append('deux ou trois techniques numérotées requises')
    if '\n---\n' not in body:errors.append('séparateur avant conclusion requis')
    if '**' in body:errors.append('gras interdit par charte éditoriale')
    if re.search('[—–]',body):errors.append('tirets cadratin et demi-cadratin interdits')
    without_links=re.sub(r'\[[^\]]*\]\([^)]*\)|https?://\S+','',body)
    if '/' in without_links:errors.append('slash de rédaction interdit hors liens')
    if not re.search(r'professionnel|psychologue|médecin|soignant',body,re.I):errors.append('rappel professionnel requis')
    return errors,count

def inspect(paths=None,allow_partial=False):
    rows=[];errors=[];by_character=defaultdict(list);fingerprints=defaultdict(list)
    public_articles=0
    for path in paths or sorted((ROOT/'content/posts').rglob('*.md')):
        if path.name=='_index.md':continue
        try:meta,body=read_article(path)
        except Exception as exc:errors.append(f'{path.relative_to(ROOT)}: {exc}');continue
        public_articles+=1
        errors.extend(f'{path.relative_to(ROOT)}: {issue}' for issue in structure_issues(meta,body))
        if meta.get('editorial_pack')!=PACK:continue
        issues,count=check_article(path,meta,body)
        errors.extend(f'{path.relative_to(ROOT)}: {issue}' for issue in issues)
        row={'path':path.relative_to(ROOT).as_posix(),'character':meta.get('personnage'),'episode':meta.get('episode'),'words':count,'slug':meta.get('slug')}
        rows.append(row);by_character[row['character']].append(row)
        # Long shared prose is a review failure, rather than acceptable padding.
        for paragraph in re.split(r'\n\s*\n',body):
            if word_count(paragraph)<80:continue
            normalized=re.sub(r'\W+',' ',paragraph.casefold()).strip()
            fingerprints[normalized].append(row['path'])
    for paragraph,files in fingerprints.items():
        if len(set(files))>1:errors.append('paragraphe de80mots ou plus répété : '+', '.join(sorted(set(files))))
    if not allow_partial:
        if len(by_character)!=13:errors.append(f'{len(by_character)} personnages du pack au lieu de13')
        if len(rows)!=156:errors.append(f'{len(rows)} articles du pack au lieu de156')
        for name,items in by_character.items():
            if sorted(row['episode'] for row in items)!=list(range(1,13)):errors.append(f'{name}: douze épisodes uniques requis')
    counts=Counter(row['slug'] for row in rows)
    errors.extend(f'slug dupliqué: {slug}' for slug,n in counts.items() if n>1)
    return {'public_articles_checked':public_articles,'pack':PACK,'characters':len(by_character),
            'articles':len(rows),'words':sum(r['words'] for r in rows),'rows':rows,'errors':errors}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('paths',nargs='*');parser.add_argument('--allow-partial',action='store_true');parser.add_argument('--report');args=parser.parse_args()
    result=inspect([Path(p).resolve() for p in args.paths] or None,args.allow_partial)
    if args.report:Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('public_articles_checked','pack','characters','articles','words','errors')},ensure_ascii=False,indent=2))
    return bool(result['errors'])

if __name__=='__main__':sys.exit(main())
