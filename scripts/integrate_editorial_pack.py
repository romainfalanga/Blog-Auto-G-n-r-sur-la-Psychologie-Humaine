"""Integrate already-written psychology packs into public metadata.

Every `data/editorial_pack_v*.json` manifest is integrated in version order
(psychology-v10 for the 13 characters of November 2026, psychology-v11 and later
for the packs produced through scripts/psychology_editorial.py).

This script never writes article bodies and never calls a model or provider.
The complete source pack must pass checks before any publication metadata changes.
"""
from pathlib import Path
from collections import Counter
import argparse
import html
import json
import sys

from check_editorial_pack import ROOT, CATEGORIES, inspect, pack_manifests, read_article

MANIFEST=ROOT/'data/editorial_pack_v10.json'

def dump(value):return json.dumps(value,ensure_ascii=False,indent=2)+'\n'

def age_band(age):
    if age<26:return 'jeune adulte (18-25 ans)'
    if age<36:return 'adulte en construction (26-35 ans)'
    if age<46:return 'adulte en pleine vie active (36-45 ans)'
    if age<56:return 'adulte en bilan de mi-vie (46-55 ans)'
    if age<=70:return 'senior en quête de sens (56-70 ans)'
    return 'adulte âgé (71 ans et plus)'

def history_row(row):
    return {key:value for key,value in {
        'date':str(row.get('date','2026-11-04'))[:10],
        'sujet':row['sujet'],'contexte':row['contexte'],
        'category':row.get('category_key',row.get('category','')),
        'titre':row.get('title',row.get('titre','')),'slug':row['slug'],
        'resume_narratif':row.get('resume_narratif',''),'evolution':row.get('evolution',''),
        **({'editorial_pack':row.get('editorial_pack','psychology-v10'),'episode':row['episode']} if 'episode' in row else {}),
    }.items()}

def public_page(character,rows,weight,*,published_on='2026-11-04T08:00:00+00:00',profile_fields=None):
    import yaml
    meta={'title':character['prenom'],'date':published_on,
          'description':character['description_dynamique'],'slug':character['id'],
          'prenom':character['prenom'],'date_naissance':character['date_naissance'],
          'profession':character['profession'],'situation':character['situation_familiale'],
          'nb_articles':len(rows),'weight':weight,'draft':False}
    meta.update(profile_fields or {})
    esc=lambda value:html.escape(str(value),quote=True)
    prefix='Parcours d’' if character['prenom'][0].lower() in 'aeiouyàâéèêëîïôùûü' else 'Parcours de '
    parts=['---',yaml.safe_dump(meta,allow_unicode=True,sort_keys=False).rstrip(),'---','',
           '<h2 class="parcours-section-title">'+prefix+esc(character['prenom'])+'</h2>',
           '<div class="perso-parcours-list">']
    for row in sorted(rows,key=lambda item:item['episode']):
        category,folder=CATEGORIES[row['category_key']]
        url='/posts/'+folder+'/'+row['slug']+'/'
        parts.extend(['  <div class="parcours-entry">',
          '    <div class="parcours-cat">'+esc(category)+'</div>',
          '    <h3 class="parcours-title"><a href="'+esc(url)+'">'+esc(row['title'])+'</a></h3>',
          '    <p class="parcours-resume">'+esc(row['resume_narratif'])+'</p>',
          '    <a href="'+esc(url)+'" class="parcours-link">Lire</a>',
          '  </div>'])
    return '\n'.join(parts+['</div>',''])

# Character page weights: v10 keeps its historical 21–33; later packs start far
# after the clinical pages (34 + order) so that weights never collide.
def pack_weight_base(pack):
    number=int(pack.rsplit('v',1)[1])
    return 21 if number==10 else 300+100*(number-11)


def validate_manifest(pack,manifest,patterns):
    characters=manifest.get('characters',[]);rows=manifest.get('rows',[])
    if manifest.get('pack')!=pack:raise ValueError('Unknown pack')
    if pack=='psychology-v10' and (len(characters)!=13 or len(rows)!=156):raise ValueError('13 characters and 156 articles required')
    if not characters or len(rows)!=12*len(characters):raise ValueError(f'{pack}: twelve articles per character required')
    names={c['prenom'] for c in characters};ids={c['id'] for c in characters}
    if len(names)!=len(characters) or len(ids)!=len(characters):raise ValueError('Duplicate character')
    for row in rows:
        if row['personnage'] not in names:raise ValueError('Unknown character in row')
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT/'content/posts'):raise ValueError('Article path outside public corpus')
        meta,body=read_article(path)
        if meta.get('editorial_pack')!=pack:raise ValueError('Pack mismatch: '+row['path'])
        if any(meta.get(k)!=row.get(k) for k in ('personnage','slug','episode','title','sujet','contexte')):raise ValueError('Manifest metadata mismatch: '+row['path'])
        if not row.get('resume_narratif') or not row.get('evolution'):raise ValueError('Narrative records required')
        if pack=='psychology-v10':
            if not row.get('pattern_ids') or set(row['pattern_ids'])-patterns:raise ValueError('Unknown or absent pattern: '+row['path'])
        else:
            # Later packs cover concepts absent from the internal catalogue; their
            # concept_id is the reviewed editorial anchor, pattern_ids stay optional.
            if not isinstance(row.get('concept_id'),str) or not row['concept_id'].strip():raise ValueError('Concept required: '+row['path'])
            if not isinstance(row.get('pattern_ids',[]),list) or set(row.get('pattern_ids',[]))-patterns:raise ValueError('Unknown pattern: '+row['path'])
        if not row.get('source_urls') or any(not u.startswith('https://') for u in row['source_urls']):raise ValueError('Source URLs required')
    for name in names:
        if sorted(r['episode'] for r in rows if r['personnage']==name)!=list(range(1,13)):raise ValueError('Incomplete arc: '+name)
    return characters,rows


def expected_outputs(manifests):
    patterns={p['id'] for p in json.loads((ROOT/'data/psychology_catalog.json').read_text())['patterns']}
    if 'psychology-v10' not in manifests:raise ValueError('psychology-v10 manifest required')
    report=inspect()
    if report['errors']:raise ValueError('\n'.join(report['errors']))
    packs={}
    for pack,manifest in manifests.items():
        packs[pack]=validate_manifest(pack,manifest,patterns)
    body_paths={row['path'] for row in report['rows']}
    if body_paths!={row['path'] for _,rows in packs.values() for row in rows}:raise ValueError('Manifest/body paths mismatch')
    pack_ids={c['id'] for characters,_ in packs.values() for c in characters}
    pack_names=[c['prenom'] for characters,_ in packs.values() for c in characters]
    if len(set(pack_ids))!=len(pack_names) or len(set(pack_names))!=len(pack_names):raise ValueError('Duplicate character across packs')
    registry=json.loads((ROOT/'scripts/personnages.json').read_text())
    old=[c for c in registry['personnages'] if c['id'] not in pack_ids]
    if len(old)!=20 or set(pack_names)&{c['prenom'] for c in old}:raise ValueError('Existing character identity conflict')
    clinical_names=set()
    for path in (ROOT/'content/personnages').glob('*.md'):
        if path.name=='_index.md':continue
        meta,_=read_article(path)
        if meta.get('clinical_code'):clinical_names.add(str(meta.get('prenom','')).casefold())
    if {name.casefold() for name in pack_names}&clinical_names:raise ValueError('Pack character name already used by a clinical arc')
    matrix=json.loads((ROOT/'scripts/matrice_combinaisons.json').read_text())
    old_rows=[r for r in matrix['articles'] if r.get('editorial_pack') not in manifests]
    published=set();old_published=set();old_names={c['prenom'] for c in old}
    for path in (ROOT/'content/posts').rglob('*.md'):
        if path.name=='_index.md':continue
        meta,_=read_article(path);published.add((path.parent.name,meta.get('slug',path.stem)))
        if meta.get('personnage') in old_names and not meta.get('editorial_pack'):old_published.add((path.parent.name,meta.get('slug',path.stem)))
    old_keys=set()
    for row in old_rows:
        if row.get('prenom') not in old_names:raise ValueError('Old row for unknown character: '+row['slug'])
        _,folder=CATEGORIES[row['category_key']]
        if (folder,row['slug']) not in published:raise ValueError('Old row has no publication: '+row['slug'])
        old_keys.add((folder,row['slug']))
    # An episode added to (or removed from) an original arc must be recorded with
    # scripts/original_arc_episode.py, which keeps the matrix and the page aligned.
    if old_published-old_keys:raise ValueError('Original article without matrix row: '+', '.join(sorted(s for _,s in old_published-old_keys)))
    new_rows=[];complete=[];pages=[]
    for pack,(characters,rows) in packs.items():
        own_rows=[]
        for row in rows:
            category,_=CATEGORIES[row['category_key']]
            meta,_=read_article(ROOT/row['path'])
            own_rows.append({**row,'category_name':category,'prenom':row['personnage'],
                             'date':str(meta['date'])[:10],'tags':meta['tags'],
                             'editorial_pack':pack,'writing_method':'direct_authored_no_generation_api'})
        new_rows.extend(own_rows)
        published_on=manifests[pack].get('published_on','2026-11-04T08:00:00+00:00')
        for index,character in enumerate(characters,pack_weight_base(pack)):
            own=[r for r in own_rows if r['prenom']==character['prenom']]
            c={**character,'tranche_age':age_band(character['age']),
               'historique_articles':[history_row(r) for r in sorted(own,key=lambda r:r['episode'])]}
            complete.append(c)
            page_date=character.get('published_on',published_on)
            pages.append((c,public_page(character,own,index,published_on=page_date)))
    all_rows=old_rows+new_rows
    counts=Counter(row['slug'] for row in all_rows)
    if any(n>1 for n in counts.values()):raise ValueError('Duplicate slug across old and new corpus')
    # Current published articles, not stale phantom history, define each arc.
    for character in old:
        character['historique_articles']=[history_row(r) for r in all_rows if r['prenom']==character['prenom']]
    registry['personnages']=old+complete
    registry['description']=f'Registre des {len(old)+len(complete)} personnages fictifs de Décode ton esprit. Les historiques correspondent aux publications conservées.'
    matrix['articles']=all_rows
    presentations=json.loads((ROOT/'data/character_presentations.json').read_text())
    outputs={ROOT/'scripts/personnages.json':dump(registry),ROOT/'scripts/matrice_combinaisons.json':dump(matrix)}
    for character,page in pages:
        outputs[ROOT/'content/personnages'/f"{character['id']}.md"]=page
        presentations[character['prenom'].lower()]={'name':character['prenom'],'presentation':character['presentation']}
    outputs[ROOT/'data/character_presentations.json']=dump(presentations)
    # The curated discovery order is kept; new pack characters are appended once.
    discovery=json.loads((ROOT/'data/character_discovery.json').read_text())
    outputs[ROOT/'data/character_discovery.json']=dump(discovery+[c['id'] for c in complete if c['id'] not in discovery])
    index_path=ROOT/'content/personnages/_index.md'
    index_meta,index_body=read_article(index_path)
    index_meta['description']='Explorez les histoires et les parcours de Décode ton esprit.'
    import yaml
    outputs[index_path]='---\n'+yaml.safe_dump(index_meta,allow_unicode=True,sort_keys=False)+'---\n'+('\n'+index_body+'\n' if index_body else '')
    return outputs,len(old)+len(complete),len(all_rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if not MANIFEST.exists():parser.error('Complete manifest missing')
    manifests={name:manifest for name,(_,manifest) in pack_manifests().items()}
    outputs,characters,articles=expected_outputs(manifests)
    if args.check:
        stale=[p.relative_to(ROOT).as_posix() for p,value in outputs.items() if not p.exists() or p.read_text()!=value]
        if stale:print('Metadata out of date: '+', '.join(stale));return 1
    else:
        for path,value in outputs.items():path.write_text(value)
    print(f'{len(outputs)} metadata files verified; {characters} characters and {articles} articles; article bodies unchanged.')
    return 0

if __name__=='__main__':sys.exit(main())
