"""Build the reviewed symptom map without modifying Mon enfance. No model calls."""
import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'editorial/symptomes/v2'
WEIGHT = {'limitee': 1, 'moderee': 2, 'critere': 3}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def build():
    themes = read(SRC / 'themes.json')
    childhood = read(ROOT / 'data/cartographie.json')
    documents = [read(SRC / 'canoniques.json')] + [read(p) for p in sorted(SRC.glob('liens-*.json'))]
    extra = read(SRC / 'sujets-supplementaires.json')
    symptoms, sources = {}, {}
    def check(condition, message):
        if not condition:
            raise ValueError(message)
    for doc in documents + [extra]:
        for source in doc.get('sources', []):
            key = source['id']
            check(source.get('url', '').startswith('https://'), 'Source sans URL HTTPS: '+key)
            if key in sources:
                check(sources[key]['url'] == source['url'], 'Source contradictoire: '+key)
            else:
                sources[key] = source
        for symptom in doc.get('symptoms', []):
            check(symptom['id'] not in symptoms, 'Symptôme dupliqué: '+symptom['id'])
            symptoms[symptom['id']] = copy.deepcopy(symptom)
    for doc in documents:
        for link in doc.get('links_to_canonical', []):
            key = link['symptom']
            check(key in symptoms and symptoms[key]['canonical'], 'Canonique inconnu: '+key)
            symptoms[key]['links'].append({k:v for k,v in link.items() if k!='symptom'})
    raw_count = len(symptoms)
    fusion = read(SRC / 'fusion.json')
    check(fusion.get('link_conflict_policy') == 'strongest_with_all_sources', 'Politique de fusion requise')
    aliases = {}
    for merge in fusion['merge']:
        keep = merge['keep']
        check(keep in symptoms and merge.get('why'), 'Fusion non justifiée: '+keep)
        for drop in merge['drop']:
            check(drop in symptoms and drop!=keep, 'Fusion invalide: '+drop)
            check(symptoms[drop].get('safety') == symptoms[keep].get('safety'), 'Fusion de sécurité ambiguë')
            symptoms[keep]['links'].extend(symptoms.pop(drop)['links'])
            aliases[drop] = keep
    presentation = read(SRC / 'presentation.json')['symptoms']
    group_ids = {g['id'] for g in themes['themes']}
    for key, symptom in symptoms.items():
        symptom.update(presentation.get(key, {}))
        check(symptom['theme'] in group_ids, 'Thème inconnu: '+key)
        check(len(symptom['description'])<=120 and len(symptom['examples'])<=2, 'Aide trop longue: '+key)
        check(symptom.get('safety') in (None,'suicide','automutilation'), 'Sécurité inconnue')
        check(bool(symptom.get('overlap_group')), 'Famille absente: '+key)
    outcomes = {o['id']:copy.deepcopy(o) for o in childhood['outcomes']}
    for outcome in extra['outcomes']:
        check(outcome['id'] not in outcomes, 'Sujet dupliqué: '+outcome['id'])
        outcomes[outcome['id']] = dict(kind='trouble', url='', **outcome)
    excluded = {'ten_diabete_type2','ten_maladies_cardiovasculaires','ten_migraine','ten_fibromyalgie','ten_surpoids','ten_sante_physique','ten_souffrance_enfance','ten_difficultes_scolaires','ten_revictimisation','icd_6b44','icd_6b45','att_desorganisation_enfant','ten_recompense_immediate_jeunes','pat_parentification'}
    links = []
    for key, symptom in symptoms.items():
        merged = {}
        for link in symptom['links']:
            oid = link['outcome']
            check(oid in outcomes, 'Sujet inconnu: '+oid)
            check(oid not in excluded and outcomes[oid].get('category')!='health_outcome', 'Sujet hors périmètre: '+oid)
            check(link['strength'] in WEIGHT, 'Poids inconnu')
            check(link.get('basis') and link.get('summary') and link.get('sources'), 'Lien non documenté')
            check(all(s in sources for s in link['sources']), 'Source non résolue: '+key+' '+oid)
            if oid not in merged:
                merged[oid] = dict(factor=key, **link)
            else:
                old = merged[oid]
                all_sources = sorted(set(old['sources']+link['sources']))
                if WEIGHT[link['strength']]>WEIGHT[old['strength']]:
                    merged[oid] = dict(factor=key, **link)
                merged[oid]['sources'] = all_sources
        links.extend(merged.values())
    used = {l['outcome'] for l in links}
    for oid in used:
        outcomes[oid].pop('contextual_only', None)
    counts = Counter(l['outcome'] for l in links if l['strength']=='critere')
    exceptions = read(SRC / 'criterion-exceptions.json')['exceptions']
    for oid, exception in exceptions.items():
        check(oid in used and exception.get('reason') and exception.get('source') and exception.get('pages'), 'Exception clinique non documentée: '+oid)
        check(counts[oid] == exception['expected_count'], 'Exception clinique périmée: '+oid)
    violations = {oid:counts[oid] for oid in sorted(used) if not 3<=counts[oid]<=12 and oid not in exceptions}
    check(not violations, 'Nombre de critères hors 3..12: '+str(violations))
    counts_theme = Counter(s['theme'] for s in symptoms.values())
    check(max(counts_theme.values())<=30, 'Thèmes dépassant 30: '+str({k:v for k,v in counts_theme.items() if v>30}))
    check(symptoms.get('idees_de_mort',{}).get('safety')=='suicide' and symptoms.get('se_faire_du_mal',{}).get('safety')=='automutilation', 'Signes de sécurité absents')
    requires = read(SRC / 'requires.json')['requires']
    for oid, ids in requires.items():
        check(oid in used and ids and all(i in symptoms for i in ids), 'Requires invalide: '+oid)
    factors = [{k:v for k,v in s.items() if k not in ('theme','links','canonical')} | {'group':s['theme'],'ace':False} for s in symptoms.values()]
    groups = [{k:v for k,v in g.items() if k!='order'} for g in themes['themes'] if counts_theme[g['id']]]
    output = dict(schema_version=1, dataset='symptomes', period=themes['period'], ui=dict(dataset='symptomes',title='Mes symptômes',api='/api/symptomes',draft_key='symptomes-brouillon-v2',period='ces 12 derniers mois'), interpretation={'type':'symptoms','diagnostic':False}, weights=WEIGHT, groups=groups, factors=factors, domains=childhood['domains'], outcomes=[outcomes[k] for k in sorted(used)], links=sorted(links,key=lambda l:(l['factor'],l['outcome'])), sources=[sources[k] for k in sorted(sources)], protections=[], exclusive=[], notes={}, retired_outcomes=[], requires=requires, ranking={'version':'essential_share_desc','order':['essential_share_desc','essential_count_desc','moderate_count_desc','id_asc'],'explanation':'Part des signes caractéristiques présents, prudente quand un sujet a peu de signes'}, coverage={'raw_symptoms':raw_count,'factors':len(factors),'outcomes':len(used),'links':len(links),'merged':len(aliases),'questions_by_theme':dict(sorted(counts_theme.items()))})
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    try: data=build()
    except (ValueError,KeyError) as error: parser.exit(1,str(error)+'\n')
    public=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    worker=copy.deepcopy(data)
    # Full provenance remains in the public/editorial JSON; runtime keeps question/link meaning.
    worker.pop('sources',None)
    for link in worker['links']:
        link.pop('basis',None);link.pop('sources',None)
    compact=json.dumps(worker,ensure_ascii=False,separators=(',',':'))
    module='// Generated by scripts/build_symptomes_carte.py; do not edit.\nexport const SYMPTOMES_CARTE = '+compact+';\n'
    if len(module.encode())>=500_000: parser.exit(1,'Worker module exceeds 500 kB\n')
    interface=copy.deepcopy(data)
    interface['sources']=[]
    details={link['factor']+'|'+link['outcome']:link['summary'] for link in data['links']}
    for link in interface['links']:
        for key in ('summary','basis','sources'):link.pop(key,None)
    outputs=[(ROOT/'data/symptomes_carte.json',public),(ROOT/'journal/src/symptomes-carte.generated.mjs',module),
        (ROOT/'data/symptomes_details.json',json.dumps(details,ensure_ascii=False,separators=(',',':'))+'\n'),
        (ROOT/'data/symptomes_interface.json',json.dumps(interface,ensure_ascii=False,separators=(',',':'))+'\n')]
    for path,text in outputs:
        if args.check:
            if not path.exists() or path.read_text()!=text:parser.exit(1,'Stale generated file: '+str(path)+'\n')
        else:path.write_text(text)
    print(json.dumps(data['coverage'],ensure_ascii=False));print('Worker bytes:',len(module.encode()))

if __name__=='__main__':main()
