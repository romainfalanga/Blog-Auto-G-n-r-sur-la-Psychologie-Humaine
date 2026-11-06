import copy
import hashlib
import json
import pathlib
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).parents[2]/'scripts'))
import generate_personal_articles as personal
import build_examples
import run_journals as runner


def source(sid='memory:user',text='Je souhaite explorer la colère.',explicit=True,revision=1):
    kind=sid.split(':')[0];field={'profile':'self_description','memory':'text','person':'relationship_text','event':'description','feedback':'correction','like':'reason' if explicit else 'title'}[kind]
    original=text;text=field+': '+text
    return {'id':sid,'type':kind,'revision':revision,'hash':hashlib.sha256(text.encode()).hexdigest(),'text':text,'explicit':explicit,'fields':{field:original}}


def portrait(s):
    return {'schema_version':2,'summary':'Un intérêt déclaré pour la colère.','summary_evidence':[s['id']],
            'axes':[{'label':'Colère','why':'Un intérêt de lecture formulé.','confidence':.2,'claim_type':'interest','evidence':[s['id']]}],
            'strengths':[],'uncertainties':['Un intérêt ne décrit pas toute la personne.'],
            'citations':[{'source_id':s['id'],'revision':s['revision'],'quote':s['text']}],
            'coverage':{'processed':[s['id']],'pending':[]},'psychological_map':[]}


def contextual_portrait():
    first=source('profile:user','Au travail, je pense souvent en boucle et cela me prend une heure.')
    second=source('person:manager','Quand mon responsable change une consigne, je vérifie plusieurs fois.')
    result=portrait(first)
    result['summary']='{{prenom}} décrit des pensées répétées au travail.'
    result['citations']=[{'source_id':s['id'],'revision':s['revision'],'quote':next(iter(s['fields'].values()))} for s in (first,second)]
    result['coverage']['processed']=[first['id'],second['id']]
    result['psychological_map']=[{
        'pattern_id':pattern,'status':'exploring','observation':'{{prenom}} décrit une réaction située.',
        'possible_function':'Une recherche de solution serait à explorer.','alternatives':['Une contrainte temporaire.'],
        'next_question':'Quand cette réflexion vous aide-t-elle ?','evidence':[s['id']],'counter_evidence':[]
    } for pattern,s in [('rumination',first),('uncertainty',second)]]
    result['psychological_map'][0]['influence']={'level':'high','contexts':['Au travail'],
        'rationale':'{{prenom}} rapporte une heure de pensées répétées.','evidence':[first['id']]}
    result['psychological_map'][0]['interactions']=[{'pattern_id':'uncertainty','kind':'amplifies',
        'explanation':'Ces vérifications pourraient entretenir la réflexion au travail.','evidence':[first['id'],second['id']]}]
    return [first,second],result


def job(kind='portrait',s=None):
    s=s or source()
    return {'schema_version':2,'id':'j','lease_token':'secret','kind':kind,'source_version':3,'memory_version':2,'memory':{'version':2},
            'sources':[s],'source_manifest':{s['id']:{k:s.get(k) for k in ('revision','hash','type','explicit')}},
            'portrait':portrait(s)}


def derivative(s):
    return {'source_id':s['id'],'revision':s['revision'],'source_hash':s['hash'],'analysis_version':2,
            'derivative':{'signals':[{'kind':'interest','text':'Intérêt pour la colère','quote':s['text'],'temporal':'non précisé'}]}}


class V2Tests(unittest.TestCase):

    def test_v3_requires_a_map_list_while_unversioned_legacy_portraits_remain_compatible(self):
        s=source()
        for absent in (False,True):
            p=portrait(s);p['analysis_version']=3
            if absent:p.pop('psychological_map')
            else:p['psychological_map']=None
            with self.subTest(absent=absent),self.assertRaises(ValueError):personal.validate_portrait_v2(p,[s])
            p.pop('analysis_version');personal.validate_portrait_v2(p,[s])

    def test_v3_language_guard_covers_axis_and_preference_labels_without_rewriting_quotes(self):
        sources,p=contextual_portrait();p['analysis_version']=3
        for field in ('axis','preferred','avoid'):
            broken=copy.deepcopy(p)
            if field=='axis':broken['axes'][0]['label']='Elle est anxieuse'
            else:broken['reading_preferences']={'preferred':[],'avoid':[]};broken['reading_preferences'][field]=[{'label':'Elle souhaite comprendre','evidence':[sources[0]['id']]}]
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'portrait_language'):personal.validate_portrait_v2(broken,sources)
        p['axes'][0]['label']='Les mots « Elle est anxieuse »'
        quotes=copy.deepcopy(p['citations']);personal.validate_portrait_v2(p,sources)
        self.assertEqual(p['citations'],quotes)

    def test_uncited_current_correction_is_counter_context_and_new_conclusions_require_own_citations(self):
        sources,p=contextual_portrait();memory=source('memory:current','Correction actuelle : je ne rumine plus et je me sens reposé.',revision=2)
        all_sources=[*sources,memory];p['coverage']['processed'].append(memory['id']);original=copy.deepcopy(p);seen=[]
        def reject(_,payload):
            data=json.loads(payload['user_prompt']);seen.append(data)
            self.assertEqual([s['id'] for s in data['original_sources']],[s['id'] for s in sources])
            self.assertEqual(data['correction_context'],[memory])
            self.assertTrue(all(memory['id'] not in claim['source_ids'] for claim in data['claims']))
            self.assertIn('jamais comme',payload['system_prompt'])
            return {'content':json.dumps({'checks':[{'path':x['path'],'supported':x['path']!='summary'} for x in data['claims']]})}
        adapter=personal.Adapter(job(),reject)
        with self.assertRaises(ValueError):personal.verify_portrait_support(p,all_sources,adapter)
        self.assertEqual(adapter.calls,1);self.assertEqual(p,original)
        p['summary']='{{prenom}} rapporte une amélioration actuelle.';p['summary_evidence']=[memory['id']]
        p['axes'][0]['evidence']=[memory['id']]
        p['citations'].append({'source_id':memory['id'],'revision':memory['revision'],'quote':memory['fields']['text']})
        p['psychological_map'][0].update(status='contradicted',counter_evidence=[memory['id']],interactions=[])
        p['psychological_map'][0].pop('influence',None)
        personal.validate_portrait_v2(p,all_sources)
        def accept(_,payload):
            data=json.loads(payload['user_prompt'])
            self.assertIn(memory['id'],[s['id'] for s in data['original_sources']])
            self.assertEqual(data['correction_context'],[])
            return {'content':json.dumps({'checks':[{'path':x['path'],'supported':True} for x in data['claims']]})}
        personal.verify_portrait_support(p,all_sources,personal.Adapter(job(),accept))
        self.assertEqual(p['summary_evidence'],[memory['id']]);self.assertEqual(p['psychological_map'][0]['counter_evidence'],[memory['id']])
        self.assertEqual(p['citations'][-1]['quote'],memory['fields']['text'])

    def test_counter_context_budget_preserves_pending_coverage_and_prioritizes_current_originals(self):
        sources,p=contextual_portrait()
        current=source('person:pending','Mon regard sur cette relation a changé.',revision=2)
        optional=source('event:other','Une rencontre agréable.')
        p['coverage']['pending']=[current['id'],optional['id']];saved=copy.deepcopy(p)
        size=lambda s:len(json.dumps(s,ensure_ascii=False,separators=(',',':')).encode('utf-16-le'))//2
        limit=sum(size(s) for s in [*sources,current])
        context=personal.portrait_support_context(p,[*sources,optional,current],source_char_limit=limit)
        self.assertEqual(context['correction_context'],[current]);self.assertEqual(context['correction_context_pending'],[current['id']])
        self.assertEqual(context['support_context_coverage']['omitted_ids'],[optional['id']]);self.assertEqual(p,saved)
        with self.assertRaises(personal.GenerationFailure) as caught:
            personal.portrait_support_context(p,[*sources,current],source_char_limit=limit-1)
        self.assertEqual(caught.exception.code,'quota_exceeded');self.assertEqual(caught.exception.validation_codes,('support_context_budget',))

    def test_valid_derivative_can_prioritize_an_original_but_never_supply_replacement_truth(self):
        sources,p=contextual_portrait();other=source('person:other','La relation semble plus apaisée aujourd’hui.')
        item=derivative(other);item['derivative']['signals'][0]['kind']='correction'
        base=sum(len(json.dumps(s,ensure_ascii=False,separators=(',',':')).encode('utf-16-le'))//2 for s in sources)
        with self.assertRaises(personal.GenerationFailure):personal.portrait_support_context(p,[*sources,other],[item],source_char_limit=base)
        invalid=copy.deepcopy(item);invalid['source_hash']='outdated'
        context=personal.portrait_support_context(p,[*sources,other],[invalid],source_char_limit=base)
        self.assertEqual(context['correction_context'],[]);self.assertEqual(context['support_context_coverage']['omitted_ids'],[other['id']])
        context=personal.portrait_support_context(p,[*sources,other],[item])
        self.assertEqual(context['correction_context'],[other]);self.assertNotIn('derivative',context['correction_context'][0])

    def test_oversized_current_correction_refuses_support_before_any_provider_call(self):
        sources,p=contextual_portrait();large=source('memory:large','Correction actuelle : '+('texte '*12000))
        p['coverage']['pending']=[large['id']];seen=[];adapter=personal.Adapter(job(),lambda *args:seen.append(args))
        with self.assertRaises(personal.GenerationFailure) as caught:personal.verify_portrait_support(p,[*sources,large],adapter)
        self.assertEqual(caught.exception.code,'quota_exceeded');self.assertEqual(caught.exception.validation_codes,('support_context_budget',))
        self.assertEqual(seen,[]);self.assertEqual(adapter.calls,0);self.assertEqual(p['coverage']['pending'],[large['id']])

    def test_large_optional_source_set_reports_omissions_without_claiming_full_context_review(self):
        sources,p=contextual_portrait();optional=[source('event:'+str(i),'Une rencontre agréable '+str(i)+'.') for i in range(1000)]
        saved=copy.deepcopy(p);context=personal.portrait_support_context(p,[*sources,*optional])
        included=context['support_context_coverage']['included_ids'];omitted=context['support_context_coverage']['omitted_ids']
        self.assertTrue(included);self.assertTrue(omitted);self.assertEqual(len(included)+len(omitted),1000)
        self.assertFalse(set(included)&set(omitted));self.assertEqual(p,saved)

    def test_declared_life_context_matches_current_worker_contract_without_civil_identity(self):
        context={'age':27,'gender':'self_described','gender_description':'  Mes propres mots  ',
                 'occupation':'  Un métier déclaré  ','first_name':'PRIVATE_FIRST','last_name':'PRIVATE_LAST',
                 'birth_date':'1999-01-01','name':'PRIVATE_NAME'}
        self.assertEqual(personal.safe_life_context({'self_context':context}),{
            'age':27,'gender':'self_described','gender_description':'Mes propres mots','occupation':'Un métier déclaré'})
        self.assertEqual(personal.safe_life_context({}),{'age':None,'gender':'unspecified','gender_description':'','occupation':''})
        for age in (True,-1,131,27.5,'27',None):
            with self.subTest(age=age):self.assertIsNone(personal.declared_life_context({'age':age})['age'])
        for age in (0,27.0,130):
            with self.subTest(age=age):self.assertEqual(personal.declared_life_context({'age':age})['age'],int(age))
        for gender in ('male','female','non_binary','unspecified','invalid'):
            with self.subTest(gender=gender):
                clean=personal.declared_life_context({'gender':gender,'gender_description':'  Accords masculins  '})
                self.assertEqual(clean['gender'],gender if gender!='invalid' else 'unspecified')
                self.assertEqual(clean['gender_description'],'Accords masculins')
        clean=personal.declared_life_context({'gender':'self_described','gender_description':'x'*201,'occupation':'y'*401})
        self.assertEqual(len(clean['gender_description']),200);self.assertEqual(len(clean['occupation']),400)

    def test_contextual_influence_and_interactions_use_personal_proof_without_changing_support(self):
        sources,p=contextual_portrait()
        clean=personal.validate_psychological_map(p['psychological_map'],sources,p['citations'])
        self.assertEqual(clean[0]['status'],'exploring')
        self.assertEqual(clean[0]['influence'],p['psychological_map'][0]['influence'])
        self.assertEqual(clean[0]['interactions'],p['psychological_map'][0]['interactions'])
        self.assertFalse(any(key in clean[0]['influence'] for key in ('score','coefficient','probability')))
        personal.validate_portrait_v2(p,sources)

    def test_contextual_influence_rejects_numeric_unproven_or_oversized_values(self):
        sources,p=contextual_portrait();base=p['psychological_map'][0]['influence']
        invalids=[{**base,'level':.9},{**base,'level':'certain'},
                  {**base,'contexts':['a']*4},{**base,'contexts':['x'*161]},
                  {**base,'contexts':['']},{**base,'rationale':'x'*601},{**base,'rationale':''},
                  {**base,'evidence':[]},{**base,'evidence':[sources[1]['id']]},
                  {**base,'evidence':['like:unknown']},None]
        for invalid in invalids:
            hypotheses=copy.deepcopy(p['psychological_map']);hypotheses[0]['influence']=invalid
            with self.subTest(value=invalid),self.assertRaises(ValueError):
                personal.validate_psychological_map(hypotheses,sources,p['citations'])
        bare=copy.deepcopy(sources);bare[0]['explicit']=False
        with self.assertRaises(ValueError):personal.validate_psychological_map(p['psychological_map'],bare,p['citations'])

    def test_unknown_without_influence_evidence_has_no_context_and_no_unsupported_prose(self):
        sources,p=contextual_portrait();h=p['psychological_map'][0]
        h['influence']={'level':'unknown','contexts':[],'rationale':'Une cause cachée inventée.','evidence':[]}
        personal.validate_portrait_v2(p,sources)
        self.assertEqual(h['influence']['rationale'],'Une cause cachée inventée.')  # input is not silently rewritten
        self.assertEqual(p['psychological_map'][0]['influence'],{'level':'unknown','contexts':[],
                         'rationale':'Influence à préciser.','evidence':[]})
        self.assertFalse(any(path.endswith('.influence') for path,_,_ in personal.portrait_claims(p)))
        p['psychological_map'][0]['influence']['contexts']=['Un contexte non étayé']
        with self.assertRaises(ValueError):personal.validate_portrait_v2(p,sources)

    def test_interactions_require_another_active_uncontradicted_target_and_proof_of_both(self):
        sources,p=contextual_portrait();base=p['psychological_map'][0]['interactions'][0]
        invalids=[{**base,'pattern_id':'rumination'},{**base,'pattern_id':'perfectionism'},
                  {**base,'pattern_id':'unknown'},{**base,'kind':'causes'},
                  {**base,'explanation':'x'*601},{**base,'explanation':''},
                  {**base,'evidence':[sources[0]['id']]},{**base,'evidence':[sources[1]['id']]},
                  {**base,'evidence':[]},{**base,'evidence':['like:unknown']},None]
        for invalid in invalids:
            hypotheses=copy.deepcopy(p['psychological_map']);hypotheses[0]['interactions']=[invalid]
            with self.subTest(value=invalid),self.assertRaises(ValueError):
                personal.validate_psychological_map(hypotheses,sources,p['citations'])
        for interactions in ([base,base],[base]*4,None):
            hypotheses=copy.deepcopy(p['psychological_map']);hypotheses[0]['interactions']=interactions
            with self.subTest(interactions=interactions),self.assertRaises(ValueError):
                personal.validate_psychological_map(hypotheses,sources,p['citations'])
        hypotheses=copy.deepcopy(p['psychological_map']);hypotheses[1].update(status='contradicted',counter_evidence=[sources[1]['id']])
        with self.assertRaises(ValueError):personal.validate_psychological_map(hypotheses,sources,p['citations'])

    def test_contradicted_hypothesis_only_accepts_unknown_influence_and_no_interactions(self):
        sources,p=contextual_portrait();h=p['psychological_map'][0]
        h.update(status='contradicted',counter_evidence=[sources[0]['id']])
        h['interactions']=[]
        with self.assertRaises(ValueError):personal.validate_psychological_map(p['psychological_map'],sources,p['citations'])
        h['influence']={'level':'unknown','contexts':[],'rationale':'Ignored without evidence','evidence':[]}
        clean=personal.validate_psychological_map(p['psychological_map'],sources,p['citations'])
        self.assertEqual(clean[0]['status'],'contradicted');self.assertEqual(clean[0]['interactions'],[])
        h['interactions']=[{'pattern_id':'uncertainty','kind':'alternative','explanation':'Lien possible.',
                            'evidence':[s['id'] for s in sources]}]
        with self.assertRaises(ValueError):personal.validate_psychological_map(p['psychological_map'],sources,p['citations'])

    def test_v3_language_guard_preserves_exact_quotes_and_inversions_but_refuses_ambiguous_pronouns(self):
        sources,p=contextual_portrait();p['analysis_version']=3
        p['summary']='{{prenom}} rapporte « elle m’a répondu ». Il reste un point à comprendre.'
        p['psychological_map'][0]['possible_function']='Il est possible d’explorer cette piste.'
        p['psychological_map'][0]['alternatives']=['Il s’agit peut-être d’une contrainte temporaire.']
        original=copy.deepcopy(p['citations']);personal.validate_portrait_v2(p,sources)
        self.assertEqual(p['citations'],original)
        for text in ('Elle décrit une difficulté.','{{prenom}} précise qu’elle rumine.','La mère pense qu’ils vont répondre.',
                     'Il semble fatigué.','Il y a vécu deux ans.','Une réaction :elle attend.','Il semble qu’elle attende.'):
            broken=copy.deepcopy(p);broken['summary']=text
            with self.subTest(text=text),self.assertRaisesRegex(ValueError,'portrait_language'):
                personal.validate_portrait_v2(broken,sources)
        legacy=copy.deepcopy(p);legacy.pop('analysis_version');legacy['summary']='Elle décrit une difficulté.'
        personal.validate_portrait_v2(legacy,sources)
        for version in (None,2,4):
            broken=copy.deepcopy(p);broken['analysis_version']=version
            with self.subTest(version=version),self.assertRaises(ValueError):personal.validate_portrait_v2(broken,sources)

    def test_v3_language_checks_contextual_explanations_and_direct_declared_gender(self):
        sources,p=contextual_portrait();p.update(analysis_version=3,identity_context={'gender':'male'})
        for field in ('rationale','explanation'):
            broken=copy.deepcopy(p)
            value=broken['psychological_map'][0]['influence'] if field=='rationale' else broken['psychological_map'][0]['interactions'][0]
            value[field]='Elle pourrait avoir une autre réaction.'
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'portrait_language'):
                personal.validate_portrait_v2(broken,sources)
        for gender,wrong,right in [('male','créative','créatif'),('female','attentif','attentive')]:
            broken=copy.deepcopy(p);broken['identity_context']={'gender':gender}
            broken['summary']='{{prenom}} se décrit comme très '+wrong+'.'
            with self.subTest(gender=gender),self.assertRaisesRegex(ValueError,'portrait_gender'):
                personal.validate_portrait_v2(broken,sources)
            broken['summary']='{{prenom}} se décrit comme plutôt '+right+'.'
            personal.validate_portrait_v2(broken,sources)
        p['summary']='{{prenom}} se dit fatiguée.'
        with self.assertRaisesRegex(ValueError,'portrait_gender'):personal.validate_portrait_v2(p,sources)
        p['summary']='{{prenom}} décrit sa mère comme créative. « {{prenom}} est créative » reste une citation.'
        personal.validate_portrait_v2(p,sources)

    def test_explicit_agreement_preferences_are_preserved_for_all_genders_and_checked_semantically(self):
        sources,p=contextual_portrait();p.update(analysis_version=3,summary='{{prenom}} se dit fatigué.')
        for gender in ('non_binary','female'):
            with self.subTest(gender=gender):
                identity={'age':30,'gender':gender,'gender_description':'NB, accords masculins','occupation':''}
                p['identity_context']=identity.copy()
                personal.validate_portrait_v2(p,sources)
                self.assertEqual(p['identity_context'],identity)
                received=[]
                def semantic_rejection(_,payload):
                    data=json.loads(payload['user_prompt']);received.append(data)
                    return {'content':json.dumps({'checks':[{'path':item['path'],'supported':item['path']!='summary'} for item in data['claims']]})}
                with self.assertRaises(ValueError):
                    personal.verify_portrait_support(p,sources,personal.Adapter(job(),semantic_rejection),identity)
                self.assertEqual(received[0]['current_declared_context']['gender_description'],'NB, accords masculins')

    def test_contextual_claims_are_semantically_checked_against_current_identity_and_both_sources(self):
        sources,p=contextual_portrait();calls=[]
        def proxy(_,payload):
            data=json.loads(payload['user_prompt']);calls.append((payload,data))
            return {'content':json.dumps({'checks':[{'path':item['path'],'supported':True} for item in data['claims']]})}
        adapter=personal.Adapter(job(),proxy)
        personal.verify_portrait_support(p,sources,adapter,{'age':31,'gender':'male','occupation':'Métier actuel'})
        payload,data=calls[0];claims={item['path']:item for item in data['claims']}
        self.assertTrue(claims['psychological_map.0.influence']['hypothesis'])
        self.assertTrue(claims['psychological_map.0.influence']['claim'].startswith('Influence possible estimée, à préciser : high'))
        self.assertTrue(claims['psychological_map.0.interaction.0']['hypothesis'])
        self.assertEqual(claims['psychological_map.0.interaction.0']['source_ids'],[s['id'] for s in sources])
        self.assertEqual(data['current_declared_context'],{'age':31,'gender':'male','gender_description':'','occupation':'Métier actuel'})
        self.assertIn('solidité des indices',payload['system_prompt']);self.assertIn('contre-exemples',payload['system_prompt'])
        def reject(_,payload):
            checks=json.loads(payload['user_prompt'])['claims']
            return {'content':json.dumps({'checks':[{'path':item['path'],'supported':not item['path'].endswith('.influence')} for item in checks]})}
        original=copy.deepcopy(p['psychological_map'])
        personal.verify_portrait_support(p,sources,personal.Adapter(job(),reject))
        self.assertNotIn('influence',p['psychological_map'][0])
        self.assertEqual(p['psychological_map'][0]['interactions'],original[0]['interactions'])
        self.assertEqual(p['psychological_map'][1],original[1])

    def test_generated_v3_identity_is_authorized_by_current_job_before_language_validation(self):
        s=source('profile:user','Je me décris comme créatif.');j=job('portrait',s)
        j['self_context']={'age':30,'gender':'male','occupation':'Métier actuel','first_name':'PRIVATE_FIRST','birth_date':'1996-01-01'}
        j['source_derivatives']=[derivative(s)];calls=[]
        def proxy(_,payload):
            data=json.loads(payload['user_prompt']);calls.append((payload,data))
            if payload['stage']=='portrait':
                p=portrait(s);p.update(analysis_version=99,identity_context={'gender':'female'})
                p['summary']='{{prenom}} est créatif.'
                return {'content':json.dumps(p)}
            return {'content':json.dumps({'checks':[{'path':c['path'],'supported':True} for c in data['claims']]})}
        result=personal.generate(j,proxy)['portrait']
        self.assertEqual(result['analysis_version'],3)
        self.assertEqual(result['identity_context'],{'age':30,'gender':'male','gender_description':'','occupation':'Métier actuel'})
        self.assertEqual(calls[0][1]['life_context'],result['identity_context'])
        self.assertEqual(calls[1][1]['current_declared_context'],result['identity_context'])
        self.assertNotIn('PRIVATE_FIRST',calls[0][0]['user_prompt']);self.assertNotIn('1996-01-01',calls[0][0]['user_prompt'])
        def wrong_gender(_,payload):
            p=portrait(s);p.update(summary='{{prenom}} est créative.',identity_context={'gender':'female'})
            return {'content':json.dumps(p)}
        with self.assertRaises(personal.GenerationFailure):personal.generate(j,wrong_gender)

    def test_semantic_filter_preserves_verified_facts_and_removes_only_rejected_optional_sections(self):
        sources,p=contextual_portrait();sid=sources[0]['id']
        p['axes'].append({**copy.deepcopy(p['axes'][0]),'why':'UNSUPPORTED axis'})
        p['strengths']=[{'text':'Une ressource à conserver.','evidence':[sid]},
                        {'text':'UNSUPPORTED strength','evidence':[sid]}]
        p['current_state']={'summary':'UNSUPPORTED current state','evidence':[sid]}
        p['evolution']={'summary':'Une évolution validée.','evidence':[sid]}
        p['reading_preferences']={'preferred':[{'label':'Un thème validé','evidence':[sid]},
                                               {'label':'UNSUPPORTED preference','evidence':[sid]}],
                                  'avoid':[{'label':'UNSUPPORTED avoid','evidence':[sid]}]}
        personal.validate_portrait_v2(p,sources);original=copy.deepcopy(p);calls=[]
        rejected={'axes.1','strengths.1','current_state','reading_preferences.preferred.1',
                  'reading_preferences.avoid.0','psychological_map.0.influence','psychological_map.1.function'}
        def proxy(_,payload):
            calls.append(payload);claims=json.loads(payload['user_prompt'])['claims']
            return {'content':json.dumps({'checks':[{'path':x['path'],'supported':x['path'] not in rejected} for x in claims]})}
        personal.verify_portrait_support(p,sources,personal.Adapter(job(),proxy))
        self.assertEqual(len(calls),1)
        self.assertEqual(p['summary'],original['summary']);self.assertEqual(p['summary_evidence'],original['summary_evidence'])
        self.assertEqual(p['axes'],original['axes'][:1]);self.assertEqual(p['strengths'],original['strengths'][:1])
        self.assertNotIn('current_state',p);self.assertEqual(p['evolution'],original['evolution'])
        self.assertEqual(p['reading_preferences'],{'preferred':original['reading_preferences']['preferred'][:1],'avoid':[]})
        self.assertEqual([h['pattern_id'] for h in p['psychological_map']],['rumination'])
        self.assertEqual(p['psychological_map'][0]['evidence'],original['psychological_map'][0]['evidence'])
        self.assertNotIn('influence',p['psychological_map'][0]);self.assertEqual(p['psychological_map'][0]['interactions'],[])
        self.assertEqual(p['citations'],original['citations']);self.assertEqual(p['coverage'],original['coverage'])
        self.assertNotIn('UNSUPPORTED',json.dumps(p))
        personal.validate_portrait_v2(p,sources)

    def test_semantic_filter_drops_failed_links_and_links_to_removed_hypotheses_without_rewriting_others(self):
        sources,p=contextual_portrait();first,second=p['psychological_map']
        third=copy.deepcopy(second);third['pattern_id']='overcontrol';p['psychological_map'].append(third)
        first['interactions'].append({**copy.deepcopy(first['interactions'][0]),'pattern_id':'overcontrol'})
        second['interactions']=[{'pattern_id':'rumination','kind':'alternative','explanation':'Une autre lecture resterait possible.',
                                 'evidence':[s['id'] for s in sources]}]
        personal.validate_portrait_v2(p,sources);original=copy.deepcopy(p)
        rejected={'psychological_map.0.interaction.0','psychological_map.2.observation'}
        def proxy(_,payload):
            claims=json.loads(payload['user_prompt'])['claims']
            return {'content':json.dumps({'checks':[{'path':x['path'],'supported':x['path'] not in rejected} for x in claims]})}
        personal.verify_portrait_support(p,sources,personal.Adapter(job(),proxy))
        self.assertEqual([h['pattern_id'] for h in p['psychological_map']],['rumination','uncertainty'])
        self.assertEqual(p['psychological_map'][0]['interactions'],[])
        self.assertEqual(p['psychological_map'][0]['influence'],original['psychological_map'][0]['influence'])
        self.assertEqual(p['psychological_map'][1],original['psychological_map'][1])
        self.assertEqual(p['citations'],original['citations'])
        personal.validate_portrait_v2(p,sources)

    def test_semantic_filter_requires_true_summary_a_kept_axis_and_a_complete_strict_boolean_report(self):
        s=source();base=portrait(s)
        def checks():return [{'path':path,'supported':True} for path,_,_ in personal.portrait_claims(base)]
        malformed=[]
        for path in ('summary','axes.0'):
            values=checks();next(x for x in values if x['path']==path)['supported']=False;malformed.append(values)
        values=checks();values[0]['supported']=1;malformed.append(values)
        values=checks();values[0]['supported']='true';malformed.append(values)
        values=checks();values[1]['path']=values[0]['path'];malformed.append(values)
        values=checks();values[0]['path']='unknown';malformed.append(values)
        malformed.extend([checks()[:-1],[None,*checks()[1:]],None])
        for values in malformed:
            with self.subTest(report=values),self.assertRaises(ValueError):
                personal.verify_portrait_support(copy.deepcopy(base),[s],personal.Adapter(job(),lambda _,payload:{'content':json.dumps({'checks':values})}))

    def test_semantic_filter_removes_a_question_presupposing_trauma_and_keeps_verified_portrait(self):
        sources,p=contextual_portrait()
        p['uncertainties']=['Quel traumatisme parental explique vos pensées répétées ?',
                            'Qu’est-ce qui change selon les contextes ?']
        personal.validate_portrait_v2(p,sources);original=copy.deepcopy(p);calls=[]
        def proxy(_,payload):
            data=json.loads(payload['user_prompt']);calls.append((payload,data))
            return {'content':json.dumps({'checks':[{'path':x['path'],'supported':x['path']!='uncertainties.0'} for x in data['claims']]})}
        personal.verify_portrait_support(p,sources,personal.Adapter(job(),proxy))
        self.assertEqual(len(calls),1)
        questions=[x for x in calls[0][1]['claims'] if x['path'].startswith('uncertainties.')]
        self.assertEqual(len(questions),2);self.assertTrue(all(x['question'] is True for x in questions))
        self.assertTrue(all(x['source_ids']==[s['id'] for s in sources] for x in questions))
        self.assertIn('ne présuppose aucun vécu, diagnostic, traumatisme',calls[0][0]['system_prompt'])
        self.assertEqual(p['uncertainties'],original['uncertainties'][1:])
        for field in ('summary','summary_evidence','axes','strengths','psychological_map','citations','coverage'):
            self.assertEqual(p[field],original[field])
        personal.validate_portrait_v2(p,sources)

    def test_retry_portrait_uses_prudent_prompt_and_only_one_synthesis_and_one_support_call(self):
        for retry in ({'attempts':2},{'native_candidate':{'rejected':True}}):
            with self.subTest(retry=retry):
                j=job();j.update(retry);j['source_derivatives']=[derivative(j['sources'][0])];calls=[]
                def proxy(_,payload):
                    calls.append(payload);data=json.loads(payload['user_prompt'])
                    result=portrait(j['sources'][0]) if payload['stage']=='portrait' else {
                        'checks':[{'path':x['path'],'supported':True} for x in data['claims']]}
                    return {'content':json.dumps(result)}
                personal.generate(j,proxy)
                self.assertEqual([x['stage'] for x in calls],['portrait','support'])
                self.assertIn('Privilégie 1–3 axes déclarés et 0–3 pistes prudentes',calls[0]['system_prompt'])
                self.assertIn('sans inférer une personnalité depuis une préférence de lecture',calls[0]['system_prompt'])

    def test_psychological_map_requires_personal_evidence_and_independent_support(self):
        s=source('profile:user','Je pense souvent en boucle.')
        h={'pattern_id':'rumination','status':'supported','observation':'La personne décrit des pensées répétées.',
           'possible_function':'Une recherche de solution serait à explorer.','alternatives':['Une difficulté ponctuelle.'],
           'next_question':'Quand cela vous aide-t-il ?','evidence':[s['id'],s['id']],'counter_evidence':[]}
        quote={'source_id':s['id'],'revision':1,'quote':s['fields']['self_description']}
        clean=personal.validate_psychological_map([h],[s],[quote])
        self.assertEqual(clean[0]['status'],'exploring')
        for invalid in ({**h,'pattern_id':'unknown'},{**h,'status':'contradicted'}):
            with self.assertRaises(ValueError):personal.validate_psychological_map([invalid],[s],[quote])
        bare={**s,'explicit':False}
        with self.assertRaises(ValueError):personal.validate_psychological_map([h],[bare],[quote])

    def test_duplicated_personal_narrative_is_not_independent_despite_different_identity_fields(self):
        text='Je repense longtemps à mes décisions. Je cherche ensuite une garantie avant d’avancer.'
        first=source('profile:user',text);first['fields']['occupation']='Un métier'
        second=source('person:manager',text);second['fields']['display_name']='Un proche'
        sources=[first,second]
        quotes=[{'source_id':first['id'],'revision':1,'quote':'Je repense longtemps à mes décisions.'},
                {'source_id':second['id'],'revision':1,'quote':'Je cherche ensuite une garantie avant d’avancer.'}]
        _,p=contextual_portrait();h=p['psychological_map'][0]
        h.pop('influence');h.pop('interactions');h.update(status='supported',evidence=[s['id'] for s in sources])
        self.assertEqual(personal.validate_psychological_map([h],sources,quotes)[0]['status'],'exploring')

    def test_self_description_is_supported_evidence_but_occupation_is_not_a_trait(self):
        s=source('profile:user','Je me décris comme sensible aux conflits et je rumine parfois.')
        j=job('portrait',s)
        self.assertEqual(personal.canonical_sources(j),[s])
        p=portrait(s);p['axes'][0].update(claim_type='declaration',confidence=.7)
        personal.validate_portrait_v2(p,[s])
        occupation=copy.deepcopy(s);occupation['fields']={'occupation':'Architecte'};occupation['text']='occupation: Architecte'
        self.assertFalse(personal.is_personal_quote(occupation,'Architecte'))


    def test_worker_usage_contract_accumulates_observed_tokens_and_cost(self):
        adapter=personal.Adapter(job('story'),lambda _,payload:{
            'content':'{}','usage':{'input_tokens':123,'output_tokens':456,'observed_cost':.001}})
        adapter(system_prompt='test',user_prompt='test')
        adapter(system_prompt='test',user_prompt='test')
        self.assertEqual((adapter.input_tokens,adapter.output_tokens),(246,912))
        self.assertAlmostEqual(adapter.observed_cost,.002)

    def test_shared_provenance_fixtures_match_worker_contract(self):
        fixtures=json.loads((pathlib.Path(__file__).parent/'fixtures/portrait-provenance-v2.json').read_text())
        for fixture in fixtures:
            with self.subTest(name=fixture['name']):
                if fixture['valid']:personal.validate_portrait_v2(fixture['portrait'],fixture['sources'])
                else:
                    with self.assertRaises(ValueError):personal.validate_portrait_v2(fixture['portrait'],fixture['sources'])


    def test_lost_completion_response_replays_same_payload_without_regenerating(self):
        calls=[];j=job();result={'portrait':portrait(source()),'lease_token':'secret'}
        def internal(path,payload):
            calls.append((path,payload))
            if path=='/claim':return {'jobs':[j] if len(calls)==1 else []}
            if path.endswith('/complete') and len([c for c in calls if c[0].endswith('/complete')])==1:raise runner.TransportFailure('safe timeout')
            return {'ok':True}
        import contextlib,io
        with patch.object(runner,'internal',side_effect=internal),patch.object(runner,'generate',return_value=result) as generate,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(),0)
        self.assertEqual(generate.call_count,1)
        completions=[payload for path,payload in calls if path.endswith('/complete')]
        self.assertEqual(completions,[result,result])

    def test_partial_extraction_coverage_is_honest_when_budget_is_nearly_spent(self):
        first=source();second=source('event:later','Une longue source restant à analyser.');j=job('portrait',first)
        j['sources'].append(second);j['source_manifest'][second['id']]={k:second.get(k) for k in ('revision','hash','type','explicit')}
        j['source_derivatives']=[derivative(first)];j['provider_calls']=18;calls=[]
        def proxy(_,payload):
            data=json.loads(payload['user_prompt']);calls.append(payload['stage'])
            if payload['stage']=='portrait':
                result=portrait(first);result['coverage']=data['coverage']
            else:result={'checks':[{'path':c['path'],'supported':True} for c in data['claims']]}
            return {'content':json.dumps(result)}
        result=personal.generate(j,proxy)
        self.assertEqual(calls,['portrait','support']);self.assertEqual(result['portrait']['coverage']['pending'],[second['id']])


    def test_configured_token_reservation_is_checked_before_provider_send(self):
        j=job('story');j['token_budget']=100;j['reserved_tokens']=90;sent=[]
        adapter=personal.Adapter(j,lambda _,p:sent.append(p))
        with self.assertRaises(RuntimeError):adapter(system_prompt='x',user_prompt='y',max_tokens=128)
        self.assertEqual(sent,[]);self.assertEqual(adapter.error_code,'quota_exceeded')

    def test_zero_like_portrait_extracts_then_synthesizes_and_checks_semantic_support(self):
        j=job();calls=[]
        def proxy(_,p):
            data=json.loads(p['user_prompt']);calls.append(p)
            if p['stage']=='extraction':out={'sources':[derivative(s) for s in data['sources']]}
            elif p['stage']=='portrait':out=portrait(j['sources'][0])
            else:out={'checks':[{'path':x['path'],'supported':True} for x in data['claims']]}
            return {'content':json.dumps(out)}
        result=personal.generate(j,proxy)
        self.assertEqual([c['stage'] for c in calls],['extraction','portrait','support'])
        self.assertNotIn('fictional_character',result['portrait'])
        self.assertEqual(len(result['source_derivatives']),1)
        self.assertTrue(all(c['temperature']==.2 for c in calls))

    def test_unchanged_derivatives_avoid_reextracting_sources(self):
        j=job();j['source_derivatives']=[derivative(j['sources'][0])];stages=[]
        def proxy(_,p):
            stages.append(p['stage']);data=json.loads(p['user_prompt'])
            return {'content':json.dumps(portrait(j['sources'][0]) if p['stage']=='portrait' else {'checks':[{'path':x['path'],'supported':True} for x in data['claims']]})}
        self.assertEqual(personal.generate(j,proxy)['source_derivatives'],[])
        self.assertEqual(stages,['portrait','support'])

    def test_semantic_checker_can_reject_a_real_quote_that_does_not_support_claim(self):
        j=job();j['source_derivatives']=[derivative(j['sources'][0])]
        def proxy(_,p):
            data=json.loads(p['user_prompt'])
            out=portrait(j['sources'][0]) if p['stage']=='portrait' else {'checks':[{'path':x['path'],'supported':False} for x in data['claims']]}
            return {'content':json.dumps(out)}
        with self.assertRaises(personal.GenerationFailure):personal.generate(j,proxy)

    def test_bare_like_confidence_and_claim_type_are_enforced(self):
        s=source('like:article','Intérêt de lecture pour la colère',False)
        p=portrait(s);personal.validate_portrait_v2(p,[s])
        p['axes'][0]['confidence']=.26
        with self.assertRaises(ValueError):personal.validate_portrait_v2(p,[s])
        p['axes'][0]['confidence']=.2;p['axes'][0]['claim_type']='declaration'
        with self.assertRaises(ValueError):personal.validate_portrait_v2(p,[s])

    def test_revised_removed_and_fabricated_citations_are_rejected(self):
        s=source();p=portrait(s)
        for field,value in [('revision',99),('quote','Une phrase inventée'),('source_id','memory:removed')]:
            broken=copy.deepcopy(p);broken['citations'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):personal.validate_portrait_v2(broken,[s])

    def test_unsourced_strength_summary_or_state_cannot_survive_deletion(self):
        s=source();p=portrait(s)
        for path in ['strengths','summary_evidence','current_state']:
            broken=copy.deepcopy(p)
            broken[path]=[{'text':'Courage','evidence':['memory:gone']}] if path=='strengths' else ['memory:gone'] if path=='summary_evidence' else {'summary':'État actuel','evidence':['memory:gone']}
            with self.subTest(path=path),self.assertRaises(ValueError):personal.validate_portrait_v2(broken,[s])


    def test_catalogue_extracts_are_exact_substrings_with_correct_offsets(self):
        root=pathlib.Path(__file__).parents[2]
        for file in (root/'content/posts').rglob('*.md'):
            if file.name=='_index.md':continue
            import yaml
            _,front,text=file.read_text().split('---',2);meta=yaml.safe_load(front)
            if meta.get('draft'):continue
            data=build_examples.extractive_metadata(text,meta)
            for q in data['summary_evidence']:self.assertEqual(text[q['start']:q['end']],q['quote'])
            self.assertEqual(data['content_hash'],hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':unittest.main()


class PsychologyGuideTests(unittest.TestCase):
    def test_every_concept_remains_available_without_full_interview_prompt(self):
        guide=personal.psychology_guide([source(text='Je rumine, je dors mal et ma concentration diminue.')])
        self.assertEqual({row[0] for row in guide['index']},personal.PSYCHOLOGY_PATTERN_IDS)
        self.assertEqual(len(guide['patterns']),24)
        self.assertIn('rumination',{p['id'] for p in guide['patterns']})
        self.assertLess(len(json.dumps(guide,ensure_ascii=False)),40000)
