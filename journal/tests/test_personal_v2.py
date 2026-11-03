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
    kind=sid.split(':')[0];field={'memory':'text','person':'relationship_text','event':'description','feedback':'correction','like':'reason' if explicit else 'title'}[kind]
    original=text;text=field+': '+text
    return {'id':sid,'type':kind,'revision':revision,'hash':hashlib.sha256(text.encode()).hexdigest(),'text':text,'explicit':explicit,'fields':{field:original},'fiction_use':'abstract'}


def portrait(s):
    return {'schema_version':2,'summary':'Un intérêt déclaré pour la colère.','summary_evidence':[s['id']],
            'axes':[{'label':'Colère','why':'Un intérêt de lecture formulé.','confidence':.2,'claim_type':'interest','evidence':[s['id']]}],
            'strengths':[],'uncertainties':['Un intérêt ne décrit pas toute la personne.'],
            'citations':[{'source_id':s['id'],'revision':s['revision'],'quote':s['text']}],
            'coverage':{'processed':[s['id']],'pending':[]}}


def job(kind='portrait',s=None):
    s=s or source()
    return {'schema_version':2,'id':'j','lease_token':'secret','kind':kind,'source_version':3,'memory_version':2,'memory':{'version':2},
            'sources':[s],'source_manifest':{s['id']:{k:s.get(k) for k in ('revision','hash','type','explicit')}},
            'portrait':portrait(s),'characters':[],'missing_slots':[1,2,3],
            'character_id':'char-a','character':{'id':'char-a','slot':1,'identity':{'name':'Alix','description':'Une relieuse fictive.'},'narrative_state':{}},
            'canon_version':1,'expected_parent_id':None,'history':[],'inspirations':[]}


def derivative(s):
    return {'source_id':s['id'],'revision':s['revision'],'source_hash':s['hash'],'analysis_version':2,
            'derivative':{'signals':[{'kind':'interest','text':'Intérêt pour la colère','quote':s['text'],'temporal':'non précisé'}]}}


def body():
    paragraph='Alix observe une possibilité fictive, prend son temps et réfléchit sans imposer de conclusion. '
    return '\n\n'.join('## '+title+'\n'+paragraph*20 for title in ['La scène','Une autre lecture','Une pratique','La suite'])+'\n\nAlix retrouve finalement la lettre et choisit de la conserver.'


def review(issues=None):
    return {'checked':['identity','relationships','chronology','acquisitions','attribution','constraints','continuity','conclusion'],'issues':issues or []}


def story_outputs():
    text=body()
    return [json.dumps({'subject':'Nommer une émotion','scene':'Un atelier','continuity':'Alix relie des livres','ending':'La lettre retrouvée','practice':'Observer une émotion'}),
            'TITRE_SEO: La lettre retrouvée\nMETA_DESCRIPTION: Un récit fictif\nSLUG: lettre\nTAGS: émotion\n---\n'+text,
            json.dumps(review()),json.dumps({'summary':'Fiction : Alix retrouve finalement la lettre et choisit de la conserver.',
                'narrative_delta':{'events':[{'text':'Alix observe','quote':'Alix observe une possibilité fictive'}],'relationships':[],'open_questions':[],
                                   'resolution':{'text':'La lettre est conservée','quote':'Alix retrouve finalement la lettre et choisit de la conserver.'}}}),
            json.dumps({'reflection_questions':['Que vous évoque cette scène ?'],'practice':'Noter une émotion, si vous en avez envie.'})]


class V2Tests(unittest.TestCase):
    def test_shared_provenance_fixtures_match_worker_contract(self):
        fixtures=json.loads((pathlib.Path(__file__).parent/'fixtures/portrait-provenance-v2.json').read_text())
        for fixture in fixtures:
            with self.subTest(name=fixture['name']):
                if fixture['valid']:personal.validate_portrait_v2(fixture['portrait'],fixture['sources'])
                else:
                    with self.assertRaises(ValueError):personal.validate_portrait_v2(fixture['portrait'],fixture['sources'])

    def test_an_old_resolution_is_retrieved_alongside_its_older_open_thread(self):
        j=job('story');j['history']=[{'id':str(i),'character_id':'char-a','summary':'Atelier ordinaire indépendant','narrative_delta':{},'episode_number':i} for i in range(40)]
        j['history'][0].update(summary='Une lettre disparue reste mystérieuse.',narrative_delta={'open_questions':[{'text':'Où est la lettre ?','quote':'lettre'}]})
        j['history'][8].update(summary='La lettre est retrouvée et la question est résolue.',narrative_delta={'resolution':{'text':'La lettre est conservée','quote':'lettre'}})
        selected=personal.story_context_v2(j)['history_retrieved']
        self.assertTrue(any(h['id']=='0' for h in selected));self.assertTrue(any(h['id']=='8' for h in selected))

    def test_only_restricted_event_can_prepare_honest_exploratory_characters(self):
        s=source('event:protected','PRIVATE SCENE');s['fiction_use']='understanding_only';j=job('characters',s)
        output={'characters':[{'slot':slot,'identity':{'name':name,'description':'Identité fictive indépendante.'},
                'facet':{'label':'Piste exploratoire','description':'Proposition fictive, sans interpréter votre vécu.','evidence':[],'exploratory':True},'narrative_state':{}}
                for slot,name in [(1,'Alix'),(2,'Sam'),(3,'Noa')]]}
        calls=[];result=personal.generate(j,lambda _,p:(calls.append(p) or {'content':json.dumps(output)}))
        self.assertEqual(len(result['characters']),3);self.assertNotIn('PRIVATE SCENE',calls[0]['user_prompt'])
        self.assertEqual(json.loads(calls[0]['user_prompt'])['allowed_evidence_ids'],[])
        output['characters'][0]['facet'].pop('exploratory')
        with self.assertRaises(ValueError):personal.validate_characters_v2(output['characters'],j)

    def test_withdrawn_facet_prose_never_reenters_a_story_brief(self):
        j=job('story');j['character']['facet']={'label':'REMOVED PRIVATE SENTINEL','description':'REMOVED PRIVATE SENTINEL','evidence':['event:gone']}
        context=personal.story_context_v2(j)
        self.assertNotIn('REMOVED PRIVATE SENTINEL',json.dumps(context));self.assertEqual(context['character']['identity']['name'],'Alix')

    def test_migration_identity_must_be_anchored_in_preserved_fiction(self):
        j=job('characters');j['characters']=[{'id':'legacy','slot':1,'identity':{}}];j['missing_slots']=[1]
        j['character_histories']={'legacy':[{'id':'old-story','body':'Camille retrouve sa lettre.','summary':'Camille retrouve sa lettre.'}]}
        character={'slot':1,'identity':{'name':'Camille','description':'Identité fictive conservée.','archive_evidence':[{'story_id':'old-story','quote':'Camille retrouve sa lettre.'}]},'facet':{'label':'Une perspective','description':'Exploration','evidence':['memory:user']},'narrative_state':{}}
        personal.validate_characters_v2([character],j)
        character['identity']['name']='Sam'
        with self.assertRaisesRegex(ValueError,'coherence_error'):personal.validate_characters_v2([character],j)

    def test_lost_completion_response_replays_same_payload_without_regenerating(self):
        calls=[];j=job('story');result={'title':'Un titre','lease_token':'secret'}
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

    def test_revised_source_also_invalidates_old_facet_prose(self):
        j=job('story');sid=j['sources'][0]['id'];old=copy.deepcopy(j['source_manifest']);old[sid]['revision']=0
        j['character']['facet']={'label':'OUTDATED INTERPRETATION','description':'OUTDATED INTERPRETATION','evidence':[sid],'source_manifest':old}
        self.assertNotIn('OUTDATED INTERPRETATION',json.dumps(personal.story_context_v2(j)))

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

    def test_three_missing_slots_only_and_existing_identity_stays_stable(self):
        j=job('characters');j['characters']=[{'slot':1,'identity':{'name':'Alix','description':'Identité ancienne conservée'}}];j['missing_slots']=[2,3]
        result={'characters':[{'slot':slot,'identity':{'name':name,'description':'Un milieu inventé distinct, des relations et des ressources.'},'facet':{'label':'Une possibilité','description':'Proposition exploratoire','evidence':['memory:user']},'narrative_state':{}} for slot,name in [(2,'Sam'),(3,'Noa')]]}
        calls=[];out=personal.generate(j,lambda _,p:(calls.append(p) or {'content':json.dumps(result)}))
        self.assertEqual([c['slot'] for c in out['characters']],[2,3]);self.assertIn('Identité ancienne conservée',calls[0]['user_prompt'])
        result['characters'][0]['slot']=1
        with self.assertRaises(ValueError):personal.validate_characters_v2(result['characters'],j)

    def test_understanding_only_does_not_enter_writer_through_sources_portrait_or_quotes(self):
        s=source('event:private','SECRET RESTRICTED SCENE');s['fiction_use']='understanding_only';j=job('story',s)
        j['portrait']['summary']='SECRET RESTRICTED SCENE';j['portrait']['axes'][0]['why']='SECRET RESTRICTED SCENE'
        context=personal.story_context_v2(j)
        self.assertNotIn('SECRET RESTRICTED SCENE',json.dumps(context))

    def test_source_corrections_after_120_chars_are_preserved(self):
        original='Contexte ordinaire. '*30+'Correction importante : je ne vis pas cette situation.'
        j=job('story',source(text=original))
        self.assertIn(original,json.dumps(personal.story_context_v2(j),ensure_ascii=False))

    def test_old_unresolved_event_is_retrieved_after_thirty_episodes(self):
        j=job('story');j['history']=[{'id':str(i),'character_id':'char-a','summary':'Un épisode ordinaire','narrative_delta':{},'episode_number':i} for i in range(35)]
        j['history'][0].update(summary='La lettre disparue attend une réponse.',narrative_delta={'open_questions':[{'text':'Où est la lettre ?','quote':'lettre'}]})
        context=personal.story_context_v2(j)
        self.assertEqual(context['history_count'],35);self.assertIn('La lettre disparue',json.dumps(context));self.assertEqual(len(context['history_recent']),5)

    def test_other_character_history_is_rejected_before_any_model_call(self):
        j=job('story');j['history']=[{'character_id':'char-b','summary':'Autre saga'}]
        with self.assertRaisesRegex(ValueError,'coherence_error'):personal.story_context_v2(j)

    def test_final_complete_body_summary_and_delta_include_ending(self):
        j=job('story');outputs=story_outputs();calls=[]
        result=personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(len(calls),5)
        summary_call=next(c for c in calls if c['stage']=='summary')
        self.assertIn('choisit de la conserver',summary_call['user_prompt'])
        self.assertIn('choisit de la conserver',result['summary'])
        self.assertEqual(calls[1]['temperature'],.7)
        self.assertEqual(calls[1]['max_tokens'],7000)
        personal.validate_narrative_delta(result['narrative_delta'],result['body'])

    def test_targeted_correction_is_rechecked_and_resummary_uses_corrected_final_body(self):
        j=job('story');outputs=story_outputs();outputs[2]=json.dumps(review([{'quote':'Alix retrouve finalement la lettre et choisit de la conserver.','reason':'Choix contraire au plan','replacement':'Alix retrouve finalement la lettre et choisit de la relire.'}]))
        outputs.insert(3,json.dumps(review()))
        outputs[4]=outputs[4].replace('conserver','relire')
        calls=[];result=personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(len(calls),6);self.assertIn('relire',result['body']);self.assertIn('relire',result['summary'])

    def test_summary_delta_cannot_cite_only_the_opening_as_resolution(self):
        delta={'events':[],'relationships':[],'open_questions':[],'resolution':{'text':'Fin','quote':'## La scène'}}
        with self.assertRaises(ValueError):personal.validate_narrative_delta(delta,body())

    def test_cumulative_call_budget_survives_retry(self):
        j=job();j['request']={'llm_calls':20};calls=[]
        with self.assertRaises(personal.GenerationFailure):personal.generate(j,lambda _,p:calls.append(p))
        self.assertEqual(calls,[])

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

    def test_ten_synthetic_profile_fixtures_preserve_sources_and_private_boundaries(self):
        # Covers context construction and validation contracts, not real literary quality.
        fixtures=[('memory:u','Je veux comprendre la colère.'),('person:p','Avec mon ami, je me sens soutenu.'),('event:e','Une réussite professionnelle me rend fier.'),('event:e','Le deuil de 2011 reste important, mais différemment.'),('feedback:f','Le conflit ne me ressemble pas, le style me plaît.'),('memory:u','Je ne veux pas éviter la colère.'),('memory:u','Je souhaite une scène ordinaire sans progrès imposé.'),('event:e','Je prépare une reconversion avec enthousiasme.'),('person:p','Ma relation a changé depuis notre discussion.'),('memory:u','Contexte. '*40+'La correction finale prime sur la première impression.')]
        self.assertEqual(len(fixtures),10)
        for sid,text in fixtures:
            with self.subTest(source=sid,text=text[:30]):
                j=job('story',source(sid,text));ctx=personal.story_context_v2(j)
                self.assertEqual(ctx['sources'][0]['text'],j['sources'][0]['text'])
                self.assertEqual(ctx['character']['id'],'char-a')
                self.assertNotIn('fictional_character',ctx['portrait'])

if __name__=='__main__':unittest.main()
