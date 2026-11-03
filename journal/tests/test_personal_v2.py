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


def guidance(sid='memory:user'):
    result=personal.empty_fiction_guidance()
    result['orientations']=[{'label':'colère','evidence':[sid]}]
    return result


def character_fixture(slot,name,evidence):
    dimensions=[('transports','équipe professionnelle','organisation'),
                ('milieu naturel','voisinage','patience'),
                ('commerce','amitié','humour')][slot-1]
    return {'slot':slot,'identity':{'name':name,'description':name+' mène une vie fictive dans '+dimensions[0]+', avec '+dimensions[2]+'.',
            'story_domain':dimensions[0],'social_setting':dimensions[1],'principal_resource':dimensions[2]},
            'facet':{'label':['Choisir son rythme','Trouver sa place','Oser une réponse'][slot-1],
                     'description':'Proposition fictive, sans interpréter votre vécu.','evidence':evidence,'exploratory':not evidence},
            'narrative_state':{}}


def story_outputs():
    text=body()
    return [json.dumps(guidance()),json.dumps({'subject':'Nommer une émotion','scene':'Un atelier','continuity':'Alix relie des livres','ending':'La lettre retrouvée','practice':'Observer une émotion'}),
            'TITRE_SEO: La lettre retrouvée\nMETA_DESCRIPTION: Un récit fictif\nSLUG: lettre\nTAGS: émotion\n---\n'+text,
            json.dumps(review()),json.dumps({'summary':'Fiction : Alix retrouve finalement la lettre et choisit de la conserver.',
                'narrative_delta':{'events':[{'text':'Alix observe','quote':'Alix observe une possibilité fictive'}],'relationships':[],'open_questions':[],
                                   'resolution':{'text':'La lettre est conservée','quote':'Alix retrouve finalement la lettre et choisit de la conserver.'}}}),
            json.dumps({'reflection_questions':['Que vous évoque cette scène ?'],'practice':'Noter une émotion, si vous en avez envie.'})]


class V2Tests(unittest.TestCase):
    def test_self_description_is_supported_evidence_but_occupation_is_not_a_trait(self):
        s=source('profile:user','Je me décris comme sensible aux conflits et je rumine parfois.')
        j=job('portrait',s)
        self.assertEqual(personal.canonical_sources(j),[s])
        p=portrait(s);p['axes'][0].update(claim_type='declaration',confidence=.7)
        personal.validate_portrait_v2(p,[s])
        occupation=copy.deepcopy(s);occupation['fields']={'occupation':'Architecte'};occupation['text']='occupation: Architecte'
        self.assertFalse(personal.is_personal_quote(occupation,'Architecte'))

    def test_single_character_receives_life_stage_without_civil_identity_or_private_prose(self):
        s=source('profile:user','PRIVATE_NARRATIVE je pense souvent en boucle.')
        j=job('characters',s);j.update(missing_slots=[1],max_characters=1,
            self_context={'age':27,'gender':'male','first_name':'PRIVATE_NAME','birth_date':'1999-01-01','occupation':'PRIVATE_JOB'})
        calls=[];outputs=[json.dumps(guidance(s['id'])),json.dumps({'characters':[character_fixture(1,'Alix',[s['id']])]})]
        result=personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(len(result['characters']),1)
        designer=json.loads(calls[-1]['user_prompt'])
        self.assertEqual(designer['life_context'],{'age_band':'18–29 ans','declared_gender':'male'})
        for marker in ('PRIVATE_NARRATIVE','PRIVATE_NAME','1999-01-01','PRIVATE_JOB'):
            self.assertNotIn(marker,calls[-1]['user_prompt'])
        j['missing_slots']=[1,2]
        with self.assertRaises(ValueError):personal.validate_characters_v2([character_fixture(1,'Alix',[s['id']]),character_fixture(2,'Noa',[s['id']])],j)

    def test_self_description_is_analyzed_but_not_replayed_as_fiction(self):
        s=source('profile:user','PRIVATE_SELF_SCENE Je voudrais éviter les disputes dans les transports.')
        j=job('story',s);outputs=story_outputs();outputs[0]=json.dumps(guidance(s['id']));calls=[]
        personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertIn('PRIVATE_SELF_SCENE',calls[0]['user_prompt'])
        self.assertNotIn('PRIVATE_SELF_SCENE',calls[1]['user_prompt'])
        self.assertNotIn('PRIVATE_SELF_SCENE',calls[2]['user_prompt'])
        with self.assertRaises(ValueError):personal.assert_no_private_reuse(s['fields']['self_description'],j)

    def test_raw_sources_never_reach_fiction_planning_writing_or_character_design(self):
        marker='ATELIER_ZEPHYR_1998_PERSONNE_TEMOIN'
        s=source('event:e','Chez mon père, '+marker+', j’ai montré un dessin imparfait à Nantes.')
        j=job('story',s);j['portrait']['summary']=marker;j['portrait']['axes'][0].update(label='Rapport à l’imperfection '+marker,why=marker)
        j['portrait']['strengths']=[{'text':marker,'evidence':[s['id']]}]
        j['portrait']['current_state']={'summary':marker,'evidence':[s['id']]}
        j['inspirations']=[{'body':marker,'title':marker,'character':marker,'keywords':['perfectionnisme']}]
        outputs=story_outputs();outputs[0]=json.dumps(guidance(s['id']));calls=[]
        personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertIn(marker,calls[0]['user_prompt']);self.assertEqual(calls[0]['stage'],'orientation')
        for call in calls[1:]:self.assertNotIn(marker,call['user_prompt'])
        j['kind']='characters';result={'characters':[character_fixture(i,name,[s['id']]) for i,name in [(1,'Alix'),(2,'Sam'),(3,'Noa')]]};calls=[]
        personal.generate(j,lambda _,p:(calls.append(p) or {'content':json.dumps(guidance(s['id']) if p['stage']=='orientation' else result)}))
        self.assertIn(marker,calls[0]['user_prompt']);self.assertNotIn(marker,calls[-1]['user_prompt'])
        self.assertIn('missing_slots',calls[-1]['system_prompt'])

    def test_orientation_rejects_arbitrary_personal_prose_before_any_fiction_call(self):
        result=guidance();result['orientations'][0]['label']='Le père peint dans son atelier privé'
        calls=[]
        with self.assertRaises(personal.GenerationFailure) as error:
            personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':json.dumps(result)}))
        self.assertEqual(error.exception.stage,'orientation');self.assertEqual(len(calls),1)

    def test_protected_claims_are_also_excluded_from_analytical_orientation(self):
        j=job('story');protected=source('event:protected','SECRET_PROTECTED_SCENE');protected['fiction_use']='understanding_only';j['sources'].append(protected)
        j['portrait']['summary']='SECRET_PROTECTED_SCENE';j['portrait']['summary_evidence']=[protected['id']]
        j['portrait']['axes'].append({'label':'SECRET_PROTECTED_SCENE','why':'SECRET_PROTECTED_SCENE','confidence':.2,'evidence':[protected['id']]})
        j['portrait']['citations'].append({'source_id':protected['id'],'revision':1,'quote':protected['text']})
        context=personal.fiction_analysis_context(j)
        self.assertNotIn('SECRET_PROTECTED_SCENE',json.dumps(context))
        self.assertEqual(len(context['sources']),1)

    def test_unmapped_reading_wish_gets_complete_analytical_review_without_blocking(self):
        original='Contexte préalable. '*20+'PREFERENCE_LIBRE : pas de téléphone dans cette histoire.'
        j=job('story',source(text=original));outputs=story_outputs();g=guidance();g['unrepresented_constraints']=True;outputs[0]=json.dumps(g);calls=[]
        result=personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertTrue(result['body']);self.assertEqual(len(calls),6)
        for call in calls:
            data=json.loads(call['user_prompt'])
            if call['stage'] in ('orientation','review'):self.assertIn(original,call['user_prompt'])
            else:self.assertNotIn('PREFERENCE_LIBRE',call['user_prompt'])
        review_call=next(c for c in calls if c['stage']=='review')
        self.assertIn('analytical_preferences',json.loads(review_call['user_prompt']))

    def test_new_character_protocol_requires_distinct_domains_relations_and_resources(self):
        j=job('characters');proposals=[character_fixture(i,name,['memory:user']) for i,name in [(1,'Alix'),(2,'Sam'),(3,'Noa')]]
        personal.validate_characters_v2(proposals,j)
        for key in ('story_domain','social_setting','principal_resource'):
            copied=copy.deepcopy(proposals);copied[1]['identity'][key]=copied[0]['identity'][key]
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'coherence_error'):personal.validate_characters_v2(copied,j)
        copied=copy.deepcopy(proposals);copied[1]['facet']['label']='perspective'
        with self.assertRaises(ValueError):personal.validate_characters_v2(copied,j)

    def test_partial_preparation_cannot_reuse_existing_dimensions_and_legacy_is_preserved(self):
        existing=character_fixture(1,'Alix',['memory:user']);existing['id']='old-character'
        j=job('characters');j['characters']=[existing];j['missing_slots']=[2,3]
        proposals=[character_fixture(2,'Sam',['memory:user']),character_fixture(3,'Noa',['memory:user'])]
        personal.validate_characters_v2(proposals,j)
        duplicate=copy.deepcopy(proposals);duplicate[0]['identity']['story_domain']=existing['identity']['story_domain']
        with self.assertRaisesRegex(ValueError,'coherence_error'):personal.validate_characters_v2(duplicate,j)
        legacy=copy.deepcopy(existing)
        for key in ('story_domain','social_setting','principal_resource'):legacy['identity'].pop(key)
        j['characters']=[legacy];before=copy.deepcopy(legacy)
        personal.validate_characters_v2(duplicate,j)
        self.assertEqual(j['characters'][0],before)

    def test_existing_identity_and_published_history_are_unchanged_by_projection(self):
        j=job('story');j['character']['identity']={'name':'Alix','description':'Une artiste dans son atelier, avec son père fictif établi.'}
        j['history']=[{'id':'old','character_id':'char-a','summary':'Alix visite son père fictif.','episode_number':1}]
        before=copy.deepcopy(j);context=personal.story_context_v2(j)
        self.assertEqual(j,before);self.assertEqual(context['character']['identity'],j['character']['identity']);self.assertEqual(context['history_recent'],j['history'])

    def test_writer_reference_contains_french_labels_without_english_ids_or_bibliography(self):
        text=json.dumps(personal.writing_reference(),ensure_ascii=False)
        for leaked in ('notice_name','Notice and Name','grounding','Grounding','assertive_limit','source_url','source_title','https://'):self.assertNotIn(leaked,text)
        self.assertIn('Observer et nommer',text)

    def test_v2_guard_never_inherits_historical_format_or_gender_instructions(self):
        calls=[];adapter=personal.Adapter(job('story'),lambda _,p:(calls.append(p) or {'content':'{}'}))
        adapter(system_prompt='Retourne JSON',user_prompt='{}')
        prompt=calls[0]['system_prompt']
        self.assertIn('JOURNAL PERSONNEL V2',prompt);self.assertNotIn('Markdown du relecteur',prompt);self.assertNotIn('épicène',prompt)
        self.assertNotIn('JSON du résumé',prompt)

    def test_editorial_leaks_are_local_blockers_even_in_long_structurally_valid_text(self):
        cases={
          'technical_block':'JSON_RÉSUMÉ:\n{"personnage":"Alix"}',
          'heading':'## JSON_RÉSUMÉ\n{"personnage":"Alix"}',
          'untranslated_jargon':'Alix pratique Notice and name puis Grounding.',
          'editorial_instructions':'La continuité reste intacte, sans progrès forcé.',
          'hidden_technical_content':'<!-- source_version: 3 -->',
          'first_person_narration':'Je rentre chez mon père et raconte ma journée.\n\n'}
        for kind,snippet in cases.items():
            text=snippet+'\n\n'+body() if kind=='first_person_narration' else body()+'\n\n'+snippet+'\n\nAlix referme la porte et reprend sa route.'
            with self.subTest(kind=kind):
                self.assertTrue(personal.editorial_body_issues(text))
                with self.assertRaises(ValueError):personal.validate_story_body({'title':'Une journée'},text)
        with self.assertRaises(ValueError):personal.validate_story_body({'title':'JSON_RÉSUMÉ'},body())
        personal.validate_story_body({'title':'La réponse'},body()+'\n\n« Je préfère partir », dit Alix, avant de fermer la porte.')

    def test_observed_prompt_echoes_cannot_pass_as_story_headings(self):
        for snippet in ('## Une scène incarnée : les papiers sur la table',
                        '## Explications naturelles des pratiques',
                        'Elle utilisait trois pratiques simples, aux noms français.'):
            with self.subTest(snippet=snippet):
                self.assertIn('editorial_instructions',personal.editorial_body_issues(body()+'\n\n'+snippet))
        self.assertNotIn('editorial_instructions',personal.editorial_body_issues(body()+'\n\n## Les papiers sur la table'))

    def test_one_editorial_repair_occurs_before_review_summary_and_publication(self):
        outputs=story_outputs();bad=outputs[2]+'\n\nJSON_RÉSUMÉ:\n{"personnage":"Alix"}\n\nLa continuité reste intacte, sans progrès forcé.';outputs.insert(2,bad);calls=[]
        result=personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','drafting','review','summary','reflection'])
        correction=json.loads(calls[3]['user_prompt'])['correction_structure']
        self.assertIn('technical_block',correction['issues']);self.assertIn('editorial_instructions',correction['issues'])
        self.assertNotIn('JSON_RÉSUMÉ',result['body'])
        self.assertNotIn('JSON_RÉSUMÉ',next(c for c in calls if c['stage']=='summary')['user_prompt'])

    def test_second_editorially_invalid_draft_stops_before_summary_or_publication(self):
        outputs=story_outputs();bad=outputs[2]+'\n\nGrounding accompagne la scène jusqu’à la porte.';outputs=outputs[:2]+[bad,bad];calls=[]
        with self.assertRaises(personal.GenerationFailure):personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','drafting'])

    def test_analytical_review_cannot_reinject_verbatim_private_source_into_body(self):
        original='PRIVATE_SENTINEL_1987 dans mon atelier. Je ne veux pas de téléphone.'
        j=job('story',source(text=original));outputs=story_outputs();g=guidance();g['unrepresented_constraints']=True;outputs[0]=json.dumps(g)
        ending='Alix retrouve finalement la lettre et choisit de la conserver.'
        outputs[3]=json.dumps(review([{'quote':ending,'reason':'Ajuster la préférence','replacement':ending+' PRIVATE_SENTINEL_1987 dans mon atelier.'}]))
        calls=[]
        with self.assertRaises(personal.GenerationFailure) as caught:personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(caught.exception.code,'coherence_error');self.assertEqual(caught.exception.stage,'correction')
        self.assertEqual(calls[-1]['stage'],'review');self.assertNotIn('summary',[c['stage'] for c in calls])

    def test_copy_guard_preserves_established_fiction_and_blocks_structured_private_identifiers(self):
        j=job('story',source(text='Dans mon atelier secret je range les aquarelles anciennes.'))
        with self.assertRaisesRegex(ValueError,'coherence_error'):personal.assert_no_private_reuse('Dans mon atelier secret je range les aquarelles anciennes.',j)
        j['history']=[{'summary':'Dans mon atelier secret je range les aquarelles anciennes.'}]
        personal.assert_no_private_reuse('Dans mon atelier secret je range les aquarelles anciennes.',j)
        j['people']=[{'display_name':'Zélie Dupont'}]
        with self.assertRaisesRegex(ValueError,'coherence_error'):personal.assert_no_private_reuse('Zélie Dupont arrive près de la porte.',j)
        personal.assert_no_private_reuse('Un atelier, une porte et une pensée.',j)

    def test_analytical_constraint_review_excludes_life_scenes_and_preserves_complete_wishes(self):
        original='Contexte. '*40+'Je ne veux pas de personnage médecin.'
        j=job('story',source(text=original));j['sources'].append(source('event:e','PRIVATE_EVENT_SCENE_ONLY'))
        context=personal.analytical_reading_constraints(j)
        self.assertIn(original,json.dumps(context,ensure_ascii=False));self.assertNotIn('PRIVATE_EVENT_SCENE_ONLY',json.dumps(context))

    def test_free_character_preference_is_checked_before_identity_becomes_stable(self):
        j=job('characters',source(text='Je ne veux pas de personnage travaillant en soins.'))
        g=guidance();g['unrepresented_constraints']=True
        first={'characters':[character_fixture(i,name,['memory:user']) for i,name in [(1,'Alix'),(2,'Sam'),(3,'Noa')]]}
        first['characters'][0]['identity']['description']='Alix travaille comme médecin dans un hôpital.'
        second=copy.deepcopy(first);second['characters'][0]['identity']['description']='Alix organise les départs des trains dans une petite gare.'
        responses=[g,first,{'checks':[{'slot':i,'allowed':i!=1} for i in (1,2,3)]},second,{'checks':[{'slot':i,'allowed':True} for i in (1,2,3)]}];calls=[]
        result=personal.generate(j,lambda _,p:(calls.append(p) or {'content':json.dumps(responses.pop(0))}))
        self.assertIn('gare',result['characters'][0]['identity']['description']);self.assertEqual(len(calls),5)
        for index in (1,3):self.assertNotIn('Je ne veux pas de personnage',calls[index]['user_prompt'])
        self.assertIn('Je ne veux pas de personnage',calls[2]['user_prompt'])
        repeated=json.loads(calls[3]['user_prompt'])['rejected_proposals']
        self.assertEqual([c['slot'] for c in repeated],[1])
        self.assertNotIn('reason',calls[3]['user_prompt'])

    def test_free_character_preference_rejection_is_bounded_and_returns_no_identities(self):
        j=job('characters');g=guidance();g['unrepresented_constraints']=True
        proposed={'characters':[character_fixture(i,name,['memory:user']) for i,name in [(1,'Alix'),(2,'Sam'),(3,'Noa')]]}
        checks={'checks':[{'slot':i,'allowed':False} for i in (1,2,3)]};responses=[g,proposed,checks,proposed,checks];calls=[]
        with self.assertRaises(personal.GenerationFailure) as caught:personal.generate(j,lambda _,p:(calls.append(p) or {'content':json.dumps(responses.pop(0))}))
        self.assertEqual(caught.exception.code,'coherence_error');self.assertEqual(len(calls),5)

    def test_invalid_summary_is_repaired_once_with_fixed_diagnostics_and_immutable_body(self):
        outputs=story_outputs();valid=json.loads(outputs[4]);invalid=copy.deepcopy(valid)
        invalid['summary']='X'*1801
        invalid['narrative_delta']['events'][0]['quote']='Citation inventée non présente.'
        invalid['narrative_delta']['resolution']['quote']='## La scène'
        valid['body']='NE DOIT JAMAIS REMPLACER LE RÉCIT'
        outputs[4]=json.dumps(valid);outputs.insert(4,json.dumps(invalid));calls=[]
        result=personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','review','summary','summary','reflection'])
        first,repair=[json.loads(c['user_prompt']) for c in calls if c['stage']=='summary']
        self.assertEqual(first['body'],repair['body']);self.assertEqual(result['body'],first['body'])
        self.assertEqual(first['resolution_excerpt'],first['body'][-max(1500,len(first['body'])//3):])
        self.assertEqual(repair['repair_summary']['invalid_response'],json.dumps(invalid))
        self.assertEqual(set(repair['repair_summary']['issues']),{'summary_length','quote_not_exact','resolution_quote_not_final'})
        self.assertTrue(all(c['max_tokens']==4000 for c in calls if c['stage']=='summary'))
        self.assertIn('800 à 1200',next(c for c in calls if c['stage']=='summary')['system_prompt'])
        self.assertNotIn('NE DOIT JAMAIS',result['body'])
        self.assertNotIn('Citation inventée',calls[-1]['user_prompt'])
        personal.validate_narrative_delta(result['narrative_delta'],result['body'])

    def test_malformed_summary_json_gets_only_one_repair_without_logging_raw_response(self):
        import contextlib,io
        outputs=story_outputs();malformed='{"summary":"PRIVATE_MALFORMED_SENTINEL",';outputs.insert(4,malformed);calls=[];out=io.StringIO()
        with contextlib.redirect_stdout(out):
            result=personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        repair=json.loads([c for c in calls if c['stage']=='summary'][1]['user_prompt'])
        self.assertEqual(repair['repair_summary']['issues'],['malformed_json'])
        self.assertEqual(repair['repair_summary']['invalid_response'],malformed)
        self.assertEqual(out.getvalue(),'');self.assertTrue(result['summary'].startswith('Fiction :'))
        self.assertNotIn('PRIVATE_MALFORMED_SENTINEL',result['body'])

    def test_two_invalid_summary_responses_stop_before_reflection_or_publication(self):
        invalid=json.dumps({'summary':'Trop court','narrative_delta':{'events':'invalid'}})
        outputs=story_outputs()[:4]+[invalid,invalid];calls=[]
        with self.assertRaises(personal.GenerationFailure) as caught:
            personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual((caught.exception.stage,caught.exception.code),('summary','validation_error'))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','review','summary','summary'])
        issues=json.loads(calls[-1]['user_prompt'])['repair_summary']['issues']
        self.assertEqual(set(issues),{'summary_length','delta_schema'})

    def test_summary_repair_respects_existing_cumulative_call_budget(self):
        j=job('story');j['request']={'provider_calls':19};calls=[]
        adapter=personal.Adapter(j,lambda _,p:(calls.append(p) or {'content':'{"summary":"short"}'}));adapter.stage='summary'
        with self.assertRaises(RuntimeError):personal.summarize_final_story(j,personal.story_context_v2(j),body(),adapter,4000)
        self.assertEqual(len(calls),1);self.assertEqual(adapter.error_code,'quota_exceeded')

    def test_post_edit_invalid_body_gets_one_repair_then_mandatory_coherence_check(self):
        outputs=story_outputs();edits=[{'quote':'## '+title,'reason':'PRIVATE_REASON_NOT_FOR_WRITER','replacement':title} for title in ('La scène','Une autre lecture','Une pratique','La suite')]
        outputs[3]=json.dumps(review(edits));outputs[4:4]=[body(),json.dumps(review())];calls=[]
        result=personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','review','correction','coherence','summary','reflection'])
        repair=json.loads(calls[4]['user_prompt'])
        self.assertEqual(repair['validation_codes'],['headings']);self.assertNotIn('## ',repair['body'])
        self.assertIn('## La scène',repair['before_edit']);self.assertNotIn('PRIVATE_REASON_NOT_FOR_WRITER',calls[4]['user_prompt'])
        self.assertNotIn('analytical_preferences',repair)
        self.assertEqual(calls[4]['purpose'],'writer');self.assertEqual(calls[4]['max_tokens'],7000)
        self.assertEqual(calls[5]['purpose'],'analyst');self.assertEqual(calls[5]['max_tokens'],4000)
        self.assertEqual(json.loads(calls[5]['user_prompt'])['body'],result['body'])
        self.assertEqual(json.loads(calls[6]['user_prompt'])['body'],result['body'])
        personal.validate_story_body({'title':result['title']},result['body'])

    def test_new_second_review_issue_gets_second_targeted_round_and_final_check(self):
        outputs=story_outputs();old='Alix retrouve finalement la lettre et choisit de la conserver.'
        middle='Alix retrouve finalement la lettre et choisit de la relire.'
        final='Alix retrouve finalement la lettre et choisit de la ranger.'
        first={'quote':old,'reason':'Nuance de continuité','replacement':middle}
        second={'quote':middle,'reason':'Dernière nuance','replacement':final}
        outputs[3]=json.dumps(review([first]));outputs[4]=outputs[4].replace('conserver','ranger')
        outputs[4:4]=[json.dumps(review([second])),json.dumps(review())];calls=[]
        result=personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','review','coherence','coherence','summary','reflection'])
        self.assertIn(middle,json.loads(calls[4]['user_prompt'])['body']);self.assertIn(final,json.loads(calls[5]['user_prompt'])['body'])
        self.assertIn(final,result['body']);self.assertIn('ranger',result['summary'])

    def test_unresolved_issues_after_two_rounds_stop_without_third_round_or_summary(self):
        outputs=story_outputs();old='Alix retrouve finalement la lettre et choisit de la conserver.'
        middle=old.replace('conserver','relire');final=old.replace('conserver','ranger')
        reports=[review([{'quote':old,'reason':'Première nuance','replacement':middle}]),
                 review([{'quote':middle,'reason':'Deuxième nuance','replacement':final}]),
                 review([{'quote':final,'reason':'PRIVATE_UNRESOLVED_REASON','replacement':old}])]
        outputs=outputs[:3]+[json.dumps(r) for r in reports];calls=[]
        with self.assertRaises(personal.GenerationFailure) as caught:personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','review','coherence','coherence'])
        self.assertEqual(caught.exception.code,'coherence_error');self.assertEqual(caught.exception.validation_codes,('review_unresolved',))
        self.assertNotIn('PRIVATE_UNRESOLVED_REASON',str(caught.exception))

    def test_post_edit_body_repair_is_global_once_not_once_per_correction_round(self):
        edits=[{'quote':'## '+title,'reason':'Retoucher','replacement':title} for title in ('La scène','Une autre lecture','Une pratique','La suite')]
        outputs=story_outputs()[:3]+[json.dumps(review(edits)),body(),json.dumps(review(edits))];calls=[]
        with self.assertRaises(personal.GenerationFailure) as caught:personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','review','correction','coherence'])
        self.assertEqual((caught.exception.stage,caught.exception.code),('correction','validation_error'))
        self.assertEqual(caught.exception.validation_codes,('headings',))

    def test_invalid_post_edit_repair_stops_with_safe_body_codes_and_no_private_logs(self):
        import contextlib,io
        edits=[{'quote':'## '+title,'reason':'Retoucher','replacement':title} for title in ('La scène','Une autre lecture','Une pratique','La suite')]
        outputs=story_outputs()[:3]+[json.dumps(review(edits)),'PRIVATE_REPAIR_BODY_TOO_SHORT'];calls=[];log=io.StringIO()
        with contextlib.redirect_stdout(log),self.assertRaises(personal.GenerationFailure) as caught:
            personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(caught.exception.validation_codes,('headings','word_count'))
        self.assertEqual(str(caught.exception),'generation_failed');self.assertEqual(log.getvalue(),'')
        self.assertEqual(calls[-1]['stage'],'correction')
        filtered=personal.GenerationFailure('correction','validation_error',['headings','PRIVATE_BAD_CODE','headings',{'secret':'bad'}])
        self.assertEqual(filtered.validation_codes,('headings',));self.assertIsInstance(filtered.validation_codes,tuple)

    def test_post_edit_repair_keeps_private_copy_guard(self):
        original='PRIVATE_SENTINEL_1987 dans mon atelier.';j=job('story',source(text=original))
        edits=[{'quote':'## '+title,'reason':'Retoucher','replacement':title} for title in ('La scène','Une autre lecture','Une pratique','La suite')]
        outputs=story_outputs()[:3]+[json.dumps(review(edits)),body()+'\n\n'+original];calls=[]
        with self.assertRaises(personal.GenerationFailure) as caught:personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(caught.exception.code,'coherence_error');self.assertEqual(calls[-1]['stage'],'correction')
        self.assertNotIn('summary',[c['stage'] for c in calls])

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

    def test_an_old_resolution_is_retrieved_alongside_its_older_open_thread(self):
        j=job('story');j['history']=[{'id':str(i),'character_id':'char-a','summary':'Atelier ordinaire indépendant','narrative_delta':{},'episode_number':i} for i in range(40)]
        j['history'][0].update(summary='Une lettre disparue reste mystérieuse.',narrative_delta={'open_questions':[{'text':'Où est la lettre ?','quote':'lettre'}]})
        j['history'][8].update(summary='La lettre est retrouvée et la question est résolue.',narrative_delta={'resolution':{'text':'La lettre est conservée','quote':'lettre'}})
        selected=personal.story_context_v2(j)['history_retrieved']
        self.assertTrue(any(h['id']=='0' for h in selected));self.assertTrue(any(h['id']=='8' for h in selected))

    def test_only_restricted_event_can_prepare_honest_exploratory_characters(self):
        s=source('event:protected','PRIVATE SCENE');s['fiction_use']='understanding_only';j=job('characters',s)
        output={'characters':[character_fixture(slot,name,[]) for slot,name in [(1,'Alix'),(2,'Sam'),(3,'Noa')]]}
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
        character={'slot':1,'identity':{'name':'Camille','description':'Identité fictive conservée.','archive_evidence':[{'story_id':'old-story','quote':'Camille retrouve sa lettre.'}]},'facet':{'label':'Retrouver la lettre','description':'Exploration','evidence':['memory:user']},'narrative_state':{}}
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
        result={'characters':[character_fixture(slot,name,['memory:user']) for slot,name in [(2,'Sam'),(3,'Noa')]]}
        calls=[];out=personal.generate(j,lambda _,p:(calls.append(p) or {'content':json.dumps(guidance() if p['stage']=='orientation' else result)}))
        self.assertEqual([c['slot'] for c in out['characters']],[2,3]);self.assertIn('Identité ancienne conservée',calls[-1]['user_prompt'])
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
        self.assertIn(original,json.dumps(personal.fiction_analysis_context(j),ensure_ascii=False))
        self.assertNotIn(original,json.dumps(personal.story_context_v2(j),ensure_ascii=False))

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
        self.assertEqual(len(calls),6)
        summary_call=next(c for c in calls if c['stage']=='summary')
        self.assertIn('choisit de la conserver',summary_call['user_prompt'])
        self.assertIn('choisit de la conserver',result['summary'])
        self.assertEqual(calls[2]['temperature'],.7)
        self.assertEqual(calls[2]['max_tokens'],7000)
        personal.validate_narrative_delta(result['narrative_delta'],result['body'])

    def test_short_draft_gets_one_structural_repair_before_review_and_summary(self):
        outputs=story_outputs()
        short='TITRE_SEO: La lettre retrouvée\n---\n## La scène\nAlix retrouve une lettre et la conserve.'
        outputs.insert(2,short);calls=[]
        result=personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','drafting','review','summary','reflection'])
        repair=json.loads(calls[3]['user_prompt'])['correction_structure']
        self.assertEqual(set(repair['issues']),{'word_count','headings'})
        self.assertIn('Alix retrouve une lettre',repair['draft'])
        self.assertNotIn('correction_structure',calls[4]['user_prompt'])
        personal.validate_story_body({'title':result['title']},result['body'])

    def test_two_invalid_drafts_stop_without_review_or_publication(self):
        outputs=story_outputs()[:2]+['TITRE_SEO: Court\n---\nTrop court.','TITRE_SEO: Court\n---\nToujours trop court.'];calls=[]
        with self.assertRaises(personal.GenerationFailure) as caught:
            personal.generate(job('story'),lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(caught.exception.stage,'drafting')
        self.assertEqual([c['stage'] for c in calls],['orientation','planning','drafting','drafting'])

    def test_targeted_correction_is_rechecked_and_resummary_uses_corrected_final_body(self):
        j=job('story');outputs=story_outputs();outputs[3]=json.dumps(review([{'quote':'Alix retrouve finalement la lettre et choisit de la conserver.','reason':'Choix contraire au plan','replacement':'Alix retrouve finalement la lettre et choisit de la relire.'}]))
        outputs.insert(4,json.dumps(review()))
        outputs[5]=outputs[5].replace('conserver','relire')
        calls=[];result=personal.generate(j,lambda _,p:(calls.append(p) or {'content':outputs.pop(0)}))
        self.assertEqual(len(calls),7);self.assertIn('relire',result['body']);self.assertIn('relire',result['summary'])

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
                self.assertEqual(personal.fiction_analysis_context(j)['sources'][0]['text'],j['sources'][0]['text'])
                self.assertNotIn('sources',ctx)
                self.assertEqual(ctx['character']['id'],'char-a')
                self.assertNotIn('fictional_character',ctx['portrait'])

if __name__=='__main__':unittest.main()
