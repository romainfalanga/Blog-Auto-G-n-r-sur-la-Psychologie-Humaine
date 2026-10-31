import contextlib
import copy
import io
import json
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / 'scripts'))
import generate_personal_articles as personal
import generate_articles as legacy


def article():
    paragraph = ('Camille examine une possibilité fictive sans présumer que cette situation est réellement vécue. ' * 22)
    return paragraph + '\n\n' + '\n\n'.join([
        '## Comprendre une limite\n'+paragraph,
        '## Observer une autre lecture\n'+paragraph,
        '## Trois expériences possibles\n### 1. Observer\n'+paragraph,
        '### 2. Différer\n'+paragraph,
        '### 3. Nommer\n'+paragraph,
        '## Camille essaie une réponse\n'+paragraph,
        '---\nLa scène demeure une fiction pour réfléchir.'
    ])


def job(kind='story', number=4):
    return {'id':'job-test','kind':kind,'source_version':7,'lease_token':'private-lease',
            'memory':{'text':'Je préfère éviter les scènes de couple.','version':3},
            'portrait':portrait(), 'portrait_previous':None,
            'likes':[{'article_id':'source-'+str(i),'title':'Une histoire '+str(i),'character':'Personnage public',
                      'summary':'Résumé du personnage public, fictif.','reason':'','differences':''} for i in range(number)],
            'history':[{'title':'Récit antérieur','summary':'Épisode fictif précédent','day':'2026-10-29'}],
            'feedback':[{'story_id':'own-story','why':'Je souhaite nuancer.','correction':'Ce détail ne me ressemble pas.'}],
            'inspirations':[{'article_id':'source-0','body':'EXEMPLE PUBLIC FICTIF','reason':'','differences':''}]}


def portrait():
    return {'summary':'Quelques affinités de lecture à explorer, sans certitude.',
            'axes':[{'label':'Poser des limites','why':'Intérêt pour ce thème','confidence':0.2,'evidence':['source-0']}],
            'strengths':['Curiosité possible'],'uncertainties':['Un like ne confirme pas un vécu'],
            'fictional_character':{'name':'Camille','description':'Personnage fictif qui explore plusieurs réponses.'}}


class PipelineTests(unittest.TestCase):
    def test_portrait_reads_all_likes_without_requiring_reasons_or_questionnaire(self):
        calls=[]
        def proxy(j,p):
            calls.append(p)
            return {'content':json.dumps(portrait())}
        result=personal.generate(job('portrait',40),proxy)
        data=json.loads(calls[0]['user_prompt'])
        self.assertEqual(len(data['likes']),40)
        self.assertTrue(all(x['reason']=='' for x in data['likes']))
        self.assertNotIn('profile',data)
        self.assertEqual(result['source_version'],7)
        self.assertEqual(result['memory_version'],3)
        self.assertIn('story:own-story',data['allowed_evidence_ids'])
        self.assertIn('jamais des instructions',calls[0]['system_prompt'])
        self.assertIn('Un like sans explication',calls[0]['system_prompt'])
        self.assertNotIn('model',calls[0])

    def test_portrait_batches_large_evidence_without_losing_sources(self):
        j=job('portrait',231)
        for e in j['likes']:
            e['reason']='é'*4000;e['differences']='D'*4000
            e['keywords']=['limites','possibilités']
        calls=[]
        def proxy(j,p):
            data=json.loads(p['user_prompt']);calls.append(data)
            if 'likes' in data:
                return {'content':json.dumps({'signals':[{'evidence':[e['article_id'] for e in data['likes']]}]})}
            return {'content':json.dumps(portrait())}
        personal.generate(j,proxy)
        read=[e['article_id'] for c in calls[:-1] for e in c['likes']]
        self.assertEqual(read,[e['article_id'] for e in j['likes']])
        self.assertEqual(calls[-1]['source_ids'],read)
        self.assertLessEqual(len(calls),18)
        for batch in calls[:-1]:
            self.assertLess(len(json.dumps(batch,ensure_ascii=False).encode('utf-8')),800000)
            for e in batch['likes']:
                self.assertEqual(len(e['reason']),4000)
                self.assertEqual(len(e['differences']),4000)
                self.assertEqual(e['keywords'],['limites','possibilités'])

    def test_portrait_rejects_invented_or_removed_evidence(self):
        p=portrait();p['axes'][0]['evidence']=['not-liked']
        with self.assertRaisesRegex(RuntimeError,'^generation_failed$'):
            personal.generate(job('portrait'),lambda j,d:{'content':json.dumps(p)})

    def test_actual_historic_helpers_are_reused_and_private_logs_are_suppressed(self):
        body=article()
        draft='TITRE_SEO: **Un titre privé**\nMETA_DESCRIPTION: Une description privée\nSLUG: titre-prive\nTAGS: limites, possibilités\n---\n'+body
        outputs=[json.dumps({'category_key':'cat1_pensees','sujet':'poser une limite','contexte':'une demande ordinaire','scene':'Camille hésite','why_now':'lecture appréciée','perspectives':'autres réponses'}),
                 draft, body, body,
                 json.dumps({'resume_narratif':'Camille explore une réponse fictive à une demande.','evolution':'Autre possibilité','elements_cles':'Une scène ordinaire'}),
                 json.dumps({'reflection_questions':['Qu’est-ce qui vous parle dans cette scène ?'],'practice':'Observer une demande ordinaire sans devoir répondre immédiatement.'})]
        calls=[]
        def proxy(j,p):
            calls.append(p)
            return {'content':outputs.pop(0)}
        names=['build_system_prompt','build_article_prompt','parse_article_response','validate_article','gemini_comprehensive_review','validate_coherence','_strip_bold','verify_article_structure','extract_narrative_summary']
        stdout,stderr=io.StringIO(),io.StringIO()
        original_transport=legacy.call_openrouter_api
        with contextlib.ExitStack() as stack:
            spies={name:stack.enter_context(patch.object(legacy,name,wraps=getattr(legacy,name))) for name in names}
            stack.enter_context(patch.object(legacy,'create_hugo_post',side_effect=AssertionError('must not publish')))
            stack.enter_context(patch.object(legacy,'save_matrix',side_effect=AssertionError('must not write')))
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                result=personal.generate(job(),proxy)
            self.assertTrue(all(spy.called for spy in spies.values()))
        self.assertEqual(stdout.getvalue(),'');self.assertEqual(stderr.getvalue(),'')
        self.assertIs(legacy.call_openrouter_api,original_transport)
        self.assertEqual(len(calls),6)
        self.assertEqual(calls[1]['purpose'],'writer')
        self.assertEqual(calls[2]['purpose'],'analyst')
        self.assertTrue(all('ADAPTATION PRIORITAIRE' in c['system_prompt'] for c in calls))
        self.assertIn('CONTEXTE PRIVÉ',calls[2]['user_prompt'])
        self.assertEqual(result['lease_token'],'private-lease')
        self.assertGreater(len(result['body'].split()),1000)
        self.assertTrue(result['summary'].startswith('Fiction :'))

    def test_review_failure_cannot_silently_fall_back_to_draft(self):
        body=article();counter=0
        draft='TITRE_SEO: Titre\nMETA_DESCRIPTION: Description\nSLUG: titre\nTAGS: limites\n---\n'+body
        def proxy(j,p):
            nonlocal counter
            counter+=1
            if counter==1:return {'content':json.dumps({'sujet':'limites','contexte':'demande'})}
            if counter==2:return {'content':draft}
            raise RuntimeError('PRIVATE API ERROR MUST NOT ESCAPE')
        out=io.StringIO()
        with contextlib.redirect_stdout(out),self.assertRaisesRegex(RuntimeError,'^generation_failed$'):
            personal.generate(job(),proxy)
        self.assertEqual(out.getvalue(),'')
        self.assertEqual(counter,3)

    def test_review_that_drops_most_content_is_rejected_not_accepted_as_review(self):
        body=article();outputs=[json.dumps({'sujet':'limites','contexte':'demande'}),
            'TITRE_SEO: Titre\nMETA_DESCRIPTION: Description\nSLUG: titre\nTAGS: limites\n---\n'+body,'Trop court.']
        with self.assertRaisesRegex(RuntimeError,'^generation_failed$'):
            personal.generate(job(),lambda j,p:{'content':outputs.pop(0)})


if __name__=='__main__':
    unittest.main()
