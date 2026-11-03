// Synthetic French annotations written before comparing engines. Grades: 3 direct,
// 2 useful neighboring reading. Unlisted articles are unjudged and scored 0.
// This is an engineering fixture set, not independent editorial/clinical validation.
const cases=[
 ['demande-colere','Je veux explorer la colère en situation de conflit.', ['colere-en-situation-de-conflit-comprendre-s-affirmer','rage-en-situation-de-conflit']],
 ['rumination','Je rumine seul après le travail et je veux comprendre cette rumination.', ['rumination-mentale-seul-face-a-soi-meme-adrien','overthinking-surpensee-travail']],
 ['travail-anxiete','Je ressens de l’anxiété au travail et beaucoup de suranalyse.', ['anxiete-au-travail-signes-discrets','overthinking-surpensee-travail']],
 ['culpabilite-enfants','Je ressens de la culpabilité avec mes enfants dans mon rôle de parent.', ['culpabilite-avec-ses-enfants-comprendre-agir','regret-avec-ses-enfants']],
 ['deuil','Je souhaite lire sur la tristesse lors d’un deuil.', ['comprendre-tristesse-lors-d-un-deuil','resilience-emotionnelle-lors-un-deuil-comprendre']],
 ['amitie-trahison','Une trahison en amitié a fragilisé ma confiance.', ['trahison-en-amitie-impacts-quotidiens','deception-en-amitie-comprendre-attentes-relations']],
 ['rupture','Après ma rupture amoureuse, je ressens un vide intérieur.', ['vide-interieur-apres-rupture-amoureuse','orientation-regret-rupture']],
 ['enthousiasme','Je suis enthousiaste pour ma reconversion professionnelle.', ['enthousiasme-reconversion-professionnelle-burn-out','espoir-après-un-succès-reconversion-hpi']],
 ['gratitude','Je souhaite explorer la gratitude face à l’argent.', ['gratitude-face-a-l-argent-et-aux-finances','culpabilite-face-a-l-argent-et-finances']],
 ['fierte','Je ressens de la fierté après un succès au travail.', ['fierte-au-travail-comprendre-manifestations-hpi','soulagement-apres-un-succes']],
 ['charge-mentale','Je vis une charge mentale émotionnelle au travail.', ['charge-mentale-emotionnelle-au-travail-solutions','fatigue-compassion-travail-consequences']],
 ['perfectionnisme','Je veux comprendre mon perfectionnisme et mes exigences élevées au travail.', ['exigences-elevees-critique-excessive-au-travail','effet-zeigarnik-manager-perfectionniste']],
 ['approbation','Je recherche l’approbation de mes parents.', ['recherche-approbation-parents','recherche-approbation-reconnaissance-amitie-enfance']],
 ['parentification','Je portais les responsabilités de mes parents enfant, je veux explorer la parentification.', ['parentification-avec-ses-parents-causes-solutions','adultification-avec-ses-parents-sortir-schema']],
 ['solitude','Je me sens seul et isolé, la solitude prend beaucoup de place.', ['solitude-face-a-la-solitude-comprendre-difference','isolement-social-alienation-solitude']],
 ['imposture','Je souhaite lire sur le syndrome de l’imposteur dans mes études.', ['syndrome-imposteur-ecole-etudes-lucas','sentiment-inadequation-ecole-etudes-hpi']],
 ['comparaison','Je me compare aux autres dans la compétition.', ['comparaison-injuste-competition','comprendre-envie-ecole-etudes-psychologie']],
 ['colocation','Je ressens de l’agacement en colocation.', ['agacement-en-colocation-comprendre-emotions','honte-en-colocation-causes-solutions']],
 ['autorite','J’ai peur face à l’autorité, je voudrais comprendre cette peur.', ['peur-face-a-lautorite','honte-face-a-l-autorite-causes-profondes']],
 ['tendresse','Je veux lire sur la tendresse avec ses enfants.', ['tendresse-avec-ses-enfants-bernard','attachement-secure-enfants-caracteristiques-developper']],
 ['couple-emotions','En couple, il m’est difficile d’identifier et nommer mes émotions.', ['alexithymie-en-couple','granularite-emotionnelle-en-couple-comprendre-ressentis']],
 ['demander-aide','Je souhaite explorer la compassion et la fatigue à aider face à la maladie.', ['fatigue-compassion-face-maladie','compassion-face-a-la-maladie-aider-proche']],
 ['emerveillement','Je ressens de l’émerveillement pendant mon expatriation.', ['emerveillement-en-voyage-ou-en-expatriation-guide','apathie-en-voyage-ou-expatriation-comprendre']],
 ['decision','Je veux comprendre le biais de confirmation et les conflits.', ['biais-confirmation-conflit','biais-attribution-fondamentale-situation-conflit']],
 ['negation-double','Je ne veux pas éviter la colère. Je veux comprendre mes émotions en conflit.', ['colere-en-situation-de-conflit-comprendre-s-affirmer','rage-en-situation-de-conflit']],
 ['correction-tardive','Je prends le temps de donner le contexte, les circonstances et les nuances, car une première interprétation ne suffit pas à résumer ce qui compte pour moi. Actuellement, je souhaite lire sur la gratitude et les finances.', ['gratitude-face-a-l-argent-et-aux-finances','culpabilite-face-a-l-argent-et-finances']],
 ['ancien-evenement','Mon deuil date de 2011 ; aujourd’hui je souhaite comprendre la nostalgie et la solitude.', ['nostalgie-face-a-la-solitude-comprendre-emotions','melancolie-face-a-la-solitude']],
 ['differences-nominales','Je souhaite lire sur le perfectionnisme au travail.', ['exigences-elevees-critique-excessive-au-travail','effet-zeigarnik-manager-perfectionniste']],
 ['souhait-style','Je souhaite une lecture sur l’écriture expressive en famille.', ['ecriture-expressive-pennebaker-en-famille-guide','journaling-pensees-intimite-sexualite-nadia']],
 ['negation-partielle','La famille ne correspond pas à mon vécu, mais je souhaite lire sur les conflits et la colère.', ['colere-en-situation-de-conflit-comprendre-s-affirmer','rage-en-situation-de-conflit']],
];
export function scenarios(examples){
 const id=slug=>{const e=examples.find(e=>e.id.endsWith('/'+slug));if(!e)throw Error('Missing annotation '+slug);return e.id;};
 const seed=id('overthinking-surpensee-travail');
 return cases.map(([name,text,slugs],i)=>({id:name,annotation:'3 = thème/situation directement demandés ; 2 = lecture voisine utile, sans présumer un vécu.',
  context:i<15?{likes:[{article_id:seed,created_at:'2026-11-01',reason:text,differences:i===4?'Le perfectionnisme au travail':''}]}:
   i%3===0?{likes:[],people:[{id:'p'+i,relationship_text:text}]}:i%3===1?{likes:[],events:[{id:'e'+i,description:text,kind:i===22?'positive':'other'}]}:{likes:[],memory:{text}},
  judgments:Object.fromEntries(slugs.filter(s=>!(i<15&&id(s)===seed)).map((s,j)=>[id(s),j===0?3:2]))}));
}
