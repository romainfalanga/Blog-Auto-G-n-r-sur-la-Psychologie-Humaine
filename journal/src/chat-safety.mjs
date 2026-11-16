// A conservative lexical backstop, not a clinical assessment or a risk score.
// The model must still consider context and signals that this list cannot cover.
export function crisisSignal(value){
 const text=String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[’']/g,' ').replace(/\s+/g,' ');
 const self=/\b(?:suicid\w*|automutil\w*|me (?:tuer|suicider|faire du mal|blesser|scarifier)|(?:je (?:veux|voudrais|vais|compte|prefere)|j (?:ai envie|aimerais|ai decide)) (?:de )?(?:mourir|disparaitre)|(?:j ai envie|je veux|je vais|je compte|j ai decide|je voudrais) (?:d |de )?en finir|je (?:n en |ne |en )?peux plus (?:de )?vivre|je ne veux plus (?:vivre|etre la)|ne (?:plus|pas) me reveiller|(?:tout le monde|les autres|mes proches|ma famille) (?:serai\w*|ira\w*) mieux sans moi)\b/;
 const means=/\b(?:dose|quantite|combien|hauteur)\b[^.!?]{0,100}\b(?:mortel\w*|mourir|tuer|letal\w*|medicaments?|comprimes?)\b|\b(?:j ai (?:avale|pris)|je viens d (?:avaler|en prendre))\b[^.!?]{0,65}\b(?:tous|toute?s?|trop|boite|flacon|surdose)\b|\b(?:je (?:vais|veux|compte)|j ai envie de) (?:me jeter|sauter)\b[^.!?]{0,50}\b(?:pont|fenetre|vide|train|toit)\b/;
 const other=/\b(?:je (?:vais|veux|compte)|j ai (?:envie|decide) de) (?:le |la |les |vous |te |t )?(?:tuer|poignarder|etrangler)\b|\b(?:il|elle|mon [a-z]+|ma [a-z]+) (?:va|veut|menace de) (?:me |nous |le |la |les )?(?:tuer|poignarder|etrangler)\b/;
 if(other.test(text))return 'violence';
 return self.test(text)||means.test(text)?'self_harm':null;
}
export const CRISIS_RESOURCES='En France, vous pouvez appeler le 3114, 24 h/24 et 7 j/7, pour vous ou pour un proche. En cas de danger immédiat, de blessure ou de prise de médicaments en excès, appelez tout de suite le 15 ou le 112. Rapprochez-vous d’une personne de confiance et restez avec elle. Hors de France, contactez les secours de votre pays.';
export function safetyReply(signal){
 if(signal==='violence')return 'Ce que vous évoquez demande de prendre la sécurité au sérieux. En cas de menace ou de violence immédiate, appelez le 17 ou le 112 ; pour une urgence médicale, le 15. Si vous le pouvez sans vous exposer, mettez de la distance avec le danger et rejoignez une personne de confiance. Hors de France, contactez les secours de votre pays.\n\nY a-t-il un danger immédiat pour vous ou pour quelqu’un d’autre ?';
 return 'Ce que vous évoquez mérite une attention immédiate à la sécurité, sans jugement.\n\n'+CRISIS_RESOURCES+'\n\nEst-ce que vous ou la personne dont vous parlez pensez au suicide ou risquez de vous faire du mal maintenant ?';
}
export function ensureSafetyResources(response,signal){
 // Append the trusted block after prose filtering so it can never be cut away.
 return response+'\n\n'+(signal==='violence'?'En cas de violence ou de menace immédiate en France, appelez le 17 ou le 112 ; pour une urgence médicale, le 15. Rejoignez une personne de confiance si vous pouvez le faire sans vous exposer. Hors de France, contactez les secours de votre pays.':CRISIS_RESOURCES);
}
