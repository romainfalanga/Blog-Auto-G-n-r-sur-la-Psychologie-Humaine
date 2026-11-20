export default function legacyPrivatePages(request) {
  if (request.method !== 'GET' && request.method !== 'HEAD') return;

  const url = new URL(request.url);
  // Also guard the handler itself: an Edge route must never redirect the home page.
  if (!/^\/(?:mon-personnage|mes-personnages|mes-histoires|journal)(?:\/|$)/.test(url.pathname)) return;
  let location = '/ambre/';
  if (/^\/journal(?:\/|$)/.test(url.pathname)) {
    const query = url.searchParams;
    const tab = query.get('tab');
    if (query.has('payment')) {
      location = '/mon-compte/?' + new URLSearchParams({ payment: query.get('payment') });
    } else if (query.has('article') || tab === 'likes' || tab === 'portrait' || tab === 'account') {
      // Liked articles are retired: their old links open the account page, without parameters.
      location = '/mon-compte/';
    }
  }

  return new Response(null, {
    status: 302,
    headers: { Location: location, 'Cache-Control': 'private, no-store', 'Referrer-Policy': 'no-referrer' },
  });
}
