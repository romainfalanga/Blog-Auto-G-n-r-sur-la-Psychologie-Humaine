export default function legacyPrivatePages(request) {
  if (request.method !== 'GET' && request.method !== 'HEAD') return;

  const url = new URL(request.url);
  let location = '/mon-accompagnement/';
  if (/^\/journal(?:\/|$)/.test(url.pathname)) {
    const query = url.searchParams;
    const tab = query.get('tab');
    if (query.has('payment')) {
      location = '/mon-compte/?' + new URLSearchParams({ payment: query.get('payment') });
    } else if (query.has('article') || tab === 'likes') {
      const reading = new URLSearchParams();
      for (const key of ['article', 'like']) if (query.has(key)) reading.set(key, query.get(key));
      location = '/mon-portrait/' + (reading.size ? '?' + reading : '') + '#mes-articles';
    } else if (tab === 'portrait' || tab === 'account') {
      location = tab === 'portrait' ? '/mon-portrait/' : '/mon-compte/';
    }
  }

  return new Response(null, {
    status: 302,
    headers: { Location: location, 'Cache-Control': 'private, no-store', 'Referrer-Policy': 'no-referrer' },
  });
}
