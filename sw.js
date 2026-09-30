// Minimal service worker: cache the app shell so Aria opens instantly / offline.
// API calls (/chat, /chat/stream, /health) are always network — never cached.
const CACHE = 'aria-v1';
const SHELL = ['/', '/index.html', '/icon.svg', '/manifest.webmanifest'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // never cache API / dynamic endpoints
  if (e.request.method !== 'GET' || /^\/(chat|health)/.test(url.pathname)) return;
  // cache-first for the app shell, falling back to network
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request)));
});
