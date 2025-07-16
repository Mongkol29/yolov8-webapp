
self.addEventListener('install', function(event) {
  console.log("✅ Service Worker ติดตั้งแล้ว");
});

self.addEventListener('fetch', function(event) {
  event.respondWith(fetch(event.request));
});
