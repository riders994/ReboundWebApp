'use strict';

// Default card covers.
//
// Cards name their image by convention (`<slug>-thumb.png`); most projects don't have one
// yet, and an empty 16:10 box reads as a broken card. This hands those cards an image from
// the `fallbacks` feed (site/assets/img/fallbacks/, listed in its manifest.json) instead.
//
// Two properties matter more than the images themselves:
//
//   Spread - a page deals from a shuffled bag rather than picking per card, so a cover is
//   only reused once every other cover has been used. With more covers than cards (the
//   case today: 12 vs 11 projects) nothing repeats at all. Each refill reshuffles, and
//   never lands the same cover twice in a row across the seam.
//
//   Stability - the shuffle is seeded from the manifest, not from Math.random(), so a card
//   keeps its cover across reloads and between the projects grid and the resume carousel.
//   A card whose picture changed on every visit would read as broken in a different way.
//   Adding or removing a cover reshuffles the deck; that is a deliberate content change.
//
// Cards are dealt a slot up front, in render order, including cards that turn out to have
// a real thumb of their own - the alternative is waiting on every 404 to settle, which
// would make the assignment depend on network timing and undo the stability above.
window.CardThumbs = (function () {
  var manifests = {};   // directory URL -> Promise<string[]>

  function load(dir) {
    if (!manifests[dir]) {
      manifests[dir] = fetch(dir + 'manifest.json')
        .then(function (r) { return r.ok ? r.json() : []; })
        .then(function (names) { return Array.isArray(names) ? names : []; })
        .catch(function () { return []; });
    }
    return manifests[dir];
  }

  // FNV-1a, so the deck order is a pure function of the manifest.
  function hash(str) {
    var h = 2166136261;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function mulberry32(seed) {
    return function () {
      seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function shuffled(list, rand) {
    var out = list.slice();
    for (var i = out.length - 1; i > 0; i--) {
      var j = Math.floor(rand() * (i + 1));
      var t = out[i]; out[i] = out[j]; out[j] = t;
    }
    return out;
  }

  // Deal `count` covers from a bag that refills (reshuffled) once emptied.
  function deal(names, count, rand) {
    var out = [], bag = [];
    for (var i = 0; i < count; i++) {
      if (bag.length === 0) {
        bag = shuffled(names, rand);
        // Don't repeat across the seam: the next pop is bag's last element.
        if (bag.length > 1 && bag[bag.length - 1] === out[out.length - 1]) {
          var t = bag[0]; bag[0] = bag[bag.length - 1]; bag[bag.length - 1] = t;
        }
      }
      out.push(bag.pop());
    }
    return out;
  }

  function blank(img) {
    // Last resort - the pre-fallback behaviour: an empty tinted box.
    img.style.background = 'var(--bg-soft)';
    img.removeAttribute('src');
  }

  // keys: card identifiers in render order. base: the image directory the cards' own
  // thumbs come from (the feed lives in `fallbacks/` beneath it).
  function prepare(keys, base) {
    var dir = (base || '') + 'fallbacks/';
    var assigned = load(dir).then(function (names) {
      var map = {};
      if (names.length === 0) return map;
      var picks = deal(names, keys.length, mulberry32(hash(names.join(' '))));
      keys.forEach(function (key, i) {
        // Local feeds store filenames; S3 feeds store full URLs (see gen-manifests.py).
        map[key] = /^(https?:)?\/\//.test(picks[i]) ? picks[i] : dir + picks[i];
      });
      return map;
    });

    return {
      apply: function (img, key) {
        assigned.then(function (map) {
          if (!map[key]) { blank(img); return; }
          img.onerror = function () { blank(img); };   // the cover itself is missing
          img.src = map[key];
        });
      }
    };
  }

  return { prepare: prepare };
})();
