// Small shared behaviours for the portfolio: mobile nav toggle + footer year.
(function () {
  'use strict';

  var toggle = document.querySelector('.nav__toggle');
  var links = document.querySelector('.nav__links');
  if (toggle && links) {
    toggle.addEventListener('click', function () {
      var open = links.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    links.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') links.classList.remove('is-open');
    });
  }

  var yearEl = document.querySelector('[data-year]');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  // Headshot slideshow: read a manifest of filenames and crossfade through them.
  var slideshow = document.querySelector('[data-slideshow]');
  if (slideshow) initSlideshow(slideshow);

  function initSlideshow(el) {
    var manifestUrl = el.getAttribute('data-manifest');
    var base = el.getAttribute('data-base') || '';
    var alt = el.getAttribute('data-alt') || '';
    var interval = parseInt(el.getAttribute('data-interval'), 10) || 4000;

    fetch(manifestUrl)
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (names) {
        if (!Array.isArray(names) || names.length === 0) return; // keep placeholder
        var placeholder = el.querySelector('.placeholder');
        if (placeholder) placeholder.remove();

        var imgs = names.map(function (name, i) {
          var img = document.createElement('img');
          img.src = base + name;
          img.alt = alt;
          img.loading = i === 0 ? 'eager' : 'lazy';
          if (i === 0) img.className = 'is-active';
          el.appendChild(img);
          return img;
        });

        if (imgs.length < 2) return; // single image: nothing to cycle
        var idx = 0;
        setInterval(function () {
          imgs[idx].classList.remove('is-active');
          idx = (idx + 1) % imgs.length;
          imgs[idx].classList.add('is-active');
        }, interval);
      })
      .catch(function () { /* leave the placeholder in place */ });
  }
})();
