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

  // Comedy clips: read a JSON list of YouTube videos and render responsive embeds.
  var clips = document.querySelector('[data-clips]');
  if (clips) initClips(clips);

  function initClips(el) {
    var url = el.getAttribute('data-clips');
    fetch(url)
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (items) {
        if (!Array.isArray(items) || items.length === 0) {
          el.innerHTML = '<p class="todo">No clips yet — add YouTube video IDs to ' + url + '.</p>';
          return;
        }
        var grid = document.createElement('div');
        grid.className = 'grid grid--2';
        items.forEach(function (item) {
          var id = typeof item === 'string' ? item : item.id;
          if (!id) return;
          var title = (typeof item === 'object' && item.title) ? item.title : 'Comedy clip';

          var figure = document.createElement('figure');
          figure.style.margin = '0';
          var frame = document.createElement('div');
          frame.className = 'embed';

          // Facade: show YouTube's own thumbnail still + a play button; load the
          // player iframe only on click (faster than N live embeds, no local image).
          var facade = document.createElement('button');
          facade.type = 'button';
          facade.className = 'embed__facade';
          facade.setAttribute('aria-label', 'Play: ' + title);
          facade.style.backgroundImage =
            "url('https://i.ytimg.com/vi/" + encodeURIComponent(id) + "/hqdefault.jpg')";
          facade.innerHTML = '<span class="embed__play" aria-hidden="true"></span>';
          facade.addEventListener('click', function () {
            var iframe = document.createElement('iframe');
            iframe.src = 'https://www.youtube.com/embed/' + encodeURIComponent(id) + '?autoplay=1';
            iframe.title = title;
            iframe.allow = 'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture; web-share';
            iframe.setAttribute('allowfullscreen', '');
            frame.innerHTML = '';
            frame.appendChild(iframe);
          });
          frame.appendChild(facade);

          figure.appendChild(frame);
          if (typeof item === 'object' && item.title) {
            var cap = document.createElement('figcaption');
            cap.textContent = item.title;
            cap.style.marginTop = '.5rem';
            figure.appendChild(cap);
          }
          grid.appendChild(figure);
        });
        el.innerHTML = '';
        el.appendChild(grid);
      })
      .catch(function () { el.innerHTML = '<p class="todo">Couldn’t load clips.</p>'; });
  }

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
          // Local feeds store filenames (prepend base); S3 feeds store full URLs.
          img.src = /^(https?:)?\/\//.test(name) ? name : base + name;
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
