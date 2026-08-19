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

  var CATEGORY_LABELS = { location: 'Location', date: 'Date', subject: 'Subject', joke: 'Jokes' };

  function initClips(el) {
    var clipsUrl = el.getAttribute('data-clips');
    var tagsUrl = el.getAttribute('data-tags');

    Promise.all([
      fetch(clipsUrl).then(function (r) { return r.ok ? r.json() : []; }).catch(function () { return []; }),
      tagsUrl
        ? fetch(tagsUrl).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; })
        : Promise.resolve(null)
    ]).then(function (res) {
      var clips = (Array.isArray(res[0]) ? res[0] : []).map(normClip).filter(function (c) { return c.id; });
      var registry = res[1] && res[1].tags ? res[1] : { categories: [], tags: {} };

      if (clips.length === 0) {
        el.innerHTML = '<p class="todo">No clips yet — add YouTube video IDs to ' + clipsUrl + '.</p>';
        return;
      }

      // Facets: registry tags that are actually used by at least one clip, grouped by category.
      var used = {};
      clips.forEach(function (c) { c.tags.forEach(function (t) { used[t] = true; }); });
      var groups = registry.categories.map(function (cat) {
        var codes = Object.keys(registry.tags).filter(function (code) {
          return registry.tags[code].category === cat && used[code];
        }).sort();
        return { category: cat, codes: codes };
      }).filter(function (g) { return g.codes.length > 0; });

      var selected = new Set(readSelectedTags());

      el.innerHTML = '';
      var filterBar = document.createElement('div');
      filterBar.className = 'filters';
      var grid = document.createElement('div');
      grid.className = 'grid grid--2 clips-grid';
      el.appendChild(filterBar);
      el.appendChild(grid);

      function passes(clip) {
        // Faceted: within a category any selected tag matches (OR); across categories all
        // active categories must match (AND).
        return groups.every(function (g) {
          var picked = g.codes.filter(function (code) { return selected.has(code); });
          if (picked.length === 0) return true;
          return picked.some(function (code) { return clip.tags.indexOf(code) !== -1; });
        });
      }

      function render() {
        // filter bar
        filterBar.innerHTML = '';
        groups.forEach(function (g) {
          var row = document.createElement('div');
          row.className = 'filter-group';
          var lab = document.createElement('span');
          lab.className = 'filter-group__label';
          lab.textContent = CATEGORY_LABELS[g.category] || g.category;
          row.appendChild(lab);
          g.codes.forEach(function (code) {
            var chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'chip' + (selected.has(code) ? ' is-active' : '');
            chip.setAttribute('aria-pressed', selected.has(code) ? 'true' : 'false');
            chip.textContent = registry.tags[code].label;
            chip.addEventListener('click', function () {
              if (selected.has(code)) selected.delete(code); else selected.add(code);
              writeSelectedTags(selected);
              render();
            });
            row.appendChild(chip);
          });
          filterBar.appendChild(row);
        });
        if (selected.size > 0) {
          var clear = document.createElement('button');
          clear.type = 'button';
          clear.className = 'filters__clear';
          clear.textContent = 'Clear filters';
          clear.addEventListener('click', function () { selected.clear(); writeSelectedTags(selected); render(); });
          filterBar.appendChild(clear);
        }

        // grid
        var visible = clips.filter(passes);
        grid.innerHTML = '';
        if (visible.length === 0) {
          grid.innerHTML = '<p class="muted">No clips match those tags.</p>';
          return;
        }
        visible.forEach(function (clip) { grid.appendChild(buildClipCard(clip)); });
      }

      render();
    });
  }

  function normClip(item) {
    if (typeof item === 'string') return { id: item, title: '', tags: [] };
    return { id: item.id, title: item.title || '', tags: Array.isArray(item.tags) ? item.tags : [] };
  }

  function buildClipCard(clip) {
    var title = clip.title || 'Comedy clip';
    var figure = document.createElement('figure');
    figure.style.margin = '0';
    var frame = document.createElement('div');
    frame.className = 'embed';

    // Facade: YouTube's own thumbnail still + play button; load the player only on click.
    var facade = document.createElement('button');
    facade.type = 'button';
    facade.className = 'embed__facade';
    facade.setAttribute('aria-label', 'Play: ' + title);
    facade.style.backgroundImage = "url('https://i.ytimg.com/vi/" + encodeURIComponent(clip.id) + "/hqdefault.jpg')";
    facade.innerHTML = '<span class="embed__play" aria-hidden="true"></span>';
    facade.addEventListener('click', function () {
      var iframe = document.createElement('iframe');
      iframe.src = 'https://www.youtube.com/embed/' + encodeURIComponent(clip.id) + '?autoplay=1';
      iframe.title = title;
      iframe.allow = 'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture; web-share';
      iframe.setAttribute('allowfullscreen', '');
      frame.innerHTML = '';
      frame.appendChild(iframe);
    });
    frame.appendChild(facade);
    figure.appendChild(frame);

    if (clip.title) {
      var cap = document.createElement('figcaption');
      cap.textContent = clip.title;
      cap.style.marginTop = '.5rem';
      figure.appendChild(cap);
    }
    return figure;
  }

  function readSelectedTags() {
    var m = /[?&]ctags=([^&]+)/.exec(location.search);
    return m ? decodeURIComponent(m[1]).split(',').filter(Boolean) : [];
  }

  function writeSelectedTags(set) {
    var params = new URLSearchParams(location.search);
    if (set.size > 0) params.set('ctags', Array.from(set).join(','));
    else params.delete('ctags');
    var qs = params.toString();
    history.replaceState(null, '', location.pathname + (qs ? '?' + qs : '') + location.hash);
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
