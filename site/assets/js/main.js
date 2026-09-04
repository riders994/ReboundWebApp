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

  // Projects carousel (Resume page): featured + in-flight projects, horizontally scrollable.
  var projectCarousel = document.querySelector('[data-project-carousel]');
  if (projectCarousel) initProjectCarousel(projectCarousel);

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
      var sortDir = readSort();   // 'new' (default, newest first) or 'old'
      var page = 1;
      var PAGE_SIZE = 9;          // 3×3 grid per page

      el.innerHTML = '';
      var filterBar = document.createElement('div');
      filterBar.className = 'filters';
      var grid = document.createElement('div');
      grid.className = 'grid grid--3 clips-grid';
      var pager = document.createElement('nav');
      pager.className = 'pager';
      pager.setAttribute('aria-label', 'Clip pages');
      el.appendChild(filterBar);
      el.appendChild(grid);
      el.appendChild(pager);

      function passes(clip) {
        // Faceted: within a category any selected tag matches (OR); across categories all
        // active categories must match (AND).
        return groups.every(function (g) {
          var picked = g.codes.filter(function (code) { return selected.has(code); });
          if (picked.length === 0) return true;
          return picked.some(function (code) { return clip.tags.indexOf(code) !== -1; });
        });
      }

      function byDate(a, b) {
        // Undated clips always sort last, regardless of direction.
        if (!a.date || !b.date) return (!a.date ? 1 : 0) - (!b.date ? 1 : 0);
        if (a.date === b.date) return 0;
        var cmp = a.date < b.date ? -1 : 1;
        return sortDir === 'old' ? cmp : -cmp;
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
          clear.addEventListener('click', function () { selected.clear(); writeSelectedTags(selected); page = 1; render(); });
          filterBar.appendChild(clear);
        }

        // sort control (always shown)
        var sortRow = document.createElement('div');
        sortRow.className = 'filter-group';
        var sortLab = document.createElement('span');
        sortLab.className = 'filter-group__label';
        sortLab.textContent = 'Sort';
        sortRow.appendChild(sortLab);
        [['new', 'Newest'], ['old', 'Oldest']].forEach(function (opt) {
          var chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'chip' + (sortDir === opt[0] ? ' is-active' : '');
          chip.setAttribute('aria-pressed', sortDir === opt[0] ? 'true' : 'false');
          chip.textContent = opt[1];
          chip.addEventListener('click', function () {
            if (sortDir === opt[0]) return;
            sortDir = opt[0]; writeSort(sortDir); page = 1; render();
          });
          sortRow.appendChild(chip);
        });
        filterBar.appendChild(sortRow);

        // grid: filter → sort → paginate
        var visible = clips.filter(passes).sort(byDate);
        grid.innerHTML = '';
        pager.innerHTML = '';
        if (visible.length === 0) {
          grid.innerHTML = '<p class="muted">No clips match those tags.</p>';
          return;
        }
        var pages = Math.ceil(visible.length / PAGE_SIZE);
        if (page > pages) page = pages;
        var start = (page - 1) * PAGE_SIZE;
        visible.slice(start, start + PAGE_SIZE).forEach(function (clip) {
          grid.appendChild(buildClipCard(clip));
        });
        renderPager(pages);
      }

      function renderPager(pages) {
        if (pages <= 1) return;
        pager.appendChild(pagerButton('‹ Prev', page > 1, function () { page -= 1; render(); focusClips(); }));
        var status = document.createElement('span');
        status.className = 'pager__status';
        status.textContent = 'Page ' + page + ' of ' + pages;
        pager.appendChild(status);
        pager.appendChild(pagerButton('Next ›', page < pages, function () { page += 1; render(); focusClips(); }));
      }

      function focusClips() {
        // Keep the grid in view when paging from the bottom of a long page.
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

      render();
    });
  }

  function normClip(item) {
    if (typeof item === 'string') return { id: item, title: '', date: '', tags: [] };
    return {
      id: item.id, title: item.title || '', date: item.date || '',
      tags: Array.isArray(item.tags) ? item.tags : []
    };
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

  function pagerButton(label, enabled, onClick) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'pager__btn';
    b.textContent = label;
    b.disabled = !enabled;
    if (enabled) b.addEventListener('click', onClick);
    return b;
  }

  function readSort() {
    var m = /[?&]csort=(new|old)\b/.exec(location.search);
    return m ? m[1] : 'new';
  }

  function writeSort(dir) {
    var params = new URLSearchParams(location.search);
    if (dir === 'old') params.set('csort', 'old');
    else params.delete('csort');
    var qs = params.toString();
    history.replaceState(null, '', location.pathname + (qs ? '?' + qs : '') + location.hash);
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

  // --- Projects carousel -----------------------------------------------------
  // Selection + card markup mirror assets/js/projects.js so the Resume carousel and the
  // Projects page agree on what "featured" means; paths are configurable via data- attrs
  // because this runs off the projects/ directory. Both also share the default-cover
  // deal in assets/js/thumbs.js, so a project keeps the same cover on either page.
  var FEATURED_LIMIT = 5;

  function initProjectCarousel(el) {
    var dataUrl = el.getAttribute('data-project-carousel');
    var pageBase = el.getAttribute('data-project-page-base') || '';
    var thumbBase = el.getAttribute('data-project-thumb-base') || '';
    fetch(dataUrl)
      .then(function (r) { return r.ok ? r.json() : { projects: [] }; })
      .then(function (data) {
        // Drop drafts before rendering, matching projects.js and scripts/projects.py.
        var live = (data.projects || []).filter(function (p) { return !p.draft; });
        renderProjectCarousel(el, live, pageBase, thumbBase);
      })
      .catch(function () { el.innerHTML = '<p class="muted">Couldn’t load projects.</p>'; });
  }

  function featuredProjectSet(projects) {
    // The FEATURED_LIMIT projects with the most recent featured_at (same rule as projects.js).
    var dated = projects.filter(function (p) { return p.featured_at; });
    dated.sort(function (a, b) { return a.featured_at < b.featured_at ? 1 : -1; });
    var set = {};
    dated.slice(0, FEATURED_LIMIT).forEach(function (p) { set[p.slug] = true; });
    return set;
  }

  function renderProjectCarousel(el, projects, pageBase, thumbBase) {
    var featured = featuredProjectSet(projects);
    var items = projects
      .filter(function (p) { return featured[p.slug] || p.status === 'in-flight'; })
      .sort(function (a, b) {
        var fa = featured[a.slug] ? (a.featured_at || '') : '';
        var fb = featured[b.slug] ? (b.featured_at || '') : '';
        return fa < fb ? 1 : fa > fb ? -1 : 0;
      });

    el.innerHTML = '';
    if (items.length === 0) {
      el.innerHTML = '<p class="muted">No featured or in-flight projects yet.</p>';
      return;
    }

    // Default covers for the items with no thumb of their own, spread across the track.
    var thumbs = window.CardThumbs
      ? window.CardThumbs.prepare(items.map(function (p) { return p.slug || ''; }), thumbBase)
      : null;

    var track = document.createElement('div');
    track.className = 'carousel__track';
    items.forEach(function (p) { track.appendChild(projectCard(p, featured[p.slug], pageBase, thumbBase, thumbs)); });

    var prev = carouselArrow('‹', 'prev');
    var next = carouselArrow('›', 'next');
    el.appendChild(prev);
    el.appendChild(track);
    el.appendChild(next);

    function step(dir) {
      var card = track.querySelector('.card');
      var by = card ? card.getBoundingClientRect().width + 24 : track.clientWidth * 0.8;
      track.scrollBy({ left: dir * by, behavior: 'smooth' });
    }
    prev.addEventListener('click', function () { step(-1); });
    next.addEventListener('click', function () { step(1); });

    function updateArrows() {
      // Threshold absorbs sub-pixel scroll positions left by scroll-snap (e.g. 1.5px).
      var max = track.scrollWidth - track.clientWidth;
      prev.disabled = track.scrollLeft <= 4;
      next.disabled = track.scrollLeft >= max - 4;
    }
    track.addEventListener('scroll', updateArrows);
    window.addEventListener('resize', updateArrows);
    updateArrows();
  }

  function carouselArrow(glyph, kind) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'carousel__btn carousel__btn--' + kind;
    b.setAttribute('aria-label', kind === 'prev' ? 'Previous projects' : 'Next projects');
    b.textContent = glyph;
    return b;
  }

  function projectCard(p, isFeatured, pageBase, thumbBase, thumbs) {
    var a = document.createElement('a');
    a.className = 'card';
    a.href = pageBase + (p.slug || '') + '.html';

    var thumb = document.createElement('img');
    thumb.className = 'card__thumb';
    thumb.alt = p.name || '';
    thumb.loading = 'lazy';
    thumb.src = thumbBase + p.slug + '-thumb.png';
    thumb.onerror = function () {
      thumb.onerror = null;
      if (thumbs) thumbs.apply(thumb, p.slug || '');
      else { thumb.style.background = 'var(--bg-soft)'; thumb.removeAttribute('src'); }
    };
    a.appendChild(thumb);

    var body = document.createElement('div');
    body.className = 'card__body';

    var meta = document.createElement('div');
    meta.className = 'card__meta';
    meta.appendChild(projectBadge(p.status === 'completed' ? 'Completed' : 'In flight',
      p.status === 'completed' ? 'completed' : 'in-flight'));
    if (isFeatured) meta.appendChild(projectBadge('★ Featured', 'featured'));
    body.appendChild(meta);

    var h = document.createElement('h3');
    h.className = 'card__title';
    h.textContent = p.name || p.slug;
    body.appendChild(h);

    if (p.blurb) {
      var blurb = document.createElement('p');
      blurb.className = 'card__blurb';
      blurb.textContent = p.blurb;
      body.appendChild(blurb);
    }

    if (p.tags && p.tags.length) {
      var tags = document.createElement('ul');
      tags.className = 'tags';
      p.tags.forEach(function (t) {
        var li = document.createElement('li');
        li.className = 'tag';
        li.textContent = t;
        tags.appendChild(li);
      });
      body.appendChild(tags);
    }

    a.appendChild(body);
    return a;
  }

  function projectBadge(text, kind) {
    var s = document.createElement('span');
    s.className = 'badge badge--' + kind;
    s.textContent = text;
    return s;
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
