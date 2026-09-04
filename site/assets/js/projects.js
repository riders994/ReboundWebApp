'use strict';

// Renders the projects index from assets/data/projects.json:
//   - a highlight of featured + in-flight projects
//   - a paginated list of completed projects, newest first
// "featured" = the 5 projects with the most recent featured_at date (same rule as the CLI).
// Projects flagged "draft" are dropped before anything else runs, so a held-back project
// can't take up a featured slot. scripts/projects.py does the same, in the same order.
(function () {
  var DATA_URL = '../assets/data/projects.json';
  var PAGE_SIZE = 6;
  var FEATURED_LIMIT = 5;

  var highlightEl = document.getElementById('project-highlight');
  var completedEl = document.getElementById('project-completed');
  var pagerEl = document.getElementById('project-pager');
  if (!highlightEl) return;

  fetch(DATA_URL)
    .then(function (r) { return r.ok ? r.json() : { projects: [] }; })
    .then(function (data) { render((data.projects || []).filter(notDraft)); })
    .catch(function () { highlightEl.innerHTML = '<p class="muted">Couldn’t load projects.</p>'; });

  function notDraft(p) { return !p.draft; }

  function featuredSet(projects) {
    var dated = projects.filter(function (p) { return p.featured_at; });
    dated.sort(function (a, b) { return a.featured_at < b.featured_at ? 1 : -1; });
    var set = {};
    dated.slice(0, FEATURED_LIMIT).forEach(function (p) { set[p.slug] = true; });
    return set;
  }

  function render(projects) {
    var featured = featuredSet(projects);

    // Highlight: featured (by recency) then in-flight-not-featured.
    var highlight = projects
      .filter(function (p) { return featured[p.slug] || p.status === 'in-flight'; })
      .sort(function (a, b) {
        var fa = featured[a.slug] ? (a.featured_at || '') : '';
        var fb = featured[b.slug] ? (b.featured_at || '') : '';
        return fa < fb ? 1 : fa > fb ? -1 : 0;
      });

    highlightEl.innerHTML = '';
    if (highlight.length === 0) {
      highlightEl.innerHTML = '<p class="muted">No featured or in-flight projects yet.</p>';
    } else {
      var hg = document.createElement('div');
      hg.className = 'grid grid--2';
      highlight.forEach(function (p) { hg.appendChild(card(p, featured[p.slug])); });
      highlightEl.appendChild(hg);
    }

    // Completed list: completed and not already in the highlight, newest first.
    var inHighlight = {};
    highlight.forEach(function (p) { inHighlight[p.slug] = true; });
    var completed = projects
      .filter(function (p) { return p.status === 'completed' && !inHighlight[p.slug]; })
      .sort(function (a, b) { return (a.completed_at || '') < (b.completed_at || '') ? 1 : -1; });

    var page = 0;
    var pages = Math.max(1, Math.ceil(completed.length / PAGE_SIZE));

    function drawPage() {
      completedEl.innerHTML = '';
      if (completed.length === 0) {
        completedEl.innerHTML = '<p class="muted">No completed projects yet.</p>';
        pagerEl.innerHTML = '';
        return;
      }
      var cg = document.createElement('div');
      cg.className = 'grid grid--3';
      completed.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE)
        .forEach(function (p) { cg.appendChild(card(p, false)); });
      completedEl.appendChild(cg);

      pagerEl.innerHTML = '';
      if (pages > 1) {
        pagerEl.appendChild(pagerBtn('‹ Prev', page > 0, function () { page--; drawPage(); }));
        var ind = document.createElement('span');
        ind.className = 'pagination__info';
        ind.textContent = 'Page ' + (page + 1) + ' of ' + pages;
        pagerEl.appendChild(ind);
        pagerEl.appendChild(pagerBtn('Next ›', page < pages - 1, function () { page++; drawPage(); }));
      }
    }
    drawPage();
  }

  function pagerBtn(label, enabled, onClick) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'pagination__btn';
    b.textContent = label;
    b.disabled = !enabled;
    if (enabled) b.addEventListener('click', onClick);
    return b;
  }

  function card(p, isFeatured) {
    var a = document.createElement('a');
    a.className = 'card';
    a.href = (p.slug || '') + '.html';

    var thumb = document.createElement('img');
    thumb.className = 'card__thumb';
    thumb.alt = p.name || '';
    thumb.loading = 'lazy';
    thumb.src = '../assets/img/' + p.slug + '-thumb.png';
    thumb.onerror = function () { thumb.style.background = 'var(--bg-soft)'; thumb.removeAttribute('src'); };
    a.appendChild(thumb);

    var body = document.createElement('div');
    body.className = 'card__body';

    var meta = document.createElement('div');
    meta.className = 'card__meta';
    meta.appendChild(badge(p.status === 'completed' ? 'Completed' : 'In flight',
      p.status === 'completed' ? 'completed' : 'in-flight'));
    if (isFeatured) meta.appendChild(badge('★ Featured', 'featured'));
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

  function badge(text, kind) {
    var s = document.createElement('span');
    s.className = 'badge badge--' + kind;
    s.textContent = text;
    return s;
  }
})();
