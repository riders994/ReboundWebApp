'use strict';

/*
 * Interactive rebound court.
 *
 * Players are placed by dragging position bubbles from the picker panel, or by tapping
 * the empty court for a default lineup. The 2017 interaction (left-click offense,
 * right-click defense, shift-click shooter) is gone: contextmenu and shiftKey have no
 * touch equivalent, so that model made the demo unusable on a phone.
 *
 * Two things here are load-bearing and easy to "tidy" into bugs:
 *
 *   1. POSITION_WIRE. The UI says PG/SG/SF/PF/C; the wire says G/G-F/F/F-C/C. The model's
 *      POSITION_MAP has no PG/SG/SF/PF keys, and Player.role falls back to 3.0 ("F") for
 *      anything it doesn't recognise -- so posting the UI's own labels would silently turn
 *      every guard into a forward. The five strings below are the model's own names for
 *      the same five rungs (1, 5/3, 3, 11/3, 5).
 *
 *   2. Player ids are stable and mean nothing positional. The old code wrote a bench index
 *      into the DOM and read it back; removing a player shifted every later index and broke
 *      the mapping between a circle and its probability. State is the single source of
 *      truth now, and the response is matched to it positionally at send time.
 *
 * Still D3 v4 (CDN) for selections and transitions. Pointer Events -- not d3.drag -- do the
 * dragging: the bubble-to-court gesture crosses from an HTML button into the SVG, which is
 * not what d3.drag is shaped for, and setPointerCapture keeps the gesture attached no matter
 * what it passes over. d3.mouse is also unusable from a native pointer handler because it
 * reads the ambient d3.event, hence toCourt() below.
 */

(function () {

  // ----------------------------------------------------------------- constants

  var CFG = window.REBOUND_CONFIG || {};
  var ENDPOINT = CFG.endpoint || '/api/rebound/predict';
  var COURT_SVG = CFG.svgUrl || 'ultimate.svg';

  var COURT_W = 500;          // user units; 50 ft across at PX_PER_FT
  var COURT_H = 470;          // 47 ft down
  var PX_PER_FT = 10;
  var R = 15;                 // visible player radius (3 ft across)
  var HIT_R = 22;             // invisible touch pad, so targets stay usable on a phone
  var TEAM_SIZE = 5;

  var POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C'];

  // UI label -> the model's own position string. See the header comment before changing.
  var POSITION_WIRE = { PG: 'G', SG: 'G-F', SF: 'F', PF: 'F-C', C: 'C' };

  // Offensive spots, in court user units. The rim is at (250, 417.5): basket at the bottom
  // centre, per rebound-app/coordinates.py. PG sits above the arc, wings at ~22 ft, big men
  // on the blocks.
  var DEFAULT_OFFENSE = {
    PG: [250, 165], SG: [100, 250], SF: [400, 250], PF: [165, 355], C: [330, 380]
  };
  var RIM = [250, 417.5];

  var TAP_SLOP = 4;           // user units of movement still counted as a tap
  var TAP_MS = 400;

  // ----------------------------------------------------------------- state

  var state = {
    side: 'offense',          // which team the bubbles place onto
    phase: 'place',           // 'place' | 'shooter' | 'result'
    players: [],              // insertion order IS bench order
    selected: null,           // player id, for the remove badge
    nextId: 1,
    busy: false               // a request is in flight
  };

  function Player(pos, isOffense, x, y) {
    this.id = 'p' + (state.nextId++);
    this.pos = pos;
    this.isOffense = isOffense;
    this.isShooter = false;
    this.x = x;
    this.y = y;
    this.prob = null;
    this.newx = null;
    this.newy = null;
  }

  // ----------------------------------------------------------------- dom handles

  var banner = document.getElementById('banner');
  var runBtn = document.getElementById('run');
  var clearBtn = document.getElementById('clear');
  var picker = document.querySelector('.picker');
  var countEl = picker.querySelector('.picker__count');
  var hintEl = picker.querySelector('.picker__hint');
  var bubblesEl = picker.querySelector('.picker__bubbles');
  var shooterWrap = picker.querySelector('.picker__shooter');
  var shooterRow = picker.querySelector('.picker__shooter-chips');
  var resultsWrap = picker.querySelector('.picker__results');
  var resultsList = picker.querySelector('.picker__results-list');
  var fillBtn = picker.querySelector('.picker__fill');

  var svg = d3.select('#court-container')
    .append('svg')
    .attr('class', 'court')
    .attr('viewBox', '0 0 ' + COURT_W + ' ' + COURT_H)
    .attr('width', COURT_W)
    .attr('height', COURT_H)
    .style('background-image', 'url(' + COURT_SVG + ')')
    .style('background-size', '100% 100%');

  var svgNode = svg.node();
  var playersG = svg.append('g').attr('class', 'court__players');
  var ghostG = svg.append('g').attr('class', 'court__ghost').style('display', 'none');

  ghostG.append('circle').attr('r', R);
  ghostG.append('text').attr('class', 'court__label').attr('dy', '.35em');

  // ----------------------------------------------------------------- helpers

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* Client coordinates -> court user units.
   *
   * Goes through the SVG's CTM rather than assuming one CSS pixel is one user unit -- with a
   * viewBox those differ at every size but 500px wide, which is the whole point of the
   * viewBox. Read the CTM fresh each gesture: it changes on scroll, resize and zoom. */
  function toCourt(clientX, clientY) {
    var pt = svgNode.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    var p = pt.matrixTransform(svgNode.getScreenCTM().inverse());
    return [p.x, p.y];
  }

  function inCourt(xy) {
    return xy[0] >= 0 && xy[0] <= COURT_W && xy[1] >= 0 && xy[1] <= COURT_H;
  }

  function counts() {
    var o = 0, d = 0;
    state.players.forEach(function (p) { p.isOffense ? o++ : d++; });
    return { offense: o, defense: d };
  }

  function byId(id) {
    for (var i = 0; i < state.players.length; i++) {
      if (state.players[i].id === id) return state.players[i];
    }
    return null;
  }

  function shooterOf() {
    for (var i = 0; i < state.players.length; i++) {
      if (state.players[i].isShooter) return state.players[i];
    }
    return null;
  }

  /* Mirrors rebounding/serve.py _validate, in the same order, so the UI never lets the user
   * build a request the backend will reject. Returns the first unmet requirement. */
  function validity() {
    var c = counts();
    if (c.offense < TEAM_SIZE) {
      return { ok: false, message: 'Place ' + (TEAM_SIZE - c.offense) + ' more on offense' };
    }
    if (c.defense < TEAM_SIZE) {
      return { ok: false, message: 'Place ' + (TEAM_SIZE - c.defense) + ' more on defense' };
    }
    if (!shooterOf()) return { ok: false, message: 'Choose the shooter' };
    return { ok: true, message: '' };
  }

  function reducedMotion() {
    return window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /* Any edit invalidates a result: drop the probabilities and hand the fill colour back to
   * CSS, or the inline attribute from the last run wins over the stylesheet forever. */
  function clearResults() {
    state.players.forEach(function (p) {
      p.prob = null; p.newx = null; p.newy = null;
    });
    playersG.selectAll('circle.court__dot').attr('fill', null);
    playersG.selectAll('g.court__player').classed('is-best', false);
    if (state.phase === 'result') state.phase = 'place';
  }

  // ----------------------------------------------------------------- placement

  function addPlayer(pos, isOffense, x, y) {
    var c = counts();
    if ((isOffense ? c.offense : c.defense) >= TEAM_SIZE) return null;
    clearResults();
    var p = new Player(pos, isOffense, clamp(x, R, COURT_W - R), clamp(y, R, COURT_H - R));
    state.players.push(p);
    if (state.players.length === 10 && !shooterOf()) state.phase = 'shooter';
    return p;
  }

  function removePlayer(id) {
    for (var i = 0; i < state.players.length; i++) {
      if (state.players[i].id === id) {
        clearResults();
        state.players.splice(i, 1);
        if (state.selected === id) state.selected = null;
        if (state.phase === 'shooter' && state.players.length < 10) state.phase = 'place';
        return;
      }
    }
  }

  /* Each defender stands a couple of feet goalside of his man -- derived rather than
   * tabulated so the relationship stays visible. */
  function defenderFor(spot) {
    var dx = RIM[0] - spot[0], dy = RIM[1] - spot[1];
    var len = Math.sqrt(dx * dx + dy * dy) || 1;
    var t = clamp(0.20 * len, 25, 45);
    return [spot[0] + t * dx / len, spot[1] + t * dy / len];
  }

  function defaultFill() {
    state.players = [];
    state.selected = null;
    clearResults();
    POSITIONS.forEach(function (pos) {
      var s = DEFAULT_OFFENSE[pos];
      state.players.push(new Player(pos, true, s[0], s[1]));
    });
    POSITIONS.forEach(function (pos) {
      var d = defenderFor(DEFAULT_OFFENSE[pos]);
      state.players.push(new Player(pos, false, d[0], d[1]));
    });
    state.phase = 'shooter';
    render();
  }

  function setShooter(id) {
    var p = byId(id);
    if (!p) return;
    if (!p.isOffense) {
      say('The shooter must be on offense');
      return;
    }
    state.players.forEach(function (q) { q.isShooter = false; });
    p.isShooter = true;
    if (state.phase === 'shooter') state.phase = 'place';
    render();
  }

  function say(msg) { banner.textContent = msg; }

  // ----------------------------------------------------------------- render

  function label(p) { return p.pos; }

  function render() {
    var join = playersG.selectAll('g.court__player')
      .data(state.players, function (d) { return d.id; });

    join.exit().remove();

    var enter = join.enter().append('g')
      .attr('class', 'court__player')
      .attr('tabindex', 0)
      .attr('role', 'button');

    // Invisible pad first so it sits under the visible dot but still catches the pointer.
    enter.append('circle')
      .attr('class', 'court__hit')
      .attr('r', HIT_R)
      .attr('fill', 'none')
      .attr('pointer-events', 'all');
    enter.append('circle').attr('class', 'court__dot').attr('r', R);
    enter.append('text').attr('class', 'court__label').attr('dy', '.35em');
    enter.append('g').attr('class', 'court__remove').style('display', 'none')
      .call(function (g) {
        g.append('circle').attr('r', 9).attr('cx', R - 1).attr('cy', -R + 1);
        g.append('text').text('×')
          .attr('x', R - 1).attr('y', -R + 1).attr('dy', '.35em');
      });

    enter.on('pointerdown', onPlayerPointerDown)
      .on('keydown', onPlayerKeyDown);

    var all = enter.merge(join);

    all.attr('transform', function (d) { return 'translate(' + d.x + ',' + d.y + ')'; })
      .attr('class', function (d) {
        return 'court__player' +
          (d.isOffense ? ' is-offense' : ' is-defense') +
          (d.isShooter ? ' is-shooter' : '') +
          (state.selected === d.id ? ' is-selected' : '');
      })
      .attr('aria-label', function (d) {
        return (d.isOffense ? 'Offense ' : 'Defense ') + d.pos +
          ', ' + Math.round(d.x / PX_PER_FT) + ' feet across, ' +
          Math.round(d.y / PX_PER_FT) + ' feet down' +
          (d.isShooter ? ', shooter' : '') + '. Press Delete to remove.';
      });

    all.select('text.court__label').text(label);
    all.select('g.court__remove')
      .style('display', function (d) { return state.selected === d.id ? null : 'none'; });

    svg.classed('court--picking-shooter', state.phase === 'shooter');

    renderPanel();
  }

  function renderPanel() {
    var c = counts();
    countEl.textContent = 'Offense ' + c.offense + '/5 · Defense ' + c.defense + '/5';

    // Bubbles never disable per position -- repeats are legal, five centres is a lineup.
    // They dim only when the selected side is full.
    var full = (state.side === 'offense' ? c.offense : c.defense) >= TEAM_SIZE;
    Array.prototype.forEach.call(bubblesEl.querySelectorAll('.bubble'), function (b) {
      b.setAttribute('aria-disabled', full ? 'true' : 'false');
    });

    Array.prototype.forEach.call(picker.querySelectorAll('[data-side]'), function (b) {
      var on = b.getAttribute('data-side') === state.side;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });

    // Shooter chips appear as soon as there is an offense to choose from, and stay -- so
    // re-picking never means re-entering a mode.
    var offense = state.players.filter(function (p) { return p.isOffense; });
    shooterWrap.hidden = offense.length < TEAM_SIZE;
    shooterRow.innerHTML = '';
    offense.forEach(function (p) {
      var chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip' + (p.isShooter ? ' is-active' : '');
      chip.setAttribute('aria-pressed', p.isShooter ? 'true' : 'false');
      chip.textContent = p.pos;
      chip.addEventListener('click', function () { setShooter(p.id); });
      shooterRow.appendChild(chip);
    });

    var ranked = state.players.filter(function (p) { return p.prob != null; });
    resultsWrap.hidden = ranked.length === 0;
    if (ranked.length) {
      ranked.sort(function (a, b) { return b.prob - a.prob; });
      resultsList.innerHTML = '';
      ranked.forEach(function (p, i) {
        var li = document.createElement('li');
        li.textContent = (i + 1) + '. ' + (p.isOffense ? 'O' : 'D') + ' · ' +
          p.pos + ' — ' + p.prob.toFixed(3);
        resultsList.appendChild(li);
      });
    }

    var v = validity();
    runBtn.disabled = !v.ok || state.busy;
    clearBtn.disabled = state.players.length === 0;

    if (state.busy) return;
    if (state.phase === 'shooter') {
      hintEl.textContent = 'Choose the shooter — tap an offensive player, or pick from the list.';
      say('Choose the shooter');
    } else if (!v.ok) {
      var next = nextInSequence();
      hintEl.textContent = next
        ? 'Click the court to drop ' + (next.isOffense ? 'offense ' : 'defense ') + next.pos +
          ', or drag any bubble.'
        : 'Drag a bubble onto the court, or press Enter to drop it.';
      say(v.message);
    } else if (state.phase === 'result') {
      hintEl.textContent = 'Tap a player to read its probability. Run again to re-sample.';
    } else {
      hintEl.textContent = 'Ready. Hit Run — or drag players to adjust first.';
      say('Ready to run');
    }
  }

  // ----------------------------------------------------------------- dragging

  var drag = null;
  /* A tap on a bubble fires pointerup AND then click, and both place a player -- so the
     pointer path swallows the click that follows it. Keyboard Enter/Space fires click with
     no pointer sequence at all, which is exactly the case that must still get through. */
  var swallowClick = false;

  function showGhost(pos, isOffense, xy) {
    ghostG.style('display', null)
      .attr('transform', 'translate(' + xy[0] + ',' + xy[1] + ')')
      .attr('class', 'court__ghost ' + (isOffense ? 'is-offense' : 'is-defense'));
    ghostG.select('text').text(pos);
  }

  function hideGhost() { ghostG.style('display', 'none'); }

  function onBubblePointerDown(e) {
    if (this.getAttribute('aria-disabled') === 'true') return;
    e.preventDefault();
    this.setPointerCapture(e.pointerId);
    this.classList.add('is-dragging');
    drag = {
      kind: 'new',
      pos: this.getAttribute('data-pos'),
      isOffense: state.side === 'offense',
      node: this,
      pointerId: e.pointerId,
      moved: 0,
      start: Date.now()
    };
  }

  function onPlayerPointerDown(d) {
    var e = d3.event;
    e.preventDefault();
    var g = this;
    g.setPointerCapture(e.pointerId);
    drag = {
      kind: 'move',
      id: d.id,
      node: g,
      pointerId: e.pointerId,
      moved: 0,
      start: Date.now(),
      from: toCourt(e.clientX, e.clientY)
    };
  }

  function onPointerMove(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    var xy = toCourt(e.clientX, e.clientY);
    if (drag.kind === 'new') {
      drag.moved += 1;
      if (inCourt(xy)) showGhost(drag.pos, drag.isOffense, xy); else hideGhost();
      return;
    }
    // 'move': write straight to the node. A full render at pointer rate would rebuild the
    // element under the captured pointer and fight the gesture.
    var p = byId(drag.id);
    if (!p) return;
    drag.moved = Math.max(drag.moved,
      Math.abs(xy[0] - drag.from[0]) + Math.abs(xy[1] - drag.from[1]));
    drag.at = xy;
    d3.select(drag.node)
      .attr('transform', 'translate(' + xy[0] + ',' + xy[1] + ')')
      .classed('is-removing', !inCourt(xy));
  }

  function onPointerUp(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    var xy = toCourt(e.clientX, e.clientY);
    var d = drag;
    drag = null;
    hideGhost();
    try { d.node.releasePointerCapture(d.pointerId); } catch (err) { /* already gone */ }

    if (d.kind === 'new') {
      d.node.classList.remove('is-dragging');
      swallowClick = true;
      var tap = d.moved < 3 && (Date.now() - d.start) < TAP_MS;
      if (tap) { placeAtDefault(d.pos, d.isOffense); return; }
      if (inCourt(xy)) { addPlayer(d.pos, d.isOffense, xy[0], xy[1]); render(); }
      return;
    }

    var p = byId(d.id);
    if (!p) { render(); return; }
    if (d.moved < TAP_SLOP && (Date.now() - d.start) < TAP_MS) {
      onPlayerTap(p);
      return;
    }
    if (inCourt(xy)) {
      clearResults();
      p.x = clamp(xy[0], R, COURT_W - R);
      p.y = clamp(xy[1], R, COURT_H - R);
    } else {
      removePlayer(p.id);
    }
    render();
  }

  function onPointerCancel(e) {
    if (!drag || e.pointerId !== drag.pointerId) return;
    var d = drag;
    drag = null;
    hideGhost();
    if (d.kind === 'new') d.node.classList.remove('is-dragging');
    render();
  }

  function onPlayerTap(p) {
    if (state.phase === 'shooter') { setShooter(p.id); return; }
    if (p.prob != null) {
      say(p.pos + ' (' + (p.isOffense ? 'offense' : 'defense') + ') · P = ' +
        p.prob.toFixed(3));
      return;
    }
    state.selected = (state.selected === p.id) ? null : p.id;
    render();
  }

  // Placing without a drag: keyboard, or a tap on a bubble. Uses the default spot, nudged
  // if it is taken, so repeated presses never stack players exactly on top of each other.
  function placeAtDefault(pos, isOffense) {
    var base = DEFAULT_OFFENSE[pos];
    var spot = isOffense ? base : defenderFor(base);
    var n = 0;
    while (n < 12 && occupied(spot)) {
      n++;
      spot = [spot[0] + 18, spot[1] + 12];
    }
    if (addPlayer(pos, isOffense, spot[0], spot[1])) render();
  }

  function occupied(xy) {
    return state.players.some(function (p) {
      return Math.abs(p.x - xy[0]) < R && Math.abs(p.y - xy[1]) < R;
    });
  }

  // ----------------------------------------------------------------- court taps

  /* The next slot in the default order: offense PG->C, then defense PG->C.
   *
   * Picks the first position that side is MISSING rather than indexing by count. On an empty
   * court the two are identical -- ten clicks still give PG,SG,SF,PF,C twice -- but they
   * diverge after an edit, and missing-first is the one that behaves: remove the SG and the
   * next click gives you an SG back, not a second C. Repeats stay legal, they just come from
   * dragging a bubble deliberately rather than from a counter drifting. */
  function nextInSequence() {
    var sides = [true, false];
    for (var s = 0; s < sides.length; s++) {
      var isOffense = sides[s];
      var taken = state.players
        .filter(function (p) { return p.isOffense === isOffense; })
        .map(function (p) { return p.pos; });
      if (taken.length >= TEAM_SIZE) continue;
      for (var i = 0; i < POSITIONS.length; i++) {
        if (taken.indexOf(POSITIONS[i]) === -1) {
          return { pos: POSITIONS[i], isOffense: isOffense };
        }
      }
      // Every label already used on this side (all repeats) but the side isn't full.
      return { pos: POSITIONS[taken.length], isOffense: isOffense };
    }
    return null;
  }

  /* A click on bare court drops the NEXT player in the sequence, at the point clicked --
   * the click says where, the sequence says who. (Filling all ten from one click would make
   * the click location meaningless; that behaviour lives on the "Fill default lineup"
   * button, where it is asked for explicitly.) */
  svg.on('click', function () {
    if (d3.event.target !== svgNode) return;   // a player handled it
    if (state.selected) { state.selected = null; render(); return; }
    var next = nextInSequence();
    if (!next) {
      say(shooterOf() ? 'All ten are placed — hit Run' : 'All ten are placed — choose the shooter');
      return;
    }
    var xy = d3.mouse(svgNode);
    addPlayer(next.pos, next.isOffense, xy[0], xy[1]);
    render();
  });

  // ----------------------------------------------------------------- keyboard

  function onPlayerKeyDown(d) {
    var e = d3.event;
    var step = e.shiftKey ? 50 : 10;
    var moved = true;
    if (e.key === 'ArrowLeft') d.x = clamp(d.x - step, R, COURT_W - R);
    else if (e.key === 'ArrowRight') d.x = clamp(d.x + step, R, COURT_W - R);
    else if (e.key === 'ArrowUp') d.y = clamp(d.y - step, R, COURT_H - R);
    else if (e.key === 'ArrowDown') d.y = clamp(d.y + step, R, COURT_H - R);
    else if (e.key === 'Delete' || e.key === 'Backspace') { removePlayer(d.id); render(); return; }
    else if (e.key === 'Enter' || e.key === ' ') { onPlayerTap(d); return; }
    else moved = false;
    if (moved) { e.preventDefault(); clearResults(); render(); }
  }

  // ----------------------------------------------------------------- run

  function run() {
    var v = validity();
    if (!v.ok || state.busy) { say(v.message); return; }

    state.busy = true;
    renderPanel();
    say('Running…');

    var bench = state.players.map(function (p) {
      return {
        x: p.x / PX_PER_FT,
        y: p.y / PX_PER_FT,
        isOffense: p.isOffense,
        isShooter: p.isShooter,
        position: POSITION_WIRE[p.pos]
      };
    });

    var xhr = new XMLHttpRequest();
    xhr.open('POST', ENDPOINT);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.timeout = 30000;

    xhr.onerror = function () {
      state.busy = false;
      say('Couldn’t reach the model. Check your connection.');
      renderPanel();
    };
    xhr.ontimeout = function () {
      state.busy = false;
      say('The model took too long. Try again.');
      renderPanel();
    };
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) return;
      state.busy = false;
      if (xhr.status !== 200) {
        // The backend sends a human sentence for PlacementError; prefer it to a bare code,
        // so any UI/API drift shows up as words instead of silence.
        var msg = 'Prediction failed (' + xhr.status + ')';
        try {
          var body = JSON.parse(xhr.response);
          if (body && body.error) msg = body.error;
        } catch (err) { /* keep the status code */ }
        say(msg);
        renderPanel();
        return;
      }
      applyResult(JSON.parse(xhr.response));
    };
    xhr.send(JSON.stringify({ bench: bench }));
  }

  /* The response is positional against state.players as it was at send time -- no ids in the
   * DOM, no index arithmetic. */
  function applyResult(response) {
    var max = 0;
    response.forEach(function (r, i) {
      var p = state.players[i];
      if (!p) return;
      p.prob = r.probability;
      p.newx = r.newx * PX_PER_FT;
      p.newy = r.newy * PX_PER_FT;
      if (r.probability > max) max = r.probability;
    });

    // Colour by probability, not by rank: the old ramp indexed a fixed 10-swatch array by
    // rank, so a ten-way tie looked exactly like a runaway favourite.
    var scale = d3.scaleLinear().domain([0, max || 1])
      .range(['#1900ff', '#ff0019']).interpolate(d3.interpolateRgb);

    state.phase = 'result';
    state.selected = null;

    var best = null;
    state.players.forEach(function (p) {
      if (!best || p.prob > best.prob) best = p;
    });

    var dur = reducedMotion() ? 0 : 1000;
    playersG.selectAll('g.court__player')
      .each(function (d) {
        // A short/garbled response must not write translate(null,null); leave the player
        // where the user put them instead.
        if (d.newx == null || d.newy == null) return;
        d.x = d.newx;
        d.y = d.newy;
      })
      .classed('is-best', function (d) { return best && d.id === best.id; })
      .transition().duration(dur).ease(d3.easeLinear)
      .attr('transform', function (d) { return 'translate(' + d.x + ',' + d.y + ')'; })
      .select('circle.court__dot')
      .attr('fill', function (d) { return scale(d.prob); });

    renderPanel();
    say('Tap a player for its probability · ranked list in the panel');
  }

  function clearAll() {
    state.players = [];
    state.selected = null;
    state.phase = 'place';
    state.nextId = 1;
    render();
  }

  // ----------------------------------------------------------------- wiring

  Array.prototype.forEach.call(bubblesEl.querySelectorAll('.bubble'), function (b) {
    b.addEventListener('pointerdown', onBubblePointerDown);
    b.addEventListener('click', function (e) {
      // Only the keyboard (or an assistive click) reaches this: a pointer tap already
      // placed the player in onPointerUp and set swallowClick.
      if (swallowClick) { swallowClick = false; return; }
      if (drag) return;
      e.preventDefault();
      if (b.getAttribute('aria-disabled') === 'true') return;
      placeAtDefault(b.getAttribute('data-pos'), state.side === 'offense');
    });
  });

  Array.prototype.forEach.call(picker.querySelectorAll('[data-side]'), function (b) {
    b.addEventListener('click', function () {
      state.side = b.getAttribute('data-side');
      renderPanel();
    });
  });

  fillBtn.addEventListener('click', defaultFill);
  runBtn.addEventListener('click', run);
  clearBtn.addEventListener('click', clearAll);

  document.addEventListener('pointermove', onPointerMove);
  document.addEventListener('pointerup', onPointerUp);
  document.addEventListener('pointercancel', onPointerCancel);
  document.addEventListener('lostpointercapture', onPointerCancel);

  if (!('PointerEvent' in window)) {
    hintEl.textContent = 'Tap a bubble to place a player.';
  }

  render();

})();
