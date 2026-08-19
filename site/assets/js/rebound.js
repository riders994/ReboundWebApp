'use strict';

/*
 * Interactive rebound court. Adapted from the original 2017 public/script.js.
 * Changes from the original:
 *   - Reads config (predict endpoint + court SVG url) from window.REBOUND_CONFIG so the
 *     page controls relative paths.
 *   - Mounts into #court-container instead of the whole page.
 *   - POSTs application/json (was form-urlencoded with a JSON string as a key).
 * Still uses D3 v4 (d3.mouse / d3.event / contextmenu) to avoid a full v4->v7 rewrite.
 */

var CFG = window.REBOUND_CONFIG || {};
var ENDPOINT = CFG.endpoint || '/api/rebound/predict';
var COURT_SVG = CFG.svgUrl || 'ultimate.svg';

var shooter = false;
var offense = [];
var defense = [];
var payload = { bench: [] };

var rgbValues = [
  'rgb(25, 0, 255)', 'rgb(50, 0, 225)', 'rgb(75, 0, 200)', 'rgb(100, 0, 175)',
  'rgb(125, 0, 150)', 'rgb(150, 0, 125)', 'rgb(175, 0, 100)', 'rgb(200, 0, 75)',
  'rgb(225, 0, 50)', 'rgb(255, 0, 25)'
];

function Player(x, y, isOffense, isShooter) {
  this.x = x;
  this.y = y;
  this.isOffense = isOffense;
  this.isShooter = isShooter;
}

var banner = document.getElementById('banner');
var trigger = document.getElementById('run');
var clearBtn = document.getElementById('clear');

var court = d3.select('#court-container')
  .append('svg')
  .style('background-image', 'url(' + COURT_SVG + ')')
  .style('background-size', 'cover')
  .style('background-color', 'rgb(47, 71, 62)')
  .style('font-size', '12px')
  .style('user-select', 'none')
  .style('cursor', 'pointer')
  .attr('width', 500)
  .attr('height', 470)
  .style('border-radius', '10px')
  .style('box-shadow', '0 8px 30px rgba(0,0,0,0.45)')
  .on('click', placeOffender)
  .on('contextmenu', placeDefender);

trigger.addEventListener('click', logPayload);
clearBtn.addEventListener('click', restore);

function placeOffender() {
  var xy = d3.mouse(this);
  if (offense.length < 5) {
    var player = court.append('circle')
      .attr('class', 'player')
      .attr('cx', xy[0]).attr('cy', xy[1]).attr('r', 15)
      .attr('fill', 'white').attr('stroke', 'black').attr('stroke-width', '3')
      .property('isOffense', true);
    offense.push(player);
    payload.bench.push(new Player(xy[0] / 10, xy[1] / 10, true, false));
    offense[offense.length - 1].attr('id', payload.bench.length - 1);
    offense[offense.length - 1].property('xid', offense.length);
    court.append('g').append('text')
      .text('O' + offense.length)
      .attr('x', xy[0]).attr('y', xy[1])
      .attr('dx', -6).attr('dy', 4);
  }
  if ((window.event && window.event.shiftKey && !shooter) ||
      (offense.length === 5 && !shooter)) {
    offense[offense.length - 1].attr('stroke', 'gold');
    payload.bench[payload.bench.length - 1].isShooter = true;
    shooter = true;
  }
}

function placeDefender() {
  d3.event.preventDefault();
  var xy = d3.mouse(this);
  if (defense.length < 5) {
    var player = court.append('circle')
      .attr('class', 'player')
      .attr('cx', xy[0]).attr('cy', xy[1]).attr('r', 15)
      .attr('fill', 'white').attr('stroke', 'black').attr('stroke-width', '3');
    defense.push(player);
    payload.bench.push(new Player(xy[0] / 10, xy[1] / 10, false, false));
    defense[defense.length - 1].attr('id', payload.bench.length - 1);
    defense[defense.length - 1].property('xid', defense.length);
    court.append('g').append('text')
      .text('D' + defense.length)
      .attr('x', xy[0]).attr('y', xy[1])
      .attr('dx', -6).attr('dy', 4);
  }
}

function logPayload() {
  if (payload.bench.length < 10) {
    banner.innerText = 'Place all 10 players first';
    return;
  }
  banner.innerText = 'Running…';
  var xhr = new XMLHttpRequest();
  xhr.open('POST', ENDPOINT);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onreadystatechange = function () {
    if (xhr.readyState !== 4) return;
    if (xhr.status !== 200) {
      banner.innerText = 'Prediction failed (' + xhr.status + ')';
      return;
    }
    var response = JSON.parse(xhr.response);
    offense.forEach(function (offender) {
      offender.property('probability', response[offender.attr('id')].probability);
    });
    defense.forEach(function (defender) {
      defender.property('probability', response[defender.attr('id')].probability);
    });
    offense.concat(defense)
      .sort(function (a, b) { return a.property('probability') - b.property('probability'); })
      .forEach(function (man, index) {
        if (index === 9) man.attr('stroke', 'blue');
        man.transition().duration(1000).ease(d3.easeLinear)
          .attr('fill', rgbValues[index])
          .attr('cx', response[man.attr('id')].newx * 10)
          .attr('cy', response[man.attr('id')].newy * 10);
      });
    banner.innerText = 'Hover a player for its rebound probability';
  };
  d3.selectAll('circle').on('mouseover', handleMouseOver);
  xhr.send(JSON.stringify(payload));
}

function handleMouseOver() {
  var suffix = parseInt(d3.select(this).property('xid'), 10);
  var prob = d3.select(this).property('probability');
  var prefix = d3.select(this).property('isOffense') === true ? 'O' : 'D';
  banner.innerText = prefix + suffix + ' | P = ' + (prob != null ? Number(prob).toFixed(3) : '?');
}

function restore() {
  offense = [];
  defense = [];
  payload.bench = [];
  shooter = false;
  banner.innerText = 'Left-click offense · right-click defense · shift-click the shooter';
  court.selectAll('*').remove();
}
