'use strict';
let shooter = false;
let offense = [];
let defense = [];
const payload = {
  bench: new Array()
}

// const response = [
//   {newx: 45, newy: 40, probability: 0.8},
//   {newx: 15, newy: 15, probability: 0.1},
//   {newx: 15, newy: 5, probability: 0.2},
//   {newx: 11, newy: 33, probability: 0.9},
//   {newx: 40, newy: 5, probability: 0.3},
//   {newx: 10, newy: 5, probability: 0.4},
//   {newx: 19, newy: 7, probability: 0.7},
//   {newx: 14, newy: 23, probability: 0.5},
//   {newx: 37, newy: 2, probability: 0.99},
//   {newx: 26, newy: 18, probability: 0.6}
// ];

const rgbValues = [
  'rgb(25, 0, 255)',
  'rgb(50, 0, 225)',
  'rgb(75, 0, 200)',
  'rgb(100, 0, 175)',
  'rgb(125, 0, 150)',
  'rgb(150, 0, 125)',
  'rgb(175, 0, 100)',
  'rgb(200, 0, 75)',
  'rgb(225, 0, 50)',
  'rgb(255, 0, 25)'
];

class Player {
  constructor(x, y, isOffense, isShooter) {
    this.x = x;
    this.y = y;
    this.isOffense = isOffense;
    this.isShooter = isShooter;
  }
}

const banner = document.getElementById('banner');
const trigger = document.getElementById('run');
const clear = document.getElementById('clear');
const container = document.getElementById('container');
const court = d3.select('#container')
                .append('svg')
                .style('background-image', 'url(svgs/ultimate.svg)')
                .style('background-size', 'cover')
                .style('background-color', 'rgb(47, 71, 62)')
                .style('font-size', '12px')
                .style('user-select', 'none')
                .style('cursor', 'pointer')
                .attr('width', 500)
                .attr('height',  470)
                .style('border', '3px outset grey')
                .style('box-shadow', '4px 6px 33px 0px rgba(0,0,0,0.75)')
                .on('click', placeOffender)
                .on('contextmenu', placeDefender);

// const hoop = court.append('circle')
//                   .attr('cx', 250)
//                   .attr('cy', 416.5)
//                   .attr('r', 7.5)
//                   .attr('stroke', 'yellow');


trigger.addEventListener('click', logPayload);
clear.addEventListener('click', restore);

function placeOffender() {

  let xy = d3.mouse(this);

  if (offense.length < 5) {

    let player = court.append('circle')
                      .attr('class', 'player')
                      .attr('cx', xy[0])
                      .attr('cy', xy[1])
                      .attr('r', 15)
                      .attr('fill', 'white')
                      .attr('stroke', 'black')
                      .attr('stroke-width', '3')
                      .property('isOffense', true)

    offense.push(player)
    payload.bench.push(new Player(
      (xy[0] / 10),(xy[1] / 10), true, false
    ));
    offense[offense.length - 1].attr('id', payload.bench.length - 1);
    offense[offense.length - 1].property('xid', offense.length);

    let label = court.append('g')
                     .attr('dx', xy[0])
                     .attr('dy', xy[1])
                     .append('text')
                     .text(`O${offense.length}`)
                     .attr('x', xy[0])
                     .attr('y', xy[1])
                     .attr("dx", function(d){return -6})
                     .attr("dy", function(d){return 4})

  }

  if ((window.event.shiftKey && !shooter) ||
      (offense.length === 5 && !shooter)) {
    offense[offense.length - 1].attr('stroke', 'gold')
    payload.bench[payload.bench.length - 1].isShooter = true;
    shooter = true;
  }
//
}

function placeDefender() {
  d3.event.preventDefault();
  let xy = d3.mouse(this);
  if (defense.length < 5) {

    let player = court.append('circle')
                      .attr('class', 'player')
                      .attr('cx', xy[0])
                      .attr('cy', xy[1])
                      .attr('r', 15)
                      .attr('fill', 'white')
                      .attr('stroke', 'black')
                      .attr('stroke-width', '3');

    defense.push(player);
    payload.bench.push(new Player(
      (xy[0] / 10), (xy[1] / 10), false, false
    ));
    defense[defense.length - 1].attr('id', payload.bench.length - 1);
    defense[defense.length - 1].property('xid', defense.length);
    let label = court.append('g')
                     .attr('dx', xy[0])
                     .attr('dy', xy[1])
                     .append('text')
                     .text(`D${defense.length}`)
                     .attr('x', xy[0])
                     .attr('y', xy[1])
                     .attr("dx", function(d){return -6})
                     .attr("dy", function(d){return 4})
  }
}

function logPayload() {
  if (payload.bench.length < 10) {
    console.log('not enough players');
    return;
  }
  let xhr = new XMLHttpRequest();
  xhr.open('POST', 'http://34.197.45.92:8080/predict');
  xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
  xhr.onreadystatechange = () => {
    if (xhr.readyState == 4) {
      console.log(xhr.response);
      if (xhr.status == 200) {
        let response = JSON.parse(xhr.response);
        offense.forEach((offender, i) => {
          offender.property('probability', response[offender.attr('id')].probability);
        });

        defense.forEach((defender, i) => {
          defender.property('probability', response[defender.attr('id')].probability);
        });

        offense.concat(defense)
        .sort((a, b) => a.property('probability') - b.property('probability'))
        .forEach((man, index) => {
          if (index === 9) {
            man.attr('stroke', 'blue')
          }
          man.transition()
          .duration(1000)
          .ease(d3.easeLinear)
          .attr('fill', rgbValues[index])
          .attr('cx', response[man.attr('id')].newx * 10)
          .attr('cy', response[man.attr('id')].newy * 10)
        });
      }
    }
  };
  d3.selectAll('circle').on("mouseover", handleMouseOver)
  xhr.send(JSON.stringify(payload));
}

function handleMouseOver() {
  let prefix = null;
  let suffix = parseInt(d3.select(this).property('xid'), 10);
  let prob = d3.select(this).property('probability');
  if (d3.select(this).property('isOffense') === true) {
    prefix = 'O'
  } else {
    prefix = 'D'
  }
  banner.innerText = `${prefix}${suffix.toString()} | P = ${prob}`;
  console.log(d3.select(this).property('isOffense'));
}

function restore() {
  offense = [];
  defense = [];
  payload.bench = [];
  shooter = false;
  banner.innerText = 'Display';
  court.selectAll('*').remove();
}
//
