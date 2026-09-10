const test = require('node:test');
const assert = require('node:assert/strict');
const {matchCue} = require('../src/storylight/static/demo-cues.js');
const {scenes} = require('../examples/alice-demo.json');

test('English and accented French cues select prepared scenes with word boundaries', () => {
  assert.equal(matchCue('the white rabbit ran', scenes).scene_id, 'white-rabbit');
  assert.equal(matchCue('the Cheshire cat grinned', scenes).scene_id, 'cheshire-cat');
  assert.equal(matchCue('a tea party!', scenes).scene_id, 'tea-party');
  assert.equal(matchCue('Une partie de croquét avec un flamant rose.', scenes).scene_id, 'croquet-fr');
  assert.equal(matchCue('La rène joue au procès avec un flamand rose.', scenes).scene_id, 'croquet-fr');
  assert.equal(matchCue('white rabbitfish', scenes), null);
  assert.equal(matchCue('hello', scenes), null);
});

test('cumulative speech selects the last cue instead of returning to the first scene', () => {
  assert.equal(matchCue('white rabbit then Cheshire cat then tea party', scenes).scene_id, 'tea-party');
});
