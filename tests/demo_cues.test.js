const test = require('node:test');
const assert = require('node:assert/strict');
const {matchCue} = require('../src/storylight/static/demo-cues.js');
const {scenes} = require('../examples/alice-demo.json');

test('complete English and French sentences select their prepared artwork', () => {
  for (const scene of scenes) {
    assert.equal(matchCue(scene.spoken_example, scenes).scene_id, scene.scene_id);
  }
  assert.equal(matchCue(' LA REINE tient un flamand rose dans le jardin! ', scenes).scene_id,
    'croquet-fr');
  assert.equal(matchCue('A hair sits beside a teapot in a garden.', scenes).scene_id,
    'tea-party');
});

test('keywords, incomplete sentences, negations and changed facts do not select artwork', () => {
  for (const text of ['white rabbit', 'flamant rose', 'A white rabbit checks a golden pocket watch',
    'A black rabbit checks a golden pocket watch in a meadow.',
    'Not a white rabbit checks a golden pocket watch in a meadow.',
    'A white rabbit checks a golden pocket watch in a meadow with a dragon.']) {
    assert.equal(matchCue(text, scenes), null, text);
  }
});
