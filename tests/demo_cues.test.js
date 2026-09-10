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

const {passageProgress} = require('../src/storylight/static/demo-cues.js');
const book = require('../src/storylight/static/books/alice.json');
const fs = require('node:fs');
const crypto = require('node:crypto');
test('Alice excerpts are traceable to bundled editions and retain their licenses', () => {
  for (const [language, source] of Object.entries(book.sources)) {
    const bytes = fs.readFileSync(`src/storylight/static/books/alice-${language}.txt`);
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), source.sha256);
    const text = bytes.toString('utf8').replace(/_/g, '').replace(/\s+/g, ' ');
    assert.match(text, /PROJECT GUTENBERG LICENSE/i);
    for (const passage of book.passages.filter(p => p.language === language)) {
      assert.ok(text.includes(passage.text));
    }
  }
});
test('passage alignment accepts complete readings, small ASR errors, and no keyword jumps', () => {
  for (const passage of book.passages) {
    assert.equal(passageProgress(passage.text, passage.text).complete, true);
    const words = passage.text.split(' ');
    assert.equal(passageProgress(words.slice(0, -6).join(' '), passage.text).complete, false);
    assert.equal(passageProgress(words.slice(-8).join(' '), passage.text).complete, false);
    assert.equal(passageProgress('white rabbit cat tea flamants', passage.text).complete, false);
    assert.equal(passageProgress('unrelated '.repeat(300), passage.text).complete, false);
    const misheard = [...words]; misheard[5] = 'something';
    assert.equal(passageProgress(misheard.join(' '), passage.text).complete, true);
  }
});
