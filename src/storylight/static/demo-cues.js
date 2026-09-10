/* Prepared scene selection, independent of the free-form generation scheduler. */
(function (root) {
  function normalize(text) {
    return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  }
  function matchCue(text, scenes) {
    const sentence = normalize(text);
    return scenes.find((scene) => scene.cues.some((cue) => normalize(cue) === sentence)) || null;
  }
  function passageProgress(text, passage) {
    const heard = normalize(text).split(" ").filter(Boolean).slice(0, 256);
    const words = normalize(passage).split(" ").filter(Boolean);
    if (!heard.length || !words.length || words.length > 256) return {words: 0, complete: false};
    // Align a cumulative recording to a prefix of the selected passage.
    let row = words.map((_, i) => i + 1);
    row.unshift(0);
    for (let i = 0; i < heard.length; i++) {
      const next = [i + 1];
      for (let j = 1; j <= words.length; j++) {
        next[j] = Math.min(row[j] + 1, next[j - 1] + 1,
          row[j - 1] + Number(heard[i] !== words[j - 1]));
      }
      row = next;
    }
    let matched = 0;
    for (let j = 1; j <= words.length; j++) {
      if (row[j] <= Math.floor(j * 0.12) && j >= heard.length * 0.85) matched = j;
    }
    const ending = heard.slice(-3).join(" ") === words.slice(-3).join(" ");
    return {words: matched, complete: matched === words.length && ending};
  }
  root.StorylightCues = {matchCue, passageProgress};
  if (typeof module !== "undefined") module.exports = {matchCue, passageProgress};
})(globalThis);
