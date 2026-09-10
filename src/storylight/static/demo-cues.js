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
  root.StorylightCues = {matchCue};
  if (typeof module !== "undefined") module.exports = {matchCue};
})(globalThis);
