/* Prepared scene selection, independent of the free-form generation scheduler. */
(function (root) {
  function normalize(text) {
    return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  }
  function matchCue(text, scenes) {
    const words = ` ${normalize(text)} `;
    let match = null;
    let position = -1;
    for (const scene of scenes) {
      for (const cue of scene.cues) {
        const index = words.lastIndexOf(` ${normalize(cue)} `);
        if (index > position) { match = scene; position = index; }
      }
    }
    return match;
  }
  root.StorylightCues = {matchCue};
  if (typeof module !== "undefined") module.exports = {matchCue};
})(globalThis);
