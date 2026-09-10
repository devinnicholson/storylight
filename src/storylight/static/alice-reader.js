/* A local reading selection. Audio is handled by the workbench's existing recorder. */
(function (root) {
  async function mount(container, {onSelect, isRecording}) {
    const response = await fetch('/workbench-assets/books/alice.json');
    if (!response.ok) throw new Error('The Alice reading pack could not be loaded.');
    const book = await response.json();
    let selected = book.passages[0];
    container.removeAttribute('role');
    container.removeAttribute('aria-live');
    container.classList.add('alice-reader');
    container.replaceChildren();
    const node = (tag, text, parent = container) => {
      const element = document.createElement(tag);
      if (text) element.textContent = text;
      parent.append(element);
      return element;
    };
    node('p', `${book.author} · French translation by Henri Bué`);
    const label = node('label', 'Choose a passage ');
    const select = node('select', '', label);
    for (const passage of book.passages) {
      const option = node('option', `${passage.title} · ${passage.language.toUpperCase()}`, select);
      option.value = passage.scene_id;
    }
    const chapter = node('h3');
    const excerpt = node('blockquote');
    const progress = node('progress');
    progress.setAttribute('aria-label', 'Recognized passage');
    const status = node('p');
    status.setAttribute('role', 'status');
    const vocabulary = node('details');
    node('summary', 'Words in this passage', vocabulary);
    const definitions = node('dl', '', vocabulary);
    const attribution = node('p');
    const source = node('a', 'Project Gutenberg edition', attribution);
    source.target = '_blank'; source.rel = 'noopener';
    node('span', ' · ', attribution);
    const download = node('a', 'Download the complete book', attribution);
    download.setAttribute('download', '');
    const notes = node('details');
    node('summary', 'About the text and artwork', notes);
    node('p', 'Public domain in the USA. Check local copyright status elsewhere. The downloaded editions include the Project Gutenberg license.', notes);
    node('p', book.text_note, notes);
    node('p', book.art_note, notes);
    const render = () => {
      chapter.textContent = selected.chapter;
      excerpt.textContent = selected.text;
      excerpt.lang = selected.language;
      progress.value = 0;
      progress.max = 100;
      status.textContent = 'Read from the beginning. The illustration appears at the end of the passage.';
      definitions.replaceChildren();
      for (const [word, meaning] of Object.entries(selected.vocabulary)) {
        node('dt', word, definitions);
        node('dd', meaning, definitions);
      }
      source.href = book.sources[selected.language].url;
      download.href = book.sources[selected.language].file;
    };
    select.addEventListener('change', () => {
      if (isRecording()) {
        select.value = selected.scene_id;
        status.textContent = 'Stop listening before choosing another passage.';
        return;
      }
      selected = book.passages.find((passage) => passage.scene_id === select.value);
      onSelect();
      render();
    });
    render();
    return {
      match(text) {
        const result = StorylightCues.passageProgress(text, selected.text);
        const total = selected.text.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
          .toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).length;
        progress.value = Math.round(100 * result.words / total);
        status.textContent = result.complete ? 'Passage recognized.'
          : `${progress.value}% of the passage recognized.`;
        return result.complete ? selected.scene_id : null;
      },
    };
  }
  root.StorylightBook = {mount};
})(globalThis);
