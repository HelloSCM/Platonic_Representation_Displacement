'use strict';

// Keep all reported aggregates explicit and separate from the original plot assets.
const agreement = {
  concept: { pointwise: [0.41, 0.36, 0.25], crossModel: [0.73, 0.91, 0.49] },
  sample: { pointwise: [0.31, 0.36, 0.19], crossModel: [0.45, 0.73, 0.31] }
};

function setTab(button) {
  const tablist = button.closest('[role="tablist"]');
  tablist.querySelectorAll('[role="tab"]').forEach(tab => {
    const selected = tab === button;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.getElementById(button.getAttribute('aria-controls')).setAttribute('aria-labelledby', button.id);
}

document.querySelectorAll('[role="tablist"]').forEach(tablist => {
  tablist.addEventListener('keydown', event => {
    const tabs = [...tablist.querySelectorAll('[role="tab"]')];
    const current = tabs.indexOf(document.activeElement);
    if (current < 0) return;
    let next;
    if (event.key === 'ArrowRight') next = (current + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      tabs[next].focus();
      tabs[next].click();
    }
  });
});

document.querySelectorAll('[data-level]').forEach(button => {
  button.addEventListener('click', () => {
    setTab(button);
    const data = agreement[button.dataset.level];
    document.querySelectorAll('.comparison').forEach((item, index) => {
      item.querySelector('[data-score]').textContent = data.crossModel[index].toFixed(2);
      item.querySelector('.pointwise-value').textContent = data.pointwise[index].toFixed(2);
      item.querySelector('.cross-model-value').textContent = data.crossModel[index].toFixed(2);
      item.querySelector('.pointwise-bar').style.setProperty('--bar', `${data.pointwise[index] * 100}%`);
      item.querySelector('.cross-model-bar').style.setProperty('--bar', `${data.crossModel[index] * 100}%`);
    });
  });
});

// One source of truth for both table views: the values exported from LaTeX Table 1.
let resultData = null;
let activeVariant = 'large';
const fullResults = document.getElementById('full-results');
function renderResults() {
  if (!resultData) return;
  const selected = fullResults.checked ? resultData.columns.map((_, index) => index) : [10, 11, 12, 13, 14, 15];
  const rows = resultData.variants[activeVariant];
  const best = selected.map(index => Math.max(...rows.map(row => row.values[index] ?? -Infinity)));
  const table = document.getElementById('results-table');
  table.replaceChildren();
  const caption = table.createCaption();
  caption.textContent = `${fullResults.checked ? 'All benchmarks' : 'Selected results'} · ViT-${activeVariant === 'large' ? 'Large' : 'Base'} · scores in percent`;
  const header = table.createTHead().insertRow();
  ['Method', ...selected.map(index => resultData.columns[index])].forEach(label => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = label;
    header.append(th);
  });
  const body = table.createTBody();
  rows.forEach(row => {
    const tr = body.insertRow();
    if (row.method === 'ShadowCLIP') tr.className = 'ours';
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = row.method;
    tr.append(name);
    selected.forEach((index, position) => {
      const cell = tr.insertCell();
      const value = row.values[index];
      cell.textContent = value === null ? '—' : value.toFixed(1);
      if (value !== null && value === best[position]) cell.className = 'best';
    });
  });
}

const variantButtons = [...document.querySelectorAll('[data-variant]')];
variantButtons.forEach(button => button.disabled = true);
fullResults.disabled = true;
fetch('assets/results.json')
  .then(response => {
    if (!response.ok) throw new Error(`Results unavailable: ${response.status}`);
    return response.json();
  })
  .then(data => {
    resultData = data;
    variantButtons.forEach(button => button.disabled = false);
    fullResults.disabled = false;
  })
  .catch(() => {
    const note = document.createElement('p');
    note.className = 'data-note';
    note.textContent = 'The interactive results could not be loaded. The ViT-Large summary is shown above; the complete results are available in Table 1 of the paper.';
    document.getElementById('results-panel').append(note);
  });

variantButtons.forEach(button => {
  button.addEventListener('click', () => {
    setTab(button);
    activeVariant = button.dataset.variant;
    renderResults();
  });
});
fullResults.addEventListener('change', renderResults);

// Native dialog provides focus trapping, Escape dismissal, and focus restoration.
const dialog = document.getElementById('figure-dialog');
const dialogImage = document.getElementById('dialog-image');
document.querySelectorAll('[data-figure]').forEach(button => {
  button.addEventListener('click', () => {
    const image = button.querySelector('img');
    const figure = button.closest('figure');
    document.getElementById('dialog-title').textContent = button.getAttribute('aria-label').replace(/^Enlarge (the )?/, '').replace(/^./, first => first.toUpperCase());
    dialogImage.src = image.src;
    dialogImage.alt = image.alt;
    document.getElementById('dialog-pdf').href = `assets/pdf/${button.dataset.figure}.pdf`;
    document.getElementById('dialog-caption').textContent = figure.querySelector('figcaption')?.textContent.replace(/(Original )?PDF\s*↗/g, '').trim() || image.alt;
    dialog.showModal();
    document.body.classList.add('dialog-open');
  });
});
document.getElementById('dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));

const copyButton = document.getElementById('copy-citation');
let copyReset;
copyButton.addEventListener('click', async () => {
  const text = document.getElementById('bibtex').textContent;
  const status = document.getElementById('copy-status');
  let copied = false;
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); copied = true; } catch (_) { /* Fall back below. */ }
  }
  if (!copied) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.cssText = 'position:fixed;top:-9999px';
    document.body.append(textarea);
    textarea.select();
    try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
    textarea.remove();
  }
  if (copied) {
    copyButton.querySelector('span').textContent = 'Copied!';
    status.textContent = 'Citation copied to clipboard.';
  } else {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.querySelector('span').textContent = 'Text selected';
    status.textContent = 'Citation selected. Press Control+C or Command+C to copy.';
  }
  clearTimeout(copyReset);
  copyReset = setTimeout(() => copyButton.querySelector('span').textContent = 'Copy citation', 2500);
});

const navToggle = document.querySelector('.nav-toggle');
const nav = document.getElementById('main-nav');
function closeNavigation() {
  nav.classList.remove('is-open');
  navToggle.setAttribute('aria-expanded', 'false');
  navToggle.setAttribute('aria-label', 'Open navigation');
}
navToggle.addEventListener('click', () => {
  const open = nav.classList.toggle('is-open');
  navToggle.setAttribute('aria-expanded', String(open));
  navToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
});
nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeNavigation));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && nav.classList.contains('is-open')) {
    closeNavigation();
    navToggle.focus();
  }
});
document.addEventListener('click', event => {
  if (!event.target.closest('.site-header')) closeNavigation();
});
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      nav.querySelectorAll('a').forEach(link => {
        const active = link.hash === `#${entry.target.id}`;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
  nav.querySelectorAll('a').forEach(link => {
    const section = document.querySelector(link.hash);
    if (section) observer.observe(section);
  });
}
