import html2canvas from 'html2canvas';
import JSZip from 'jszip';

/**
 * State
 */
let project = { pages: [], base_url: '' };
let currentPage = 0;

const canvasEl = document.getElementById('canvas');
const imgs = [0,1,2,3].map(i => document.getElementById(`img-${i}`));
const pageIndicator = document.getElementById('page-indicator');
const thumbsEl = document.getElementById('thumbs');

function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

function renderPage(idx) {
  if (!project.pages.length) return;
  currentPage = clamp(idx, 0, project.pages.length - 1);
  const page = project.pages[currentPage];
  pageIndicator.textContent = `Page ${currentPage+1}/${project.pages.length}`;
  for (let i=0;i<4;i++) {
    const frame = page.frames[i];
    if (frame && frame.path) {
      const src = resolvePath(frame.path);
      imgs[i].src = src;
    } else {
      imgs[i].src = '';
    }
  }
  // remove previous bubbles
  Array.from(canvasEl.querySelectorAll('.bubble')).forEach(n => n.remove());
  // add bubbles
  for (const b of page.bubbles || []) {
    mountBubble(b);
  }
  renderThumbs();
}

function resolvePath(p) {
  if (p.startsWith('http') || p.startsWith('/')) return p;
  const base = project.base_url || '';
  if (base) {
    return `${base.replace(/\/$/, '')}/${p}`;
  }
  return `/${p}`;
}

function createBubbleEl(bubble) {
  const el = document.createElement('div');
  el.className = 'bubble';
  el.textContent = bubble.text || '';
  el.style.left = `${bubble.x || 10}px`;
  el.style.top = `${bubble.y || 10}px`;
  el.style.width = `${bubble.width || 180}px`;
  el.style.height = `${bubble.height || 28}px`;
  enableDrag(el, bubble);
  el.addEventListener('dblclick', () => {
    el.setAttribute('contenteditable', 'true');
    el.focus();
  });
  el.addEventListener('blur', () => {
    el.removeAttribute('contenteditable');
    bubble.text = el.textContent || '';
    persist();
  });
  return el;
}

function mountBubble(bubble) {
  const el = createBubbleEl(bubble);
  canvasEl.appendChild(el);
}

function addBubble() {
  const page = project.pages[currentPage];
  const bubble = { text: '', frame_index: 0, x: 10, y: 10, width: 180, height: 28 };
  page.bubbles = page.bubbles || [];
  page.bubbles.push(bubble);
  mountBubble(bubble);
  persist();
}

function enableDrag(el, bubble) {
  let dragging = false;
  let startX = 0, startY = 0;
  let origX = 0, origY = 0;

  el.addEventListener('mousedown', (e) => {
    if (el.getAttribute('contenteditable') === 'true') return;
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    origX = parseFloat(el.style.left);
    origY = parseFloat(el.style.top);
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    const nx = clamp(origX + dx, 0, canvasEl.clientWidth - el.clientWidth);
    const ny = clamp(origY + dy, 0, canvasEl.clientHeight - el.clientHeight);
    el.style.left = `${nx}px`;
    el.style.top = `${ny}px`;
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    bubble.x = parseFloat(el.style.left);
    bubble.y = parseFloat(el.style.top);
    persist();
  });
}

async function exportCurrentPagePNG() {
  const canvas = await html2canvas(canvasEl, { backgroundColor: null, scale: 1, width: 400, height: 540 });
  const data = canvas.toDataURL('image/png');
  downloadDataUrl(data, `page_${String(currentPage+1).padStart(3,'0')}.png`);
}

async function exportAllPagesZip() {
  const zip = new JSZip();
  for (let i = 0; i < project.pages.length; i++) {
    renderPage(i);
    // wait a tick for images to settle
    // eslint-disable-next-line no-await-in-loop
    await new Promise(r => setTimeout(r, 150));
    // eslint-disable-next-line no-await-in-loop
    const canvas = await html2canvas(canvasEl, { backgroundColor: null, scale: 1, width: 400, height: 540 });
    // eslint-disable-next-line no-await-in-loop
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
    zip.file(`page_${String(i+1).padStart(3,'0')}.png`, blob);
  }
  const content = await zip.generateAsync({ type: 'blob' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(content);
  a.download = 'comic_pages.zip';
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadDataUrl(dataUrl, filename) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

function renderThumbs() {
  thumbsEl.innerHTML = '';
  project.pages.forEach((p, i) => {
    const div = document.createElement('div');
    div.className = 'page-item';
    div.innerHTML = `<span>Page ${i+1}</span><span class="small">${(p.frames||[]).length} frames</span>`;
    div.addEventListener('click', () => renderPage(i));
    thumbsEl.appendChild(div);
  });
}

// persistence: keep edits locally
function persist() {
  try {
    localStorage.setItem('project.json', JSON.stringify(project));
  } catch {}
}

function maybeLoadPersisted() {
  try {
    const raw = localStorage.getItem('project.json');
    if (raw) project = JSON.parse(raw);
  } catch {}
}

// controls
document.getElementById('prev').addEventListener('click', () => renderPage(currentPage - 1));
document.getElementById('next').addEventListener('click', () => renderPage(currentPage + 1));
document.getElementById('add-bubble').addEventListener('click', addBubble);
document.getElementById('export-page').addEventListener('click', exportCurrentPagePNG);
document.getElementById('export-all').addEventListener('click', exportAllPagesZip);

document.getElementById('load-project').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      project = JSON.parse(reader.result);
      persist();
      renderPage(0);
    } catch (err) {
      alert('Invalid project.json');
    }
  };
  reader.readAsText(file);
});

document.getElementById('load-server').addEventListener('click', async () => {
  try {
    const res = await fetch('http://localhost:8000/api/project');
    if (!res.ok) throw new Error('server not ready');
    const data = await res.json();
    project = data;
    if (!project.base_url) project.base_url = 'http://localhost:8000/outputs';
    persist();
    renderPage(0);
  } catch (e) {
    alert('Could not load from server. Start backend server first.');
  }
});

maybeLoadPersisted();
if (project.pages && project.pages.length) {
  renderPage(0);
}

