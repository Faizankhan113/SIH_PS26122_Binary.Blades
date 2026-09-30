const $ = (s) => document.querySelector(s);

const fileInput = $('#fileInput');
const dropzone = $('#dropzone');
const fileName = $('#fileName');
if (fileInput) {
  fileInput.addEventListener('change', () => {
    const selectedFile = fileInput.files?.[0];
    if (fileName) {
      fileName.textContent = selectedFile?.name || '';
      fileName.classList.toggle('hidden', !selectedFile);
    }
  });
}

const rawText = $('#raw_text');
const rawTextCount = $('#rawTextCount');
if (rawText && rawTextCount) {
  const updateRawTextCount = () => { rawTextCount.textContent = String(rawText.value.length); };
  rawText.addEventListener('input', updateRawTextCount);
  updateRawTextCount();
}
if (dropzone) {
  ['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
  ['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer?.files;
    if (!fileInput || !files || !files.length) return;
    const transfer = new DataTransfer();
    Array.from(files).forEach((file) => transfer.items.add(file));
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

const ingestForm = $('#ingestForm');
if (ingestForm) {
  ingestForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = ingestForm.querySelector('button[type="submit"]');
    btn.disabled = true; btn.textContent = 'Reading input…';
    $('#ingestError').classList.add('hidden');
    try {
      const res = await fetch('/api/ingest', { method: 'POST', body: new FormData(ingestForm) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Ingestion failed');
      location.href = data.redirect;
    } catch (err) {
      $('#ingestError').textContent = err.message;
      $('#ingestError').classList.remove('hidden');
      btn.disabled = false; btn.innerHTML = 'Continue to Extraction <span>→</span>';
    }
  });
}

const toMatch = $('#toMatch');
if (toMatch) {
  toMatch.addEventListener('click', async () => {
    toMatch.disabled = true; toMatch.textContent = 'Running matcher…';
    try {
      const res = await fetch('/api/continue-to-match', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Matching failed');
      location.href = data.redirect;
    } catch (err) {
      $('#matchError').textContent = err.message;
      $('#matchError').classList.remove('hidden');
      toMatch.disabled = false; toMatch.innerHTML = 'Continue to Matching <span>→</span>';
    }
  });
}

const resetBtn = $('#resetBtn');
if (resetBtn) {
  resetBtn.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    location.href = '/ingest';
  });
}

// Supervisor review queue: account approve/reject, match approve/reject.
// Each button posts to its endpoint, then removes its own row on success
// rather than reloading the whole page.
// Approving an item that came from a failed extraction asks twice. The
// first click shows a warning; the server also refuses the approval unless the
// request says confirm_degraded, so this cannot be skipped by the page alone.
const DEGRADED_WARNING =
  'WARNING: extraction failed for this item, so its fields are keyword guesses ' +
  '(status, dates and delay reason may be wrong or missing).\n\n' +
  'Approving it will write those guesses to the plan. Have you checked the field statement yourself?';

// Remove a decided card and keep the count and the empty message right.
function removeReviewRow(rowId) {
  const row = document.getElementById(rowId);
  if (row) row.remove();
  if (!rowId.startsWith('review-row-')) return;
  const remaining = document.querySelectorAll('[id^="review-row-"]').length;
  const count = document.getElementById('reviewCount');
  if (count) count.textContent = String(remaining);
  const empty = document.getElementById('reviewEmpty');
  if (empty) empty.style.display = remaining ? 'none' : '';
}

function wireReviewButtons(selector, urlBuilder, rowIdBuilder, dataAttr) {
  document.querySelectorAll(selector).forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset[dataAttr];
      let confirmDegraded = false;
      if (btn.dataset.degraded === '1') {
        if (!window.confirm(DEGRADED_WARNING)) return;
        confirmDegraded = true;
      }
      btn.disabled = true;
      const original = btn.textContent;
      btn.textContent = '…';
      try {
        const options = { method: 'POST' };
        if (confirmDegraded) {
          options.headers = { 'Content-Type': 'application/json' };
          options.body = JSON.stringify({ confirm_degraded: true });
        }
        let res = await fetch(urlBuilder(id), options);
        let data = await res.json();
        // Safety net: the server says this needs the extra confirmation (e.g. the badge was missing).
        if (res.status === 409 && data.needs_confirmation && !confirmDegraded) {
          if (!window.confirm(`${data.error}\n\n${DEGRADED_WARNING}`)) {
            btn.disabled = false;
            btn.textContent = original;
            return;
          }
          res = await fetch(urlBuilder(id), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confirm_degraded: true }),
          });
          data = await res.json();
        }
        if (res.status === 409 && data.already_decided) {
          // Someone (or an earlier click) already decided this item. Say so,
          // then drop the stale card; the plan was not touched a second time.
          alert(data.error);
          removeReviewRow(rowIdBuilder(id));
          return;
        }
        if (!res.ok || !data.ok) throw new Error(data.error || 'Action failed');
        removeReviewRow(rowIdBuilder(id));
      } catch (err) {
        alert(err.message);
        btn.disabled = false;
        btn.textContent = original;
      }
    });
  });
}

wireReviewButtons('.account-approve-btn', (id) => `/api/accounts/${id}/approve`, (id) => `acct-row-${id}`, 'userId');
wireReviewButtons('.account-reject-btn', (id) => `/api/accounts/${id}/reject`, (id) => `acct-row-${id}`, 'userId');
wireReviewButtons('.review-approve-btn', (id) => `/api/reviews/${id}/approve`, (id) => `review-row-${id}`, 'updateId');
wireReviewButtons('.review-reject-btn', (id) => `/api/reviews/${id}/reject`, (id) => `review-row-${id}`, 'updateId');

// Planner list. Dismiss / Mark as new activity post the optional note, then
// reload so the tabs, the counts and the navigation number are all right. If the
// item was already decided, the server keeps the first decision and says so.
function wirePlannerButtons(selector, path) {
  document.querySelectorAll(selector).forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.updateId;
      const noteBox = document.getElementById(`note-${id}`);
      const buttons = document.querySelectorAll(`#planner-row-${id} button`);
      buttons.forEach((b) => { b.disabled = true; });
      try {
        const res = await fetch(`/api/planner/${id}/${path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ note: noteBox ? noteBox.value : '' }),
        });
        const data = await res.json();
        if (res.status === 409 && data.already_decided) {
          alert(data.error);
          location.reload();
          return;
        }
        if (!res.ok || !data.ok) throw new Error(data.error || 'Action failed');
        location.reload();
      } catch (err) {
        alert(err.message);
        buttons.forEach((b) => { b.disabled = false; });
      }
    });
  });
}
wirePlannerButtons('.planner-dismiss-btn', 'dismiss');
wirePlannerButtons('.planner-convert-btn', 'convert');

// Loading screen. Starts extraction (safe to call twice), then polls the
// status until every statement is done or failed, and reloads to show results.
const extractProgress = $('#extractProgress');
if (extractProgress) {
  const stateLabel = { pending: 'Waiting', running: 'Reading…', done: 'Done', failed: 'Failed' };
  const total = Number(extractProgress.dataset.total || 0);
  const paint = (data) => {
    $('#extractCount').textContent = `${data.finished} of ${data.total} statements finished`;
    $('#extractBar').style.width = `${data.total ? Math.round((100 * data.finished) / data.total) : 0}%`;
    (data.items || []).forEach((it) => {
      const li = extractProgress.querySelector(`li[data-idx="${it.idx}"]`);
      if (!li) return;
      li.className = `extract-item state-${it.state}`;
      li.querySelector('[data-role="state"]').textContent = stateLabel[it.state] || it.state;
    });
  };
  const fail = (message) => {
    const box = $('#extractError');
    box.textContent = message;
    box.classList.remove('hidden');
  };
  const poll = async () => {
    try {
      const res = await fetch('/api/extract/status');
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || 'Could not read extraction progress');
      paint(data);
      if (data.complete) { location.reload(); return; }
      setTimeout(poll, 1000);
    } catch (err) {
      fail(`${err.message}. Retrying…`);
      setTimeout(poll, 3000);
    }
  };
  (async () => {
    try {
      const res = await fetch('/api/extract/start', { method: 'POST' });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || 'Could not start extraction');
      paint(data);
      if (data.complete) { location.reload(); return; }
    } catch (err) {
      fail(err.message);
      return;
    }
    poll();
  })();
}

// Extraction results page: run the statements that failed once more.
const retryExtract = $('#retryExtract');
if (retryExtract) {
  retryExtract.addEventListener('click', async () => {
    retryExtract.disabled = true;
    try {
      await fetch('/api/extract/start', { method: 'POST' });
    } finally {
      location.reload();
    }
  });
}

// Retry the matcher for one event that ended in ERROR.
document.querySelectorAll('.retry-match').forEach((btn) => {
  btn.addEventListener('click', async () => {
    btn.disabled = true; btn.textContent = 'Retrying…';
    try {
      const res = await fetch('/api/match/retry/' + encodeURIComponent(btn.dataset.eventId), { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Retry failed');
      location.reload();
    } catch (err) {
      btn.disabled = false; btn.textContent = 'Retry';
      alert(err.message);
    }
  });
});
