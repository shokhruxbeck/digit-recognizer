/* Handwritten Digit Recognition — PyTorch CNN running in the browser.
 *
 * Flow: pen strokes -> normalised 8x8 grayscale grid -> ONNX Runtime Web -> softmax.
 * All predictions come from the trained network in digits_cnn.onnx.
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------
  // Config
  // ------------------------------------------------------------------
  var CONFIG = {
    modelUrl: 'digits_cnn.onnx',
    ortWasmBase: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/',
    penWidth: 22,        // on-screen pen width, in canvas pixels (canvas is 560x560)
    hiRes: 80,           // working resolution before the final 8x8 reduction (80 = 8 cells x 10 px)
    supersample: 8,      // anti-aliasing factor when re-drawing the strokes
    fill: 0.85,          // fraction of the 80px frame the digit's longest side occupies
    strokeRel: 0.18,     // pen thickness relative to the digit's longest side
    minSpan: 24          // ignore scribbles smaller than this (canvas pixels)
  };

  // ------------------------------------------------------------------
  // Pure pipeline: strokes -> 64 floats in the model's training scale
  // ------------------------------------------------------------------
  // strokes: Array<Array<{x:number,y:number}>> in canvas pixel coordinates.
  // makeCanvas(w, h): returns an object with getContext('2d') (browser canvas or node-canvas).
  function preprocess(strokes, makeCanvas) {
    var pts = [];
    strokes.forEach(function (s) { s.forEach(function (p) { pts.push(p); }); });
    if (!pts.length) return { error: 'empty' };

    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    pts.forEach(function (p) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    });
    var span = Math.max(maxX - minX, maxY - minY);
    if (span < CONFIG.minSpan) return { error: 'small' };

    var HI = CONFIG.hiRes, SS = CONFIG.supersample, N = HI * SS;
    var box = HI * CONFIG.fill;               // target size of the digit in the 80px frame
    var scale = box / span;
    var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    var map = function (p) {
      return [(HI / 2 + (p.x - cx) * scale) * SS, (HI / 2 + (p.y - cy) * scale) * SS];
    };

    // 1. Re-draw the strokes: white on black, centred, fixed thickness relative to the digit.
    var c = makeCanvas(N, N), ctx = c.getContext('2d');
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, N, N);
    ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    var lw = CONFIG.strokeRel * box * SS;
    ctx.lineWidth = lw;
    strokes.forEach(function (s) {
      if (!s.length) return;
      var a = map(s[0]);
      if (s.length === 1) {
        ctx.beginPath(); ctx.arc(a[0], a[1], lw / 2, 0, Math.PI * 2); ctx.fill(); return;
      }
      ctx.beginPath(); ctx.moveTo(a[0], a[1]);
      for (var i = 1; i < s.length; i++) { var b = map(s[i]); ctx.lineTo(b[0], b[1]); }
      ctx.stroke();
    });

    // 2. Grayscale + box-average (area) down to 8x8. Values 0..255.
    var data = ctx.getImageData(0, 0, N, N).data;
    var cell = N / 8, sums = new Float64Array(64);
    for (var y = 0; y < N; y++) {
      var row = Math.floor(y / cell) * 8;
      for (var x = 0; x < N; x++) {
        var k = (y * N + x) * 4;
        sums[row + Math.floor(x / cell)] += 0.299 * data[k] + 0.587 * data[k + 1] + 0.114 * data[k + 2];
      }
    }
    var avg = new Float64Array(64), peak = 0;
    for (var j = 0; j < 64; j++) { avg[j] = sums[j] / (cell * cell); if (avg[j] > peak) peak = avg[j]; }
    if (peak <= 0) return { error: 'empty' };

    // 3. Map to the dataset's 0..16 scale (contrast-stretched so the brightest cell is 16),
    //    then divide by 16 exactly like the training code: X = digits.images / 16.
    var grid16 = new Float32Array(64), input = new Float32Array(64);
    for (var m = 0; m < 64; m++) {
      grid16[m] = (avg[m] / peak) * 16;
      input[m] = grid16[m] / 16;
    }
    return { input: input, grid16: grid16 };
  }

  function softmax(logits) {
    var max = Math.max.apply(null, logits), exps = [], sum = 0, i;
    for (i = 0; i < logits.length; i++) { exps[i] = Math.exp(logits[i] - max); sum += exps[i]; }
    return exps.map(function (e) { return e / sum; });
  }

  function b64ToBytes(b64) {
    var bin = atob(b64), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  var core = { CONFIG: CONFIG, preprocess: preprocess, softmax: softmax };
  if (typeof module !== 'undefined' && module.exports) { module.exports = core; return; }
  root.DigitCore = core;

  // ------------------------------------------------------------------
  // Browser UI
  // ------------------------------------------------------------------
  var $ = function (id) { return document.getElementById(id); };
  var pad = $('pad'), ctx = pad.getContext('2d');
  var hint = $('hint'), message = $('message'), statusEl = $('status'), statusText = $('status-text');
  var btnPredict = $('btn-predict'), btnClear = $('btn-clear');
  var digitEl = $('digit'), confVal = $('conf-val'), confBar = $('conf-bar');
  var gridEl = $('grid'), barsEl = $('bars'), metaEl = $('meta');

  var strokes = [], current = null, session = null;

  // Build static DOM: 64 input cells and 10 probability rows.
  var cells = [];
  for (var i = 0; i < 64; i++) { var el = document.createElement('i'); gridEl.appendChild(el); cells.push(el); }
  var rows = [];
  for (var d = 0; d < 10; d++) {
    var r = document.createElement('div');
    r.className = 'bar';
    r.innerHTML = '<span>' + d + '</span><div class="track"><div class="fill"></div></div><span class="pct">0.0%</span>';
    barsEl.appendChild(r);
    rows.push({ row: r, fill: r.querySelector('.fill'), pct: r.querySelector('.pct') });
  }

  function setStatus(state, text) {
    statusEl.dataset.state = state; statusText.textContent = text;
  }
  function say(text, isErr) {
    message.textContent = text || ''; message.classList.toggle('err', !!isErr);
  }

  // ---- drawing ----
  function resetPad() {
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, pad.width, pad.height);
    ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff';
    ctx.lineWidth = CONFIG.penWidth; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  }
  function pos(e) {
    var rect = pad.getBoundingClientRect();
    return { x: (e.clientX - rect.left) * pad.width / rect.width,
             y: (e.clientY - rect.top) * pad.height / rect.height };
  }
  pad.addEventListener('pointerdown', function (e) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.preventDefault();
    pad.setPointerCapture(e.pointerId);
    var p = pos(e);
    current = [p]; strokes.push(current);
    ctx.beginPath(); ctx.arc(p.x, p.y, CONFIG.penWidth / 2, 0, Math.PI * 2); ctx.fill();
    hint.classList.add('hidden'); say('');
  });
  pad.addEventListener('pointermove', function (e) {
    if (!current) return;
    var events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    if (!events.length) events = [e];
    events.forEach(function (ev) {
      var p = pos(ev), q = current[current.length - 1];
      current.push(p);
      ctx.beginPath(); ctx.moveTo(q.x, q.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    });
  });
  function endStroke() { current = null; }
  pad.addEventListener('pointerup', endStroke);
  pad.addEventListener('pointercancel', endStroke);
  pad.addEventListener('lostpointercapture', endStroke);

  function clearAll() {
    strokes = []; current = null; resetPad();
    hint.classList.remove('hidden'); say('');
    digitEl.textContent = '–'; digitEl.dataset.empty = 'true';
    confVal.textContent = '—'; confBar.style.width = '0';
    cells.forEach(function (c) { c.style.background = '#000'; c.title = ''; });
    rows.forEach(function (r) { r.fill.style.width = '0'; r.pct.textContent = '0.0%'; r.row.classList.remove('top'); });
    metaEl.textContent = '';
  }
  btnClear.addEventListener('click', clearAll);

  // ---- model ----
  async function loadModel() {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime Web failed to load (check your internet connection).');
    ort.env.wasm.wasmPaths = CONFIG.ortWasmBase;
    ort.env.wasm.numThreads = 1;   // no cross-origin isolation needed
    var opts = { executionProviders: ['wasm'] };
    try {
      var res = await fetch(CONFIG.modelUrl);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return await ort.InferenceSession.create(new Uint8Array(await res.arrayBuffer()), opts);
    } catch (fetchErr) {
      // Opening index.html straight from disk (file://) blocks fetch(); use the embedded copy
      // of the very same digits_cnn.onnx bytes instead.
      if (root.DIGITS_CNN_B64) return await ort.InferenceSession.create(b64ToBytes(root.DIGITS_CNN_B64), opts);
      throw fetchErr;
    }
  }

  async function predict() {
    if (!session) return;
    if (!strokes.length) { say('Draw a digit first.', true); return; }
    var out = preprocess(strokes, function (w, h) {
      var c = document.createElement('canvas'); c.width = w; c.height = h; return c;
    });
    if (out.error === 'small') { say('That mark is too small. Draw the digit larger.', true); return; }
    if (out.error) { say('Draw a digit first.', true); return; }
    say('');

    var t0 = performance.now();
    var result = await session.run({ input: new ort.Tensor('float32', out.input, [1, 1, 8, 8]) });
    var ms = performance.now() - t0;
    var probs = softmax(Array.prototype.slice.call(result.logits.data));

    var best = 0;
    probs.forEach(function (p, k) { if (p > probs[best]) best = k; });

    digitEl.textContent = best; digitEl.dataset.empty = 'false';
    digitEl.classList.remove('pop'); void digitEl.offsetWidth; digitEl.classList.add('pop');
    confVal.textContent = (probs[best] * 100).toFixed(1) + '%';
    confBar.style.width = (probs[best] * 100) + '%';
    rows.forEach(function (r, k) {
      r.fill.style.width = (probs[k] * 100) + '%';
      r.pct.textContent = (probs[k] * 100).toFixed(1) + '%';
      r.row.classList.toggle('top', k === best);
    });
    out.grid16.forEach(function (v, k) {
      var g = Math.round((v / 16) * 255);
      cells[k].style.background = 'rgb(' + g + ',' + g + ',' + g + ')';
      cells[k].title = v.toFixed(1) + ' / 16';
    });
    metaEl.textContent = 'inference ' + ms.toFixed(1) + ' ms';
    if (probs[best] < 0.6) say('Low confidence. Try a larger, cleaner digit.');
  }
  btnPredict.addEventListener('click', predict);

  resetPad();
  loadModel().then(function (s) {
    session = s; btnPredict.disabled = false;
    setStatus('ready', 'Model ready · ONNX Runtime Web');
  }).catch(function (err) {
    console.error(err);
    setStatus('error', 'Model failed to load');
    say(String(err.message || err) + ' Serve this folder with a local web server (see README).', true);
  });
})(typeof window !== 'undefined' ? window : globalThis);
