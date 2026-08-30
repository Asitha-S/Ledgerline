/* Ledgerline — review interface.
 *
 * Reads pre-exported static JSON from web/data/. No backend, nothing recomputed at
 * render time, no model called. Every figure was produced by the Python pipeline and
 * cross-checked by export.py against exposure.py before being written to disk.
 *
 * Nothing on this page is simulated. The clock is the real system clock; the ticker,
 * the status chip and the record count all read from the exported JSON. There are no
 * invented events and no fake progress.
 *
 * Money is stored in the JSON as integer CENTS and formatted here for display only.
 */

const { createElement, useState, useEffect, useMemo, useRef, useCallback,
        Fragment } = React;
const html = htm.bind(createElement);

const DATA = "data";
const CURRENCY = "USD";

const BATCHES = [
  { key: "eval",  label: "BenchRec eval" },
  { key: "synth", label: "Synthetic 50k" },
];

/* One colour per batch, so which one is loaded is legible without reading a word.
 * Neither is a class colour inside its own batch — eval's classes are cobalt, green
 * and violet, and the synthetic batch has no missing_counterparty at all — so the
 * chrome never borrows a hue that means something else on the same screen. */
const BATCH_HUE = { eval: "#BFF442", synth: "#2563EB" };
const hueForBatch = (k) => BATCH_HUE[k] || "#BFF442";

const CURTAIN_MS = 260;      // half the wipe; the swap happens between the two halves

/* ---------------------------------------------------------------- palette */

const PAL = {
  lime: "#BFF442", violet: "#8B6FF5", tangerine: "#FF6B3D",
  cobalt: "#2563EB", yellow: "#F5C518", green: "#3DBB5E",
  ink: "#131A17", cream: "#F5F2EA", cream2: "#FBF9F3", dark: "#0F1512",
};

/* Fixed per card position, not derived from anything the card measures. The spread of
 * the palette across the overview comes from here. Class colours stay where they carry
 * meaning — the treemap and the queue — and are not read into these. */
const CARD_HUE = [PAL.tangerine, PAL.yellow, PAL.cobalt, PAL.violet, PAL.lime];
// The curve belongs to section 04, whose rail chip and accent word are violet.
// Tangerine is section 03's colour and reads as the queue, which this is not.
const CHART_ACCENT = PAL.violet;

const CLASS_HUE = {
  missing_counterparty:          PAL.cobalt,
  incomplete_set:                PAL.green,
  duplicate_reference_suspected: PAL.violet,
  fee_band_match:                PAL.tangerine,
  ambiguous_allocation:          PAL.yellow,
};
const hueFor = (c) => CLASS_HUE[c] || PAL.yellow;

/* Darkened for use as TEXT. The class hues are picked to be legible as large filled
 * areas; at pill size, several of them (yellow especially) have nowhere near enough
 * contrast against cream to carry words. */
const CLASS_INK = {
  missing_counterparty:          "#1B49CE",
  incomplete_set:                "#156B2C",
  duplicate_reference_suspected: "#5333C9",
  fee_band_match:                "#B23F16",
  ambiguous_allocation:          "#7A5A00",
};
const inkFor = (c) => CLASS_INK[c] || "#7A5A00";

/** The class hue at a given alpha, for tints that sit under text. */
function tint(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

const MONEY = { correct: PAL.green, queue: PAL.yellow, wrong: PAL.tangerine };

/** Pick ink or white for text sitting on a filled accent. */
function readableOn(hex) {
  const n = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.42 ? PAL.ink : "#FFFFFF";
}

/* ---------------------------------------------------------------- formatting */

const nf = (min, max) =>
  new Intl.NumberFormat("en-US", { minimumFractionDigits: min, maximumFractionDigits: max });
const f2 = nf(2, 2);
const f0 = nf(0, 0);

function money(cents) {
  if (cents === null || cents === undefined) return "—";
  return f2.format(cents / 100);
}
function moneyShort(cents) {
  if (cents === null || cents === undefined) return "—";
  const d = Math.abs(cents) / 100;
  const sign = cents < 0 ? "-" : "";
  for (const [div, suf] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]]) {
    if (d >= div) return sign + f2.format(d / div) + suf;
  }
  return sign + f2.format(d);
}
const pct = (v, dp = 3) =>
  (v === null || v === undefined ? "—" : nf(dp, dp).format(v) + "%");
const int = (v) => (v === null || v === undefined ? "—" : f0.format(v));
const score = (v) => (v === null || v === undefined ? "—" : nf(4, 4).format(v));

/* ---------------------------------------------------------------- data */

/* `onProgress(url, received, total)` reports real bytes off the wire, read from the
 * response stream against its Content-Length. It is what the boot bar is drawn from —
 * there is no timed animation standing in for a download. Where streaming or the
 * header is unavailable the body is taken whole and reported as complete on arrival. */
function useJson(url, onProgress) {
  const [state, setState] = useState({ data: null, error: null });
  useEffect(() => {
    let live = true;
    setState({ data: null, error: null });

    const whole = (r) => r.json();
    const streamed = (r) => {
      const total = Number(r.headers.get("content-length"));
      if (!total || !r.body || !r.body.getReader) return whole(r);
      const reader = r.body.getReader();
      const chunks = [];
      let got = 0;
      const pump = () => reader.read().then(({ done, value }) => {
        if (done) {
          const buf = new Uint8Array(got);
          let at = 0;
          for (const c of chunks) { buf.set(c, at); at += c.length; }
          return JSON.parse(new TextDecoder().decode(buf));
        }
        chunks.push(value);
        got += value.length;
        if (live && onProgress) onProgress(url, got, total);
        return pump();
      });
      return pump();
    };

    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return onProgress ? streamed(r) : whole(r);
      })
      .then((d) => {
        if (!live) return;
        if (onProgress) onProgress(url, 1, 1, true);
        setState({ data: d, error: null });
      })
      .catch((e) => live && setState({ data: null, error: e.message }));
    return () => { live = false; };
  }, [url]);
  return state;
}

/* The queue arrives in pages.
 *
 * A single synthetic queue file is 2.1 MB, and the browser cannot draw a row until all
 * of it has landed. The export writes an index carrying the first page inline plus the
 * remaining pages beside it, so the first fetch is ~200 KB and the table is usable
 * immediately. The rest is fetched behind it and committed in one update rather than
 * page by page, which would re-run the treemap layout once per page.
 *
 * `complete` says whether every row is in hand. Anything that claims to show the whole
 * queue has to wait for it; anything that shows a prefix of it does not. */
function useQueue(batch) {
  const [state, setState] = useState({ data: null, error: null, complete: false });
  useEffect(() => {
    let live = true;
    setState({ data: null, error: null, complete: false });
    const base = `${DATA}/queue_${batch}`;
    const grab = (u) => fetch(u).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    });

    grab(`${base}.json`)
      .then((head) => {
        if (!live) return;
        const pages = head.pages || 1;
        setState({ data: head, error: null, complete: pages <= 1 });
        if (pages <= 1) return;
        return Promise.all(
          Array.from({ length: pages - 1 }, (_, i) => grab(`${base}_p${i + 1}.json`))
        ).then((rest) => {
          if (!live) return;
          const all = head.queue.slice();
          rest.sort((a, b) => a.page - b.page).forEach((p) => {
            for (const row of p.queue) all.push(row);
          });
          setState({ data: { ...head, queue: all }, error: null, complete: true });
        });
      })
      .catch((e) => live && setState({ data: null, error: e.message, complete: false }));
    return () => { live = false; };
  }, [batch]);
  return state;
}

/* ---------------------------------------------------------------- treemap
 * Squarified. Layout only — it sizes rectangles from figures already in the export
 * and decides nothing. Iterative so 3,000+ items cannot exhaust the stack. */
function treemap(items, X, Y, W, H) {
  const out = [];
  const total = items.reduce((a, i) => a + i.value, 0);
  if (!total || W <= 0 || H <= 0) return out;
  let nodes = items.map((i) => ({ item: i, area: (i.value / total) * W * H }));
  let x = X, y = Y, w = W, h = H;
  const worst = (sum, mn, mx, short) => {
    const s2 = sum * sum, w2 = short * short;
    return Math.max((w2 * mx) / s2, s2 / (w2 * mn));
  };
  while (nodes.length && w > 1e-9 && h > 1e-9) {
    const short = Math.min(w, h);
    let k = 0, sum = 0, mn = Infinity, mx = 0, best = Infinity;
    for (; k < nodes.length; k++) {
      const a = nodes[k].area;
      const nSum = sum + a, nMn = Math.min(mn, a), nMx = Math.max(mx, a);
      const r = worst(nSum, nMn, nMx, short);
      if (k > 0 && r > best) break;
      sum = nSum; mn = nMn; mx = nMx; best = r;
    }
    const row = nodes.slice(0, k), thick = sum / short;
    let off = 0;
    if (w >= h) {
      for (const n of row) { const len = n.area / thick;
        out.push({ x, y: y + off, w: thick, h: len, item: n.item }); off += len; }
      x += thick; w -= thick;
    } else {
      for (const n of row) { const len = n.area / thick;
        out.push({ x: x + off, y, w: len, h: thick, item: n.item }); off += len; }
      y += thick; h -= thick;
    }
    nodes = nodes.slice(k);
  }
  if (nodes.length) {
    const len = Math.max(w, 1e-6) / nodes.length;
    nodes.forEach((n, i) => out.push({ x: x + i * len, y, w: len,
                                       h: Math.max(h, 1e-6), item: n.item }));
  }
  return out;
}

/* ---------------------------------------------------------------- motion */

function prefersReducedMotion() {
  return typeof window === "undefined" || !window.matchMedia ||
         window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function useReveal() {
  const ref = useRef(null);
  const [seen, setSeen] = useState(() => prefersReducedMotion());
  useEffect(() => {
    if (seen || !ref.current || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { setSeen(true); io.disconnect(); }
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.04 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [seen]);
  return [ref, seen];
}

const easeOut = (t) => 1 - Math.pow(1 - t, 3);

function useCountUp(target, ms = 700) {
  const reduced = prefersReducedMotion();
  const [v, setV] = useState(() => (reduced ? target : 0));
  useEffect(() => {
    if (reduced || typeof requestAnimationFrame === "undefined") { setV(target); return; }
    let raf = 0, t0 = 0;
    const step = (t) => {
      if (!t0) t0 = t;
      const p = Math.min(1, (t - t0) / ms);
      setV(target * easeOut(p));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms, reduced]);
  return v;
}

const CountPct = ({ value, dp = 2 }) => html`<span>${pct(useCountUp(value), dp)}</span>`;
const CountInt = ({ value }) => html`<span>${int(Math.round(useCountUp(value)))}</span>`;
const CountMoneyShort = ({ cents }) =>
  html`<span>${moneyShort(Math.round(useCountUp(cents)))}</span>`;

/* ---------------------------------------------------------------- boot
 * Shown once, while the three exported files are actually in flight. Each row flips
 * when its own fetch resolves; the counter starts only once summary.json has landed
 * and counts to the record count that file reports. Nothing here is simulated, and it
 * never returns — a batch switch reuses the small inline loading line instead. */

function Boot({ batch, records, pct: p, out }) {
  const n = useCountUp(records || 0, 700);
  return html`
    <div class=${"boot" + (out ? " out" : "")}>
      <div class="boot-inner">
        <div class="boot-mark">Ledgerline</div>
        <div class="boot-lab">${"Loading the " + batch + " batch"}</div>
        <div class="boot-count num">
          ${records ? int(Math.round(n)) : "0"}<span class="boot-unit">records</span>
        </div>
        <div class="boot-bar"><span style=${{ width: p.toFixed(1) + "%" }}></span></div>
      </div>
    </div>`;
}

/* ---------------------------------------------------------------- chrome */

/** The real system clock. Ticks once a second; frozen if motion is reduced. */
function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    if (prefersReducedMotion()) return;
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const p = (n) => String(n).padStart(2, "0");
  return html`<span class="clock">
    ${p(now.getHours())}:${p(now.getMinutes())}:${p(now.getSeconds())}</span>`;
}

/** Ticker content, built entirely from the exported JSON. No invented events. */
function tickerItems(summary, queue) {
  const out = [];
  (queue.queue || []).slice(0, 5).forEach((r) => {
    out.push([`RANK ${String(r.rank).padStart(2, "0")}`,
              `${money(r.exposure_cents)} ${CURRENCY}`,
              r.exception_class.replace(/_/g, " ").toUpperCase()]);
  });
  (summary.exception_classes || []).forEach((c) => {
    out.push([c.exception_class.replace(/_/g, " ").toUpperCase(),
              `${int(c.rows)} ROWS`,
              `${moneyShort(c.exposure_cents)} ${CURRENCY} EXPOSURE`]);
  });
  (summary.triggers || []).filter((t) => t.rows > 0).forEach((t) => {
    out.push([t.trigger.replace(/_/g, " ").toUpperCase(),
              t.verdict.toUpperCase(),
              `${pct(t.correct_pct, 1)} ALREADY CORRECT`]);
  });
  out.push(["AUTO-CLOSE PRECISION", pct(summary.auto_close_precision_by_rows_pct, 2) + " BY ROWS",
            pct(summary.auto_close_precision_by_value_pct, 2) + " BY VALUE"]);
  return out;
}

function Ticker({ summary, queue }) {
  const items = useMemo(() => tickerItems(summary, queue), [summary, queue]);
  const run = (key) => html`
    <div class="ticker-track-half" key=${key}>
      ${items.map((it, i) => html`
        <span class="ticker-item" key=${i}>
          <b>${it[0]}</b><span class="ticker-sep">/</span>${it[1]}
          <span class="ticker-sep">/</span>${it[2]}
        </span>`)}
    </div>`;
  return html`
    <div class="ticker" role="marquee" aria-label="Live figures from the exported batch">
      <div class="ticker-track">${run("a")}${run("b")}</div>
    </div>`;
}

function TopBar({ summary, batch, setBatch, synthAvailable, onHome }) {
  const hue = hueForBatch(batch);
  const fg = readableOn(hue);
  return html`
    <header class="topbar">
      <button class="wordmark" onClick=${onHome} title="Back to the cover">Ledgerline</button>
      <${Clock} />
      <span class="chip on-lime"><span class="dot-pulse"></span>Batch loaded</span>
      <span class="chip on-batch" style=${{ background: hue, color: fg }}>${
        `${(summary.batch || "").toUpperCase()} · ${int(summary.records_processed)} records`
      }</span>
      <span class="bar-spacer"></span>
      <span class="switcher">
        ${BATCHES.map((b) => html`
          <button key=${b.key} aria-pressed=${batch === b.key}
                  disabled=${b.key === "synth" && !synthAvailable}
                  title=${b.key === "synth" && !synthAvailable
                    ? "Not exported — run: python export.py --synth" : b.label}
                  style=${batch === b.key
                    ? { background: hueForBatch(b.key), color: readableOn(hueForBatch(b.key)) }
                    : {}}
                  onClick=${() => setBatch(b.key)}>${b.label}</button>`)}
      </span>
    </header>`;
}

const NAV = [
  ["overview", "01", "Overview", PAL.lime],
  ["summary",  "02", "Summary",  PAL.yellow],
  ["queue",    "03", "Queue",    PAL.tangerine],
  ["curve",    "04", "Exposure", PAL.violet],
];

/* Drawn, not an HTML entity — htm does not decode entities, so `&rarr;` would
 * render as its own literal text. */
const ARROW = html`
  <svg class="arrow" viewBox="0 0 16 12" aria-hidden="true">
    <path d="M9.6 1 14.6 6l-5 5M14.6 6H1.4" fill="none" stroke="currentColor"
          stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
  </svg>`;

function Rail({ active, onGo, onTour, tourTaken }) {
  return html`
    <nav class="rail" aria-label="Sections">
      ${NAV.map(([id, n, label, hue]) => html`
        <a key=${id} href=${"#" + id} class=${active === id ? "active" : ""}
           aria-current=${active === id ? "true" : "false"}
           onClick=${(e) => { e.preventDefault(); onGo(id); }}>
          <span class="n">${n}</span>
          <svg class="glyph" viewBox="0 0 10 10" aria-hidden="true">
            <rect x="0" y="0" width="10" height="10" rx="2" fill=${hue} />
          </svg>
          <span class="rlabel">${label}</span>
          ${ARROW}
        </a>`)}
      <button class="railtour" onClick=${onTour}
              title=${tourTaken
                ? "You have taken this tour — play it again"
                : "Walk through the whole page in eight steps"}>
        ${TOUR_PLAY}Play tour
      </button>
      <div class="rail-foot">
        Static export. Nothing is recomputed in the browser.
      </div>
    </nav>`;
}

/* ---------------------------------------------------------------- section head */

/* `batch` is printed under every section title. Scrolling past any header then says
 * which batch is on screen, rather than leaving the reader to remember what the top
 * bar said several screens ago. */
function SecHead({ n, kicker, batch, children }) {
  return html`
    <div class="sec-head">
      <div class="sec-num">${n}</div>
      <div>
        <div class="sec-kicker">${kicker}</div>
        <h2 class="display sec-title">${children}</h2>
        ${batch ? html`<div class="sec-batch">${batch}</div>` : ""}
      </div>
    </div>`;
}

/** "BenchRec eval · 32,048 records", from the loaded summary. */
function batchLine(s) {
  return `${s.batch} · ${int(s.records_processed)} records`;
}

/* ---------------------------------------------------------------- metric cards */

const GLYPHS = {
  circle: html`<svg width="22" height="22" viewBox="0 0 22 22"><circle cx="11" cy="11" r="8"
             fill="none" stroke="currentColor" stroke-width="2.5"/></svg>`,
  square: html`<svg width="22" height="22" viewBox="0 0 22 22"><rect x="3.5" y="3.5" width="15"
             height="15" rx="2" fill="none" stroke="currentColor" stroke-width="2.5"/></svg>`,
  tri:    html`<svg width="22" height="22" viewBox="0 0 22 22"><path d="M11 3.5 19 18.5H3z"
             fill="none" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/></svg>`,
  bars:   html`<svg width="22" height="22" viewBox="0 0 22 22"><g stroke="currentColor"
             stroke-width="2.5"><path d="M4 18V9"/><path d="M11 18V4"/><path d="M18 18v-6"/></g></svg>`,
  ring:   html`<svg width="22" height="22" viewBox="0 0 22 22"><circle cx="11" cy="11" r="8"
             fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="11" cy="11" r="3"
             fill="currentColor"/></svg>`,
};

function MetricCard({ bg, glyph, label, value, sub }) {
  const fg = readableOn(bg);
  return html`
    <div class="mcard" style=${{ background: bg, color: fg }}>
      <div class="glyph">${GLYPHS[glyph]}</div>
      <div class="m-lab">${label}</div>
      <div>
        <div class="m-val">${value}</div>
        ${sub ? html`<div class="m-sub">${sub}</div>` : ""}
      </div>
    </div>`;
}

/* ---------------------------------------------------------------- overview */

/** Layout for the whole queue. Each cell keeps its source row so a caller can annotate
 *  a specific rectangle. */
function treemapCells(queue, W, H) {
  if (!queue || !queue.length) return [];
  const items = queue.map((r) => ({ value: r.exposure_cents, cls: r.exception_class, row: r }))
                     .sort((a, b) => b.value - a.value);
  return treemap(items, 0, 0, W, H);
}

/* Deterministic stand-in for randomness: the ambient cycle needs to look unpatterned
 * but must render identically every time, or the server-render check has nothing
 * stable to assert against. */
const jitter = (i) => {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

const DRAW_MS = 900;         // largest to smallest, start to finish
const CELL_MS = 260;         // how long one rectangle takes
const PULSE_CELLS = 216;     // only rectangles big enough to see it breathe
const BANDS = 6;             // how many independent breathing cycles

/* `floor` is the opacity of the smallest rectangle. Raising the range between the
 * smallest and the largest is what turns 3,133 rectangles from an even texture into a
 * visible hierarchy; the gap between them widens with size for the same reason.
 *
 * `animate` draws them in from largest to smallest and then lets the big ones drift on
 * long, offset cycles. Both are pure CSS off per-cell custom properties, so the main
 * thread is not running 3,133 timers. */
function Treemap({ queue, W = 1600, H = 720, fit = "xMidYMid meet", cells: given,
                   floor = 0.46, highlight = -1, animate = false }) {
  const cells = useMemo(() => given || treemapCells(queue, W, H), [given, queue, W, H]);
  const maxA = cells.length ? cells[0].w * cells[0].h : 1;
  const big = highlight >= 0 ? cells[highlight] : null;
  const span = Math.max(cells.length - 1, 1);

  const cell = (c, i) => {
    const rel = Math.sqrt((c.w * c.h) / maxA);
    const gap = 2.4 + 7 * rel;
    const w = Math.max(c.w - gap, 0.4), h = Math.max(c.h - gap, 0.4);
    const o = floor + (1 - floor) * rel;
    return html`
      <rect key=${i} x=${(c.x + gap / 2).toFixed(2)} y=${(c.y + gap / 2).toFixed(2)}
            width=${w.toFixed(2)} height=${h.toFixed(2)}
            rx=${w > 16 && h > 16 ? 4 : 0}
            class=${animate ? "tm-cell" : null}
            style=${animate ? { "--o": o.toFixed(3),
                                "--d": ((i / span) * (DRAW_MS - CELL_MS)).toFixed(0) + "ms" }
                            : undefined}
            fill=${hueFor(c.item.cls)}
            opacity=${o.toFixed(3)} />`;
  };

  /* Breathing runs on a handful of groups rather than on every rectangle. Animating
   * 216 SVG rects individually put ~75% of idle wall time into style recalculation —
   * SVG child opacity is not compositable, so every frame restyled the subtree. Six
   * groups on offset cycles cost six animations instead of 216, and because
   * consecutive rectangles land in different groups, neighbours still drift apart
   * and the mosaic reads as breathing rather than as six blocks blinking. */
  const banded = [];
  if (animate) {
    for (let b = 0; b < BANDS; b++) banded.push([]);
    for (let i = 0; i < Math.min(PULSE_CELLS, cells.length); i++) {
      banded[i % BANDS].push([cells[i], i]);
    }
  }

  return html`
    <svg viewBox=${`0 0 ${W} ${H}`} role="img" preserveAspectRatio=${fit}
         aria-label="Every escalated transaction, sized by exposure, coloured by exception class">
      <rect x="0" y="0" width=${W} height=${H} fill=${PAL.dark} />
      ${animate
        ? banded.map((band, b) => html`
            <g key=${"band" + b} class="tm-band"
               style=${{ "--pt": (8 + jitter(b) * 7).toFixed(1) + "s",
                         "--pd": (DRAW_MS + 400 + jitter(b + 41) * 7000).toFixed(0) + "ms" }}>
              ${band.map(([c, i]) => cell(c, i))}
            </g>`)
        : ""}
      ${(animate ? cells.slice(PULSE_CELLS) : cells)
          .map((c, k) => cell(c, animate ? k + PULSE_CELLS : k))}
      ${big ? html`
        <rect key=${"hi" + highlight}
              x=${(big.x + 2).toFixed(2)} y=${(big.y + 2).toFixed(2)}
              width=${Math.max(big.w - 4, 1).toFixed(2)} height=${Math.max(big.h - 4, 1).toFixed(2)}
              rx="4" fill="none" stroke=${PAL.lime} stroke-width="4"
              class=${animate ? "tm-hi" : null}
              vector-effect="non-scaling-stroke" />` : ""}
    </svg>`;
}

function Overview({ summary, queue, complete, batchLine }) {
  const [ref, seen] = useReveal();
  const v = summary.value;
  const top = (queue.queue || [])[0];
  const classes = (summary.exception_classes || [])
    .slice().sort((a, b) => b.exposure_cents - a.exposure_cents);

  return html`
    <section id="overview" ref=${ref} class=${"reveal" + (seen ? " in" : "")}>
      <div class="wrap">
        <${SecHead} n="01" kicker="Overview" batch=${batchLine}>
          Cash reconciliation, reviewed by <span class="acc-tang">exposure</span>.
        <//>
        <p class="lede sec-sub">
          It closes the matches it can defend, escalates the rest with a reason, and ranks
          the queue by how much money is sitting in it. Every rectangle below is one
          escalated transaction, sized by exposure and coloured by exception class.
        </p>

        <div class="stage">
          <div class="treemap-wrap">${complete
            ? html`<${Treemap} queue=${queue.queue} />`
            : html`<div class="map-wait">${
                `Loading all ${int(queue.rows)} escalated transactions…`
              }</div>`}</div>

          <svg class="pointer" style=${{ inset: 0, width: "100%", height: "100%" }}
               viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <line x1="26" y1="17" x2="12" y2="30" stroke=${PAL.lime} stroke-width=".35"
                  vector-effect="non-scaling-stroke" />
            <line x1="72" y1="74" x2="86" y2="62" stroke=${PAL.violet} stroke-width=".35"
                  vector-effect="non-scaling-stroke" />
          </svg>

          ${top ? html`
            <div class="callout on-lime" style=${{ top: "6%", left: "26%" }}>
              <div class="c-lab">Largest single exposure</div>
              <div class="c-val num">${money(top.exposure_cents)} ${CURRENCY}</div>
              <div class="c-note">
                B_id ${top.b_id} — ${top.exception_class.replace(/_/g, " ")}
              </div>
            </div>` : ""}

          ${classes[0] ? html`
            <div class="callout" style=${{ bottom: "8%", right: "4%" }}>
              <div class="c-lab">Largest class by value</div>
              <div class="c-val num">${moneyShort(classes[0].exposure_cents)} ${CURRENCY}</div>
              <div class="c-cls" style=${{ color: inkFor(classes[0].exception_class) }}>${
                classes[0].exception_class.replace(/_/g, " ")
              }</div>
              <div class="c-note">${
                `${int(classes[0].rows)} rows, ` +
                `${pct(classes[0].pct_of_queue_value, 1)} of queue value`
              }</div>
            </div>` : ""}
        </div>

        <div class="metrics">
          <${MetricCard} bg=${CARD_HUE[0]} glyph="circle" label="Records processed"
            value=${html`<${CountInt} value=${summary.records_processed} />`}
            sub=${`${summary.batch}`} />
          <${MetricCard} bg=${CARD_HUE[1]} glyph="square" label="Auto-closed"
            value=${html`<${CountInt} value=${summary.auto_closed} />`}
            sub=${pct(summary.auto_close_rate_pct, 2) + " coverage"} />
          <${MetricCard} bg=${CARD_HUE[2]} glyph="tri" label="Escalated"
            value=${html`<${CountInt} value=${summary.escalated} />`}
            sub=${`${moneyShort(v.in_exception_queue)} ${CURRENCY} at risk`} />
          <${MetricCard} bg=${CARD_HUE[3]} glyph="bars" label="Throughput"
            value=${summary.throughput_records_per_sec
              ? nf(1, 1).format(summary.throughput_records_per_sec) : "—"}
            sub=${summary.throughput_records_per_sec ? "records / second" : "not recorded"} />
          <${MetricCard} bg=${CARD_HUE[4]} glyph="ring" label="Auto-close precision"
            value=${html`<${CountPct} value=${summary.auto_close_precision_by_rows_pct} dp=${2} />`}
            sub=${pct(summary.auto_close_precision_by_value_pct, 2) + " by value"} />
        </div>
      </div>
    </section>`;
}

/* ---------------------------------------------------------------- summary */

function Fig({ label, value, right, tone }) {
  const col = tone ? MONEY[tone] : null;
  return html`
    <div class="fig">
      <div class="swatch" style=${{ background: col || "transparent" }}></div>
      <div class="lab">${label}</div>
      <div class="val num" style=${col ? { color: col } : {}}>${value}</div>
      <div class="pct num">${right || ""}</div>
    </div>`;
}

/* Categorised, not alarming: the class hue as a wash behind the word, and the word
 * itself in the darkened hue at full strength. A column of these reads as a taxonomy;
 * a column of saturated fills reads as 3,133 alerts. */
function ClassPill({ name }) {
  const h = hueFor(name);
  return html`<span class="pill" style=${{ background: tint(h, .15), color: inkFor(name),
                                           borderColor: tint(h, .40) }}>${name}</span>`;
}

function Summary({ s, batchLine }) {
  const [ref, seen] = useReveal();
  const v = s.value;
  const share = (c) => pct((c / v.total_batch) * 100, 2);
  const byRows = s.auto_close_precision_by_rows_pct;
  const byValue = s.auto_close_precision_by_value_pct;
  const gap = byValue - byRows;

  return html`
    <section id="summary" ref=${ref} class=${"reveal" + (seen ? " in" : "")}>
      <div class="wrap">
        <${SecHead} n="02" kicker="Executive summary" batch=${batchLine}>
          What the controller <span class="acc-violet">closed</span>.
        <//>
        <p class="sec-sub note">
          ${s.batch}. Counts and throughput as recorded by the decision layer; money figures
          re-derived from the same audit records. All amounts ${CURRENCY}, stored as integer
          cents and formatted here for display.
        </p>

        <div class="grid2">
          <div class="panel" style=${{ padding: "26px 28px 24px" }}>
            <h3 class="eyebrow u-mb10">Volumes</h3>
            <div class="figs">
              <${Fig} label="Records processed" value=${int(s.records_processed)} />
              <${Fig} label="Auto-closed" value=${int(s.auto_closed)}
                      right=${pct(s.auto_close_rate_pct, 2)} />
              <${Fig} label="Escalated for review" value=${int(s.escalated)}
                      right=${pct(s.escalation_rate_pct, 2)} />
              <${Fig} label="Throughput"
                      value=${s.throughput_records_per_sec
                        ? nf(1, 1).format(s.throughput_records_per_sec) + " rec/s" : "—"}
                      right=${s.wall_clock_sec ? nf(1, 1).format(s.wall_clock_sec) + " s" : ""} />
            </div>
          </div>
          <div class="panel" style=${{ padding: "26px 28px 24px" }}>
            <h3 class="eyebrow u-mb10">Value (${CURRENCY})</h3>
            <div class="figs">
              <${Fig} label="Total in batch" value=${money(v.total_batch)} />
              <${Fig} label="Auto-closed correctly" value=${money(v.auto_closed_correct)}
                      right=${share(v.auto_closed_correct)} tone="correct" />
              <${Fig} label="In the exception queue" value=${money(v.in_exception_queue)}
                      right=${share(v.in_exception_queue)} tone="queue" />
              <${Fig} label="Auto-closed incorrectly" value=${money(v.auto_closed_incorrect)}
                      right=${share(v.auto_closed_incorrect)} tone="wrong" />
            </div>
          </div>
        </div>

        <h3 class="eyebrow u-head34">Precision, measured two ways</h3>
        <div class="prec panel">
          <div>
            <div class="k">By row count</div>
            <div class="v num">${pct(byRows)}</div>
            <div class="d">
              Of the ${int(s.auto_closed)} transactions closed without review, this share was
              correct. Every transaction counts once, whatever it was worth.
            </div>
          </div>
          <div>
            <div class="k">By value</div>
            <div class="v num">${pct(byValue)}</div>
            <div class="d">
              Of the ${money(v.auto_closed_total)} ${CURRENCY} closed without review, this
              share was correct. Each transaction counts in proportion to its amount.
            </div>
          </div>
          <div class="gapcell">
            <div class="k">Difference</div>
            <div class="g num">${(gap >= 0 ? "+" : "−") + nf(3, 3).format(Math.abs(gap))}</div>
            <div class="d">
              ${gap >= 0
                ? "points higher when weighted by amount — errors fall on smaller-than-average transactions."
                : "points lower when weighted by amount — errors fall on larger-than-average transactions."}
            </div>
          </div>
        </div>

        ${s.reference ? html`
          <div class="refbox">
            <div class="ref-lab">For reference</div>
            <div class="ref-rows">
              <div class="ref-row me">
                <div class="ref-who">This system, every answer posted blind</div>
                <div class="ref-fig num">${pct(s.overall_match_pct, 4)}</div>
                <div class="ref-fig num">${pct(s.overall_precision_pct, 4)}</div>
                <div class="ref-fig num">${pct(s.overall_abstention_pct, 4)}</div>
              </div>
              <div class="ref-row">
                <div class="ref-who mono">${s.reference.source_file}</div>
                <div class="ref-fig num">${pct(s.reference.match_rate_pct, 4)}</div>
                <div class="ref-fig num">${pct(s.reference.match_precision_pct, 4)}</div>
                <div class="ref-fig num">${pct(s.reference.abstention_rate_pct, 4)}</div>
              </div>
              <div class="ref-row head">
                <div class="ref-who"></div>
                <div class="ref-fig">match rate</div>
                <div class="ref-fig">precision</div>
                <div class="ref-fig">abstention</div>
              </div>
            </div>
            <p class="note-sm ref-note">${
              `${s.reference.provenance} Scored against ` +
              `${s.reference.scored_against} by score.py; these figures are read from ` +
              `score.log at export time, not typed in here.`
            }</p>
          </div>` : ""}

        <div class="grid2 u-mt38">
          <div class="panel" style=${{ padding: "26px 28px 24px" }}>
            <h3 class="eyebrow u-mb10">Exception classes</h3>
            <div class="tscroll"><table>
              <thead><tr>
                <th>Class</th><th class="r">Rows</th>
                <th class="r">Exposure (${CURRENCY})</th><th class="r">% of queue</th>
              </tr></thead>
              <tbody>
                ${s.exception_classes.slice().sort((a, b) => b.exposure_cents - a.exposure_cents)
                  .map((c) => html`
                  <tr key=${c.exception_class}>
                    <td><${ClassPill} name=${c.exception_class} /></td>
                    <td class="r num">${int(c.rows)}</td>
                    <td class="r num">${money(c.exposure_cents)}</td>
                    <td class="r num">${pct(c.pct_of_queue_value, 1)}</td>
                  </tr>`)}
              </tbody>
            </table></div>
          </div>
          <div class="panel" style=${{ padding: "26px 28px 24px" }}>
            <h3 class="eyebrow u-mb10">Triggers</h3>
            <div class="tscroll"><table>
              <thead><tr>
                <th>Trigger</th><th class="r">Rows</th>
                <th class="r">Already correct</th><th>Verdict</th>
              </tr></thead>
              <tbody>
                ${s.triggers.map((t) => html`
                  <tr key=${t.trigger}>
                    <td class="mono">${t.trigger}</td>
                    <td class="r num">${int(t.rows)}</td>
                    <td class="r num">${t.correct_pct === null ? "—" : pct(t.correct_pct, 1)}</td>
                    <td class="note-sm">${t.verdict}</td>
                  </tr>`)}
              </tbody>
            </table></div>
            <p class="note-sm u-mt9">
              “Already correct” is the share of rows a trigger escalated that would have been
              right if closed blind — coverage given up rather than error caught.
            </p>
          </div>
        </div>
      </div>
    </section>`;
}

/* ---------------------------------------------------------------- queue */

const ROW_H = 52;
const OVERSCAN = 8;

function Queue({ queue, total, complete, selected, onSelect, batchLine }) {
  const [q, setQ] = useState("");
  const [cls, setCls] = useState("");
  const [scrollTop, setScrollTop] = useState(0);
  const [viewH, setViewH] = useState(588);
  const boxRef = useRef(null);
  const [ref, seen] = useReveal();

  const classes = useMemo(
    () => Array.from(new Set(queue.map((r) => r.exception_class))).sort(), [queue]);

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return queue.filter((r) => {
      if (cls && r.exception_class !== cls) return false;
      if (!needle) return true;
      return r.b_id.includes(needle) ||
             r.exception_class.toLowerCase().includes(needle) ||
             (r.triggers || []).some((t) => t.toLowerCase().includes(needle));
    });
  }, [queue, q, cls]);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewH(el.clientHeight));
    ro.observe(el); setViewH(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  const reset = () => { setScrollTop(0); if (boxRef.current) boxRef.current.scrollTop = 0; };
  const maxExposure = useMemo(
    () => queue.reduce((m, r) => Math.max(m, r.exposure_cents), 1), [queue]);

  const first = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN);
  const count = Math.min(rows.length - first, Math.ceil(viewH / ROW_H) + OVERSCAN * 2);
  const slice = rows.slice(first, first + Math.max(count, 0));
  const totalExposure = useMemo(() => rows.reduce((a, r) => a + r.exposure_cents, 0), [rows]);

  return html`
    <section id="queue" ref=${ref} class=${"reveal" + (seen ? " in" : "")}>
      <div class="wrap">
        <${SecHead} n="03" kicker="Exception queue" batch=${batchLine}>
          What it <span class="acc-tang">refused</span> to close.
        <//>
        <p class="sec-sub note">
          Every escalated transaction, ranked by absolute amount descending. Exposure is the
          full transaction value at risk pending review — including rows whose correct answer
          turns out to be “no match”. Select a row for its candidates and evidence.
        </p>

        <div class="qbar">
          <input type="search" placeholder="Filter by B_id, class or trigger"
                 value=${q} onInput=${(e) => { setQ(e.target.value); reset(); }} />
          <select value=${cls} onChange=${(e) => { setCls(e.target.value); reset(); }}>
            <option value="">All classes</option>
            ${classes.map((c) => html`<option key=${c} value=${c}>${c}</option>`)}
          </select>
          <span class="note-sm">${
            `${int(rows.length)} of ${int(total === undefined ? queue.length : total)} rows` +
            (complete === false ? ` (loading the rest)` : ``) +
            ` · ${money(totalExposure)} ${CURRENCY}` +
            ` · ${slice.length} mounted`
          }</span>
        </div>

        <div class="qpanel">
          <div class="qhead">
            <div class="r">Rank</div>
            <div class="r">Amount (${CURRENCY})</div>
            <div>Exception class</div>
            <div>Triggers</div>
            <div>Evidence</div>
          </div>
          <div class="qscroll" ref=${boxRef}
               onScroll=${(e) => setScrollTop(e.currentTarget.scrollTop)}>
            ${rows.length === 0
              ? html`<div class="empty">No rows match this filter.</div>`
              : html`
                <div style=${{ height: rows.length * ROW_H + "px", position: "relative" }}>
                  <div style=${{ transform: `translateY(${first * ROW_H}px)` }}>
                    ${slice.map((r) => {
                      const hue = hueFor(r.exception_class);
                      const barW = Math.max((r.exposure_cents / maxExposure) * 100, 0.6);
                      return html`
                      <button key=${r.b_id}
                              class=${"qrow" + (r.rank === 1 ? " top" : "")}
                              role="option" aria-selected=${selected === r.b_id}
                              onClick=${() => onSelect(r.b_id)}>
                        <span class="rank num">${int(r.rank)}</span>
                        <span class="amtcell">
                          <span class="bar" aria-hidden="true" style=${{ background:
                            `linear-gradient(to left, ${tint(hue, .26)} 0%,` +
                            ` ${tint(hue, .26)} ${barW.toFixed(2)}%,` +
                            ` ${tint(hue, 0)} ${Math.min(barW + 16, 100).toFixed(2)}%)` }}></span>
                          <span class="amt num">${money(r.exposure_cents)}</span>
                        </span>
                        <span><${ClassPill} name=${r.exception_class} /></span>
                        <span>${(r.triggers || []).map((t) =>
                          html`<span key=${t} class="tag">${t}</span>`)}</span>
                        <span class="ev" title=${r.evidence}>${r.evidence}</span>
                      </button>`;
                    })}
                  </div>
                </div>`}
          </div>
        </div>
        <p class="note-sm u-mt9">
          Virtualised: only the rows in view are mounted, so this section never
          renders ${int(queue.length)} rows at once.
        </p>
      </div>
    </section>`;
}

/* ---------------------------------------------------------------- curve */

function Curve({ curve, batchLine }) {
  const [ref, seen] = useReveal();
  const drawn = seen ? " drawn" : "";
  const W = 900, H = 400, P = { t: 20, r: 24, b: 50, l: 66 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const pts = [{ review_top_pct: 0, ranked_pct: 0, random_pct: 0, rows_reviewed: 0 }]
    .concat(curve.points);
  const x = (p) => P.l + (p / 100) * iw;
  const y = (v) => P.t + ih - (v / 100) * ih;
  const line = (key) => pts.map((p, i) =>
    `${i ? "L" : "M"}${x(p.review_top_pct).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");
  const area =
    pts.map((p, i) => `${i ? "L" : "M"}${x(p.review_top_pct).toFixed(1)},${y(p.ranked_pct).toFixed(1)}`).join("") +
    pts.slice().reverse().map((p) => `L${x(p.review_top_pct).toFixed(1)},${y(p.random_pct).toFixed(1)}`).join("") + "Z";
  const ticks = [0, 25, 50, 75, 100];
  const at10 = curve.points.find((p) => p.review_top_pct === 10);

  return html`
    <section id="curve" ref=${ref} class=${"reveal" + (seen ? " in" : "")}>
      <div class="wrap">
        <${SecHead} n="04" kicker="Exposure analysis" batch=${batchLine}>
          Where the <span class="acc-violet">money</span> is.
        <//>
        <p class="sec-sub note">
          Share of total queue exposure retired by reviewing the top N% of the queue. Ranking
          by exposure is only worth doing if it beats reviewing in a random order, so both are
          drawn and the gap between them is shaded. Random is
          the mean of ${curve.random_seeds} shuffles.
        </p>

        <div class="grid-curve stage">
          <div class="panel-dark chart">
            <svg viewBox=${`0 0 ${W} ${H}`} width="100%" role="img"
                 aria-label="Exposure retired against share of queue reviewed">
              ${ticks.map((t) => html`
                <g key=${"g" + t}>
                  <line x1=${P.l} x2=${W - P.r} y1=${y(t)} y2=${y(t)}
                        stroke=${t === 0 ? "rgba(245,242,234,.34)" : "rgba(245,242,234,.10)"} />
                  <text x=${P.l - 12} y=${y(t) + 4} text-anchor="end" font-size="11.5"
                        fill="rgba(245,242,234,.58)" font-family="JetBrains Mono, monospace">
                    ${t}%</text>
                </g>`)}
              ${ticks.map((t) => html`
                <text key=${"x" + t} x=${x(t)} y=${H - P.b + 24} text-anchor="middle"
                      font-size="11.5" fill="rgba(245,242,234,.58)"
                      font-family="JetBrains Mono, monospace">${t}%</text>`)}

              <path d=${area} fill=${CHART_ACCENT} class=${"area-fade" + drawn} />
              <path d=${line("random_pct")} fill="none" stroke="rgba(245,242,234,.46)"
                    stroke-width="2.5" stroke-dasharray="6 5" stroke-linecap="round"
                    pathLength="1" class=${"draw d-random" + drawn} />
              <path d=${line("ranked_pct")} fill="none" stroke=${CHART_ACCENT} stroke-width="4"
                    stroke-linecap="round" stroke-linejoin="round" pathLength="1"
                    class=${"draw d-ranked" + drawn} />
              ${curve.points.map((p) => html`
                <circle key=${"p" + p.review_top_pct} cx=${x(p.review_top_pct)}
                        cy=${y(p.ranked_pct)} r="4.5" fill=${CHART_ACCENT}
                        class=${"cal-fade" + drawn} />`)}
              ${at10 ? html`
                <g class=${"cal-fade" + drawn}>
                  <line x1=${x(10)} x2=${x(10)} y1=${y(at10.ranked_pct)} y2=${y(at10.random_pct)}
                        stroke=${CHART_ACCENT} stroke-width="2" stroke-dasharray="4 4" />
                  <circle cx=${x(10)} cy=${y(at10.random_pct)} r="4"
                          fill="rgba(245,242,234,.62)" />
                </g>` : ""}
              <line x1=${P.l} x2=${P.l} y1=${P.t} y2=${P.t + ih}
                    stroke="rgba(245,242,234,.34)" />
              <text x=${P.l + iw / 2} y=${H - 8} text-anchor="middle" font-size="12"
                    fill="rgba(245,242,234,.72)">Share of queue reviewed</text>
              <text transform=${`translate(18,${P.t + ih / 2}) rotate(-90)`}
                    text-anchor="middle" font-size="12"
                    fill="rgba(245,242,234,.72)">Exposure retired</text>
            </svg>
            <div class="legend">
              <span><i style=${{ borderColor: CHART_ACCENT }}></i>Ranked by exposure</span>
              <span><i style=${{ borderColor: "rgba(245,242,234,.46)",
                                 borderTopStyle: "dashed" }}></i>Random order</span>
              <span><span class="sw-fill" style=${{ background: CHART_ACCENT, opacity: .3 }}></span>
                Gain from ranking</span>
            </div>
          </div>

          <div class="panel" style=${{ padding: "26px 28px 24px" }}>
            <div class="tscroll"><table>
              <thead><tr>
                <th class="r">Review</th><th class="r">Rows</th>
                <th class="r">Ranked</th><th class="r">Random</th><th class="r">Gain</th>
              </tr></thead>
              <tbody>
                ${curve.points.map((p) => html`
                  <tr key=${p.review_top_pct}>
                    <td class="r num">${p.review_top_pct}%</td>
                    <td class="r num">${int(p.rows_reviewed)}</td>
                    <td class="r num u-bold">${pct(p.ranked_pct, 1)}</td>
                    <td class="r num u-muted">${pct(p.random_pct, 1)}</td>
                    <td class="r num">+${nf(1, 1).format(p.ranked_pct - p.random_pct)}</td>
                  </tr>`)}
              </tbody>
            </table></div>
            <p class="note-sm u-mt10">${
              `Queue holds ${money(curve.total_exposure_cents)} ${CURRENCY} ` +
              `across ${int(curve.queue_rows)} transactions.`
            }</p>
          </div>

          ${at10 ? html`
            <div class="callout on-tang" style=${{ top: "-14px", left: "34%" }}>
              <div class="c-lab">Reviewing the top 10%</div>
              <div class="c-val num">${pct(at10.ranked_pct, 1)} retired</div>
              <div class="c-note">${
                `against ${pct(at10.random_pct, 1)} in a random order — a gain of ` +
                `${nf(1, 1).format(at10.ranked_pct - at10.random_pct)} points`
              }</div>
            </div>` : ""}
        </div>
      </div>
    </section>`;
}

/* ---------------------------------------------------------------- detail */

function triggerReasons(d) {
  const out = [];
  const has = (t) => (d.triggers || []).includes(t);
  if (has("no_candidate")) {
    out.push(["no_candidate",
      `No candidate survived blocking — the pool held ${int(d.candidate_pool_size || 0)} rows. ` +
      `Escalated as a proposed no-match: “no allocation” may be the correct answer, but it is ` +
      `not posted without review.`]);
  }
  if (has("fee_band_only")) {
    const top = (d.candidates || [])[0];
    out.push(["fee_band_only",
      `The best candidate was admitted only by the widened fee band, not by an exact amount` +
      (top ? ` — it differs from the transaction by ${money(top.delta_from_b_cents)} ${CURRENCY}.` : ".")]);
  }
  if (has("completion_added")) {
    const ps = (d.added_keys || []).map((a) => a.probability)
      .filter((p) => p !== null && p !== undefined);
    out.push(["completion_added",
      `${(d.added_keys || []).length} allocation key(s) came from the completion classifier ` +
      `rather than from retrieval` +
      (ps.length ? ` (probability ${ps.map((p) => nf(3, 3).format(p)).join(", ")}).` : ".") +
      ` Added keys are weaker evidence than retrieved ones, so the answer is not self-closing.`]);
  }
  if (has("low_confidence")) {
    out.push(["low_confidence",
      `Top-1 similarity ${score(d.top1_score)} or the rank-1/rank-2 margin ${score(d.margin)} ` +
      `fell below the configured floor.`]);
  }
  return out;
}

function Investigation({ inv, detail }) {
  if (!inv) {
    return html`
      <div class="banner none">
        <strong>Not investigated.</strong> No explanation has been generated for this
        transaction. Explanations cover a subset of the queue only; this row is not in it.
      </div>`;
  }
  const bad = inv.grounded === false;
  const evidenceNums = [];
  (detail.candidates || []).forEach((c) => {
    evidenceNums.push(`delta ${money(c.delta_from_b_cents)}`);
    evidenceNums.push(`score ${score(c.similarity_score)}`);
  });
  return html`
    <div>
      <div class=${"banner " + (bad ? "bad" : "good")}>
        ${bad
          ? html`<strong>Grounding check failed.</strong> This explanation
                 contains ${inv.ungrounded_tokens.length} number(s) that do not appear in the
                 evidence the model was given. Treat the text below as unverified.`
          : html`<strong>Grounding check passed.</strong> Every number in this explanation
                 appears in the evidence the model was given.`}
        ${inv.no_match_proposed === false
          ? html` The model also appears to have proposed a match, which it was instructed
                  not to do.` : ""}
      </div>
      ${bad ? inv.ungrounded_tokens.map((tok) => html`
        <div class="cmp" key=${tok}>
          <div><div class="k">Claimed in explanation</div>
               <div class="claimed num">${tok}</div></div>
          <div><div class="k">Present in the evidence</div>
               <div class="actual num">${evidenceNums.length
                 ? evidenceNums.join(" · ")
                 : "no candidate figures were supplied for this row"}</div></div>
        </div>`) : ""}
      <h3>Why it could not be closed</h3>
      <p class="prose">${inv.explanation}</p>
      <h3>Recommended action</h3>
      <p class="prose">${inv.recommended_action}</p>
      <h3>What would resolve it</h3>
      <p class="prose">${inv.information_needed}</p>
      <p class="note-sm">
        Generated by ${inv.model || "an unnamed model"}${inv.provider ? ` (${inv.provider})` : ""}.
        The model explains a decision that was already made; it does not make, rank or revise
        any match.
      </p>
    </div>`;
}

function Detail({ bId, batch, onClose }) {
  const { data: d, error } = useJson(`${DATA}/detail/${batch}/${bId}.json`);
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const t = d && d.transaction ? d.transaction : {};
  const cur = t.currency || CURRENCY;

  return html`
    <div>
      <div class="scrim" onClick=${onClose}></div>
      <aside class="drawer" role="dialog" aria-modal="true" aria-label=${`Transaction ${bId}`}>
        <header>
          <div>
            <div class="eyebrow">Transaction</div>
            <h2 class="bid">${bId}</h2>
            ${d ? html`<div class="note-sm u-mt4">
              ${d.decision === "escalate" ? "Escalated for review" : "Auto-closed"}
              ${d.exception_class ? ` · ${d.exception_class}` : ""}</div>` : ""}
          </div>
          <button class="xbtn" onClick=${onClose}>Close</button>
        </header>
        <div class="body">
          ${error ? html`<div class="banner bad">Could not load detail: ${error}</div>` : ""}
          ${!d && !error ? html`<div class="loading">Loading…</div>` : ""}
          ${d ? html`
            <h3>Transaction</h3>
            <dl class="kv">
              <dt>Amount</dt>
              <dd class="num u-bold">
                ${money(t.amount_cents)} ${cur}${t.debit_or_credit ? ` (${t.debit_or_credit})` : ""}
              </dd>
              <dt>Exposure if wrong</dt><dd class="num">${money(t.exposure_cents)} ${cur}</dd>
              <dt>Value date</dt><dd>${t.value_date || "—"}</dd>
              <dt>Import date</dt><dd>${t.import_date || "—"}</dd>
              <dt>References</dt><dd class="mono">${t.references || "—"}</dd>
              <dt>Attributes</dt><dd class="mono">${t.attributes || "—"}</dd>
            </dl>

            <h3>Why it was escalated</h3>
            ${triggerReasons(d).length === 0
              ? html`<p class="note">No trigger recorded.</p>`
              : triggerReasons(d).map(([name, why]) => html`
                  <div class="why" key=${name}>
                    <div class="t">${name}</div><div class="d">${why}</div>
                  </div>`)}

            <h3>Candidates considered (${(d.candidates || []).length})</h3>
            ${(d.candidates || []).length === 0
              ? html`<p class="note">
                  The candidate pool was empty after blocking, so no candidate was scored.</p>`
              : html`
                <div class="tscroll"><table>
                  <thead><tr>
                    <th class="r">#</th><th>Ledger ID</th>
                    <th class="r">Amount (${cur})</th><th class="r">Δ vs transaction</th>
                    <th class="r">Similarity</th><th>Exact</th>
                  </tr></thead>
                  <tbody>
                    ${d.candidates.map((c) => html`
                      <tr key=${c.a_id}>
                        <td class="r num">${c.rank}</td>
                        <td class="mono">${c.a_id}</td>
                        <td class="r num">${money(c.amount_cents)}</td>
                        <td class=${"r num" + (c.delta_from_b_cents === 0 ? "" : " u-neg")}>
                          ${money(c.delta_from_b_cents)}</td>
                        <td class="r num">${score(c.similarity_score)}</td>
                        <td>${c.exact_amount ? "yes" : "no"}</td>
                      </tr>`)}
                  </tbody>
                </table></div>`}

            <h3>Decision</h3>
            <dl class="kv">
              <dt>Outcome</dt>
              <dd>${d.decision === "escalate" ? "Escalated — not posted" : "Auto-closed"}</dd>
              <dt>Exception class</dt>
              <dd>${d.exception_class
                ? html`<${ClassPill} name=${d.exception_class} />` : "—"}</dd>
              <dt>Triggers fired</dt>
              <dd>${(d.triggers || []).length
                ? d.triggers.map((x) => html`<span key=${x} class="tag">${x}</span>`) : "none"}</dd>
              <dt>Top-1 similarity</dt><dd class="num">${score(d.top1_score)}</dd>
              <dt>Rank 1–2 margin</dt><dd class="num">${score(d.margin)}</dd>
              <dt>Exact amount on top-1</dt>
              <dd>${d.exact_amount_top1 === null || d.exact_amount_top1 === undefined
                ? "—" : d.exact_amount_top1 ? "yes" : "no"}</dd>
              <dt>Duplicate reference</dt>
              <dd>${d.duplicate_reference_among_candidates
                ? "yes — two or more candidates share a reference" : "no"}</dd>
              <dt>Proposed answer</dt>
              <dd>${(d.answer_keys || []).length
                ? html`<div class="mono u-break">
                    ${d.answer_keys.map((k) => html`<div key=${k}>${k}</div>`)}</div>`
                : html`<span class="note">empty — proposed no-match</span>`}</dd>
            </dl>

            <h3>Keys added by the completion classifier</h3>
            ${(d.added_keys || []).length === 0
              ? html`<p class="note">None. Every key in the proposed answer came from retrieval.</p>`
              : html`
                <div class="tscroll"><table>
                  <thead><tr><th>Allocation key</th><th class="r">Probability</th></tr></thead>
                  <tbody>
                    ${d.added_keys.map((a, i) => html`
                      <tr key=${i}>
                        <td class="mono u-break">${a.allocation_key}</td>
                        <td class="r num">${a.probability === null || a.probability === undefined
                          ? "not recorded" : nf(4, 4).format(a.probability)}</td>
                      </tr>`)}
                  </tbody>
                </table></div>`}

            <h3>Investigation</h3>
            <${Investigation} inv=${d.investigation} detail=${d} />
          ` : ""}
        </div>
      </aside>
    </div>`;
}

/* ---------------------------------------------------------------- guided tour
 *
 * Eight steps over the real page: each one navigates to the element it is talking
 * about, dims everything else, rings the element, and puts a card beside it. Every
 * figure in the copy is read from the exported JSON at render time — the tour states
 * what this batch actually did, not what a demo batch might have done.
 *
 * Whether it has been taken is remembered for the session only, and only to relabel
 * the button; the tour never starts on its own. */

const TOUR_SEEN_KEY = "ledgerline.tour.seen";
const TOUR_GAP = 18;         // between the ring and the card
const TOUR_EDGE = 16;        // from the card to the viewport
const TOUR_TOP = 104;        // where a step scrolls its element to
const TOUR_MIN_RING = 96;    // a ring never shrinks below this to make room

function tourSeen() {
  try { return window.sessionStorage.getItem(TOUR_SEEN_KEY) === "1"; } catch (e) { return false; }
}
function markTourSeen() {
  try { window.sessionStorage.setItem(TOUR_SEEN_KEY, "1"); } catch (e) { /* private mode */ }
}

/** The eight steps, with their copy built from this batch's own figures. */
function tourSteps(s, curve) {
  const v = s.value;
  const at10 = (curve.points || []).find((p) => p.review_top_pct === 10);
  const byRows = s.auto_close_precision_by_rows_pct;
  const byValue = s.auto_close_precision_by_value_pct;
  const gap = at10 ? at10.ranked_pct - at10.random_pct : 0;

  return [
    { view: "landing", target: ".landing-head", head: "What this is",
      body: "A reconciliation controller decides which bank-to-ledger matches it can " +
            "defend, and closes those automatically. Everything else goes to a human " +
            "with a reason attached." },

    { view: "landing", target: ".cover-figs", head: "The numbers",
      body: `It closed ${int(s.auto_closed)} of ${int(s.records_processed)} transactions ` +
            `on its own at ${pct(byRows, 2)} accuracy, and escalated ${int(s.escalated)} ` +
            `holding ${moneyShort(v.in_exception_queue)} ${CURRENCY}.` },

    { view: "landing", target: ".cover-map-panel", head: "The picture",
      body: "Every rectangle is one escalated transaction, sized by the money at risk " +
            "and coloured by why it stopped. The skew is the point: a handful of " +
            "rectangles hold most of the value." },

    { view: "controller", section: "summary", target: ".prec", head: "Two ways to measure",
      body: `${pct(byRows, 3)} by row count, ${pct(byValue, 3)} by value. The gap means ` +
            `errors land on ${byValue >= byRows ? "smaller" : "larger"}-than-average ` +
            `transactions.` },

    { view: "controller", section: "summary", target: ".grid2.u-mt38", head: "Why it stopped",
      body: `Every escalation carries a reason — ${int((s.exception_classes || []).length)} ` +
            `classes here — and each trigger is scored on whether it catches more wrong ` +
            `answers than right ones.` },

    { view: "controller", section: "queue", target: ".qpanel", head: "The queue",
      body: `All ${int(s.escalated)} escalations ranked by exposure, largest first, so a ` +
            `reviewer with limited hours works the biggest money first.` },

    { view: "controller", section: "queue", target: ".drawer", open: true, settle: 40,
      head: "One decision",
      body: "Every candidate considered, its amount, its score, and the triggers that " +
            "fired. Where an LLM explanation exists, so does its grounding check against " +
            "the evidence it was given." },

    { view: "controller", section: "curve", target: ".grid-curve", head: "Does ranking help",
      body: at10
        ? `Reviewing the top ${at10.review_top_pct}% retires ${pct(at10.ranked_pct, 1)} of ` +
          `queue value against ${pct(at10.random_pct, 1)} in a random order. That ` +
          `${nf(1, 1).format(gap)}-point gap is why the queue is ordered this way.`
        : "The curve compares reviewing by exposure against reviewing in a random order." },
  ];
}

/** Where the card goes. Never over the ring, never off the viewport.
 *  Below, then above, then right, then left; `ring` is already clamped to the screen. */
function tourCardPlacement(ring, card, vw, vh) {
  const cx = (x) => Math.min(Math.max(x, TOUR_EDGE), Math.max(vw - card.w - TOUR_EDGE, TOUR_EDGE));
  const cy = (y) => Math.min(Math.max(y, TOUR_EDGE), Math.max(vh - card.h - TOUR_EDGE, TOUR_EDGE));
  if (!ring) return { left: (vw - card.w) / 2, top: (vh - card.h) / 2, side: "centre" };

  const rb = ring.top + ring.height, rr = ring.left + ring.width;
  if (rb + TOUR_GAP + card.h <= vh - TOUR_EDGE)
    return { left: cx(ring.left), top: rb + TOUR_GAP, side: "below" };
  if (ring.top - TOUR_GAP - card.h >= TOUR_EDGE)
    return { left: cx(ring.left), top: ring.top - TOUR_GAP - card.h, side: "above" };
  if (rr + TOUR_GAP + card.w <= vw - TOUR_EDGE)
    return { left: rr + TOUR_GAP, top: cy(ring.top), side: "right" };
  if (ring.left - TOUR_GAP - card.w >= TOUR_EDGE)
    return { left: ring.left - TOUR_GAP - card.w, top: cy(ring.top), side: "left" };
  return { left: cx(ring.left), top: cy(vh - card.h - TOUR_EDGE), side: "forced" };
}

/** The ring, clamped to the viewport and then shortened if that is the only way to
 *  leave the card somewhere to stand. A tall panel is taller than the screen anyway,
 *  so framing the part you can see loses nothing. */
function tourRing(r, card, vw, vh, pad) {
  const top = Math.max(r.top - pad, TOUR_EDGE / 2);
  const left = Math.max(r.left - pad, TOUR_EDGE / 2);
  const right = Math.min(r.right + pad, vw - TOUR_EDGE / 2);
  let bottom = Math.min(r.bottom + pad, vh - TOUR_EDGE / 2);

  const needed = card.h + TOUR_GAP + TOUR_EDGE;
  const fitsBeside = (right + TOUR_GAP + card.w <= vw - TOUR_EDGE) ||
                     (left - TOUR_GAP - card.w >= TOUR_EDGE);
  const fitsAbove = top - TOUR_GAP - card.h >= TOUR_EDGE;
  if (!fitsBeside && !fitsAbove && bottom + needed > vh) {
    bottom = Math.max(vh - needed, top + TOUR_MIN_RING);
  }
  /* Whole pixels. The four shades are laid out edge to edge against these numbers,
   * and a fractional boundary leaves a hairline of undimmed page between them that
   * reads as a stray rule running off the highlighted element. */
  const t = Math.round(top), l = Math.round(left);
  return { top: t, left: l,
           width: Math.max(Math.round(right) - l, 8),
           height: Math.max(Math.round(bottom) - t, 8) };
}

const TOUR_PLAY = html`
  <svg class="tour-play" viewBox="0 0 12 12" aria-hidden="true">
    <path d="M3 2.1 10.2 6 3 9.9z" fill="currentColor" />
  </svg>`;

function Tour({ summary, curve, navigate, onClose }) {
  const steps = useMemo(() => tourSteps(summary, curve), [summary, curve]);
  const total = steps.length;
  const [step, setStep] = useState(0);
  const [ring, setRing] = useState(null);
  const [place, setPlace] = useState(null);
  const cardRef = useRef(null);
  const done = step >= total;

  const measure = useCallback(() => {
    const card = cardRef.current;
    if (!card) return;
    const box = { w: card.offsetWidth, h: card.offsetHeight };
    const vw = window.innerWidth, vh = window.innerHeight;
    const st = steps[step];
    const el = st && document.querySelector(st.target);
    if (!el) { setRing(null); setPlace(tourCardPlacement(null, box, vw, vh)); return; }
    const r = tourRing(el.getBoundingClientRect(), box, vw, vh, st.pad === undefined ? 8 : st.pad);
    setRing(r);
    setPlace(tourCardPlacement(r, box, vw, vh));
  }, [step, steps]);

  /* Navigate, then place the ring and the card in the same frame.
   *
   * This used to smooth-scroll and measure on a fixed 520ms timer. Three things went
   * wrong with that. The card kept its old position while showing the new step's text,
   * so the words arrived well before the shape. The ring then appeared more than half a
   * second later. And because a smooth scroll is still running at 520ms, the first
   * measurement was of a moving target, which the scroll listener then corrected —
   * the drift.
   *
   * So: clear both first, so nothing is shown in the wrong place; scroll instantly,
   * which is invisible under the dim; wait only for the element to exist rather than
   * for a guessed duration; and measure once the page is still. The ring and the card
   * then fade in together in one frame. */
  useEffect(() => {
    if (done) {
      /* The last card has no element to ring, but it still has to be placed — without
       * this it kept `visibility: hidden` behind a full-screen dim, which looked like
       * the tour had ended and left the page dark. */
      setRing(null);
      markTourSeen();
      const raf0 = requestAnimationFrame(() => measure());
      return () => cancelAnimationFrame(raf0);
    }
    const st = steps[step];
    setRing(null);
    setPlace(null);
    navigate(st);

    let alive = true, raf = 0, timer = 0, tries = 0;
    const settle = () => {
      if (!alive) return;
      if (st.settle) timer = setTimeout(() => alive && measure(), st.settle);
      else raf = requestAnimationFrame(() => alive && measure());
    };
    const attempt = () => {
      if (!alive) return;
      const el = document.querySelector(st.target);
      if (!el) {
        // the view is still committing; try again next frame rather than guess a delay
        if (tries++ < 30) raf = requestAnimationFrame(attempt);
        else measure();
        return;
      }
      if (getComputedStyle(el).position !== "fixed") {
        const max = Math.max(document.documentElement.scrollHeight - window.innerHeight, 0);
        const y = Math.min(Math.max(el.getBoundingClientRect().top + window.scrollY - TOUR_TOP, 0), max);
        if (Math.abs(y - window.scrollY) > 1) {
          // "instant" and not the two-argument form: the stylesheet sets
          // scroll-behavior: smooth on html, which a bare scrollTo(x, y) obeys. That
          // left the ring measuring a page that was still moving under it.
          window.scrollTo({ top: y, left: 0, behavior: "instant" });
          raf = requestAnimationFrame(settle);
          return;
        }
      }
      settle();
    };
    raf = requestAnimationFrame(attempt);
    return () => { alive = false; cancelAnimationFrame(raf); clearTimeout(timer); };
  }, [step, done]);

  // keep up with the page
  useEffect(() => {
    let queued = false;
    const again = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; measure(); });
    };
    window.addEventListener("scroll", again, { passive: true });
    window.addEventListener("resize", again);
    return () => {
      window.removeEventListener("scroll", again);
      window.removeEventListener("resize", again);
    };
  }, [measure]);

  // While the tour is up, entrance animations elsewhere are suppressed: the ring has
  // to measure a settled element, and waiting for a slide put this step a quarter of a
  // second behind every other one.
  useEffect(() => {
    const root = document.documentElement;
    root.classList.add("tour-running");
    return () => root.classList.remove("tour-running");
  }, []);

  useEffect(() => {
    const k = (e) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") setStep((v) => Math.min(v + 1, total));
      else if (e.key === "ArrowLeft") setStep((v) => Math.max(v - 1, 0));
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose, total]);

  const pos = place || { left: -9999, top: -9999 };

  /* The dim is the ring's own outer shadow, spread far enough to cover any viewport.
   *
   * It used to be four rectangles butted up against the hole. That cannot be made
   * clean: leave a sub-pixel gap and undimmed page shows through as a bright rule
   * running off the element; overlap them instead and the two translucent layers
   * double up into a dark one. Either way a line crosses the screen. One element with
   * one shadow has no seam to get wrong. */
  return html`
    <div class="tour" role="dialog" aria-modal="true" aria-label="Guided tour">
      <div class=${"tour-dim" + (ring ? "" : " solid")}></div>
      ${ring
        ? html`<div class="tour-ring"
                    style=${{ top: ring.top + "px", left: ring.left + "px",
                              width: ring.width + "px", height: ring.height + "px" }}></div>`
        : ""}

      <div class=${"tour-card" + (place ? " in" : "")} ref=${cardRef}
           style=${{ left: pos.left + "px", top: pos.top + "px",
                     visibility: place ? "visible" : "hidden" }}>
        ${done
          ? html`
            <div class="tour-step">Done</div>
            <h3 class="tour-head">That is the whole loop</h3>
            <p class="tour-body">${
              `It closed what it could defend, escalated ${int(summary.escalated)} it could ` +
              `not, and ranked those by the money standing behind them. Every figure on ` +
              `this page came out of the pipeline and was cross-checked before export. ` +
              `The same view runs over a 50,000-row synthetic batch, switchable at top right.`
            }</p>
            <div class="tour-btns">
              <button class="tour-btn" onClick=${() => setStep(0)}>Restart</button>
              <button class="tour-btn primary" onClick=${onClose}>End tour</button>
            </div>`
          : html`
            <div class="tour-step">${`Step ${step + 1} of ${total}`}</div>
            <h3 class="tour-head">${steps[step].head}</h3>
            <p class="tour-body">${steps[step].body}</p>
            <div class="tour-btns">
              <button class="tour-btn ghost" onClick=${onClose}>Skip</button>
              ${step > 0
                ? html`<button class="tour-btn" onClick=${() => setStep(step - 1)}>Back</button>`
                : ""}
              <button class="tour-btn primary" onClick=${() => setStep(step + 1)}>${
                step === total - 1 ? "Finish" : "Next"
              }</button>
            </div>`}
      </div>
    </div>`;
}

/* The wipe that covers a batch switch.
 *
 * There was no page transition anywhere in this build before now — rail clicks scroll
 * and nothing else — so this is a new component rather than a reuse. It exists because
 * changing batch changes every figure on the page at once, and without it that reads
 * as numbers quietly updating rather than as a change of context. The swap happens
 * while the screen is covered, so no panel is ever seen half-updated. */
function Curtain({ hue, phase, label }) {
  return html`
    <div class=${"curtain " + phase} style=${{ background: hue }} aria-hidden="true">
      <div class="curtain-label" style=${{ color: readableOn(hue) }}>${label}</div>
    </div>`;
}

/* ---------------------------------------------------------------- app */

/* Highlights the rail item for whatever is under the top of the viewport.
 *
 * `ready` matters: the sections do not exist in the DOM until the data has loaded and
 * the controller view is mounted, so an effect that runs on first render observes
 * nothing and the rail stays stuck on 01 forever.
 *
 * The section is chosen by geometry, not by which ones intersect an observation band.
 * Adjacent sections share an edge, so at rest after a click the outgoing section still
 * laps a pixel or two into any band that starts at the reading line — enough for
 * `isIntersecting`, which is what made the rail name the section above the one you had
 * just jumped to. Instead: the active section is the LAST one whose top edge has passed
 * the reading line. IntersectionObserver is the trigger for that recount, and a
 * throttled scroll listener catches where a smooth scroll comes to rest.
 *
 * A nav click then has to SUSPEND that spying. A smooth scroll from 01 to 03 physically
 * travels through 02, and every intermediate frame is a legitimate scroll event, so the
 * rail would light 02 on the way past and only settle on 03 at the end — the highlight
 * visibly detours to a neighbour and comes back. So a click pins the rail to its target
 * and the spy stands down until the page arrives there (or the user grabs the scroll
 * themselves, or 1.6s passes and something has clearly gone wrong). */
const BOOT_MS = 1200;        // minimum time the boot screen stays up
const BOOT_FADE_MS = 320;

const CHROME_PX = 96;
const READ_LINE = CHROME_PX + 32;        // just below the fixed chrome
const JUMP_MS = 1600;                    // ceiling on how long a nav jump may pin the rail

/** Last section whose top edge has passed the reading line. `tops` are viewport-relative;
 *  null for a section that is not in the DOM. Above the first section, it stays first.
 *  At the bottom of the document the last section wins outright: if it is shorter than
 *  the viewport its top can never reach the line, and there is nothing below it. */
function activeSectionFrom(ids, tops, line, atBottom) {
  const present = ids.filter((id, i) => tops[i] !== null && tops[i] !== undefined);
  if (atBottom && present.length) return present[present.length - 1];
  let current = ids[0];
  ids.forEach((id, i) => {
    if (tops[i] !== null && tops[i] !== undefined && tops[i] <= line) current = id;
  });
  return current;
}

function useActiveSection(ready) {
  const [active, setActive] = useState("overview");
  const jump = useRef(null);               // { id, t } while a nav jump is in flight

  const go = useCallback((id) => {
    if (typeof document === "undefined") return;
    const el = document.getElementById(id);
    jump.current = { id, t: Date.now() };
    setActive(id);
    if (el) el.scrollIntoView({ block: "start",
      behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }, []);

  useEffect(() => {
    if (!ready || typeof document === "undefined") return;
    const ids = NAV.map(([id]) => id);
    const els = ids.map((id) => document.getElementById(id));
    if (!els.some(Boolean)) return;

    /* Layout position, not painted position. Sections carry a 24px translateY until
     * their reveal fires, and getBoundingClientRect() reports that transform — which
     * pushed a section back below the reading line and left the rail naming the one
     * above it. offsetTop is unaffected by transforms. */
    const layoutTop = (el) => {
      let y = 0;
      for (let n = el; n; n = n.offsetParent) y += n.offsetTop;
      return y - window.scrollY;
    };

    const pick = () => {
      const tops = els.map((el) => (el ? layoutTop(el) : null));
      const doc = document.documentElement;
      const atBottom = window.innerHeight + window.scrollY >= doc.scrollHeight - 2;
      const found = activeSectionFrom(ids, tops, READ_LINE, atBottom);
      if (jump.current) {
        if (found !== jump.current.id && Date.now() - jump.current.t < JUMP_MS) return;
        jump.current = null;               // arrived, or waited long enough
      }
      setActive(found);
    };

    let queued = false;
    const recount = () => {
      if (queued) return;
      queued = true;
      const run = () => { queued = false; pick(); };
      if (typeof requestAnimationFrame === "undefined") run();
      else requestAnimationFrame(run);
    };
    // a hand on the wheel outranks a jump still in flight
    const release = () => { jump.current = null; recount(); };

    let io = null;
    if (typeof IntersectionObserver !== "undefined") {
      io = new IntersectionObserver(recount,
        { rootMargin: `-${READ_LINE}px 0px 0px 0px`, threshold: 0 });
      els.forEach((el) => el && io.observe(el));
    }
    window.addEventListener("scroll", recount, { passive: true });
    window.addEventListener("resize", recount);
    window.addEventListener("wheel", release, { passive: true });
    window.addEventListener("touchstart", release, { passive: true });
    pick();
    return () => {
      if (io) io.disconnect();
      window.removeEventListener("scroll", recount);
      window.removeEventListener("resize", recount);
      window.removeEventListener("wheel", release);
      window.removeEventListener("touchstart", release);
    };
  }, [ready]);
  // `mark` sets the rail without scrolling; the batch switch has already jumped the
  // page itself and only needs the highlight to agree.
  return [active, go, setActive];
}

/* Portrait, to fill the panel beside the copy. preserveAspectRatio="none" keeps the
 * viewBox mapping linear so the callout can be placed on a specific rectangle in
 * percentages of the panel — which also means it can never leave the panel. */
const COVER_W = 900, COVER_H = 1080;

function CoverFig({ value, label, sub, hue }) {
  return html`
    <div class="cfig">
      <div class="cfig-rule" style=${{ background: hue }}></div>
      <div class="cfig-v num">${value}</div>
      <div class="cfig-l">${label}</div>
      <div class="cfig-s">${sub}</div>
    </div>`;
}

/* Where the annotation card goes, in percentages of the panel.
 *
 * It must never cover the rectangle it is annotating, and must not leave the panel.
 * Beside the rectangle is preferred — right, then left. The largest exposures are
 * sometimes wide enough that neither side has room, and then the card goes below it,
 * or above it if the rectangle is near the bottom. Returns the card's corner, the
 * point on the rectangle the leader leaves from, and the point on the card it meets. */
function calloutPlacement(big) {
  if (!big) return null;
  const lx = (big.x / COVER_W) * 100;
  const rx = ((big.x + big.w) / COVER_W) * 100;
  const ty = (big.y / COVER_H) * 100;
  const by = ((big.y + big.h) / COVER_H) * 100;
  const midY = Math.min(Math.max((ty + by) / 2, 11), 86);

  if (rx + 3 + CALLOUT_W <= 98) {
    return { x: rx + 3, y: midY, vertical: false,
             from: [rx, midY], to: [rx + 3, midY] };
  }
  if (lx - 3 - CALLOUT_W >= 2) {
    const x = lx - 3 - CALLOUT_W;
    return { x, y: midY, vertical: false, from: [lx, midY], to: [x + CALLOUT_W, midY] };
  }
  const midX = Math.min(Math.max((lx + rx) / 2 - CALLOUT_W / 2, 2), 98 - CALLOUT_W);
  const stem = Math.min(Math.max((lx + rx) / 2, 3), 97);
  const below = by + 3 + CALLOUT_H <= 98;
  const y = below ? by + 3 : Math.max(ty - 3 - CALLOUT_H, 2);
  return { x: midX, y, vertical: true,
           from: [stem, below ? by : ty], to: [stem, below ? y : y + CALLOUT_H] };
}

const CYCLE_TOP = 10;        // the ten largest exposures
const CYCLE_MS = 4200;       // how long each one is held
const CALLOUT_W = 44;        // callout width as a % of the panel; matches styles.css
const CALLOUT_H = 19;        // its height, generously, as a % of the panel

/* `calm` is set while the guided tour is running. The hero animates 3,133 elements
 * for the first 900ms and then keeps six groups breathing; stepping through the tour
 * against that put the ring nearly half a second behind the click, because both were
 * competing for the same frames. During the tour the map simply sits still. */
function Landing({ summary, queue, onEnter, calm, complete }) {
  const v = summary.value;
  const cells = useMemo(() => treemapCells(queue.queue, COVER_W, COVER_H), [queue]);

  /* Walks the ten largest exposures. Nothing here is decoration: the index selects a
   * real row, and the callout reads that row's own figures. */
  const [hi, setHi] = useState(0);
  const n = Math.min(CYCLE_TOP, cells.length);
  useEffect(() => {
    if (calm || prefersReducedMotion() || n < 2) return;
    let id = 0;
    const start = setTimeout(() => {
      id = setInterval(() => setHi((h) => (h + 1) % n), CYCLE_MS);
    }, DRAW_MS + 1400);                       // let the draw-in finish first
    return () => { clearTimeout(start); clearInterval(id); };
  }, [n, calm]);

  const big = cells[Math.min(hi, cells.length - 1)];
  const row = big && big.item.row;
  const place = useMemo(() => calloutPlacement(big), [big]);

  return html`
    <section class="landing">
      <div class="cover-grid">
        <div class="cover-copy">
          <div class="landing-mark">Ledgerline</div>
          <p class="cover-what">
            A reconciliation controller that decides what it can close and what a human
            has to look at.
          </p>
          <h1 class="display landing-head">
            It closed what it could <span class="acc-tang">defend</span>, and escalated
            the rest.
          </h1>
          <div class="cover-figs">
            <${CoverFig} hue=${PAL.tangerine} value=${int(summary.auto_closed)}
                         label="Closed without review"
                         sub=${pct(summary.auto_close_rate_pct, 2) + " of the batch"} />
            <${CoverFig} hue=${PAL.violet}
                         value=${pct(summary.auto_close_precision_by_rows_pct, 2)}
                         label="Correct when it closed"
                         sub=${pct(summary.auto_close_precision_by_value_pct, 2) + " by value"} />
            <${CoverFig} hue=${PAL.lime} value=${int(summary.escalated)}
                         label="Escalated for review"
                         sub=${moneyShort(v.in_exception_queue) + " " + CURRENCY + " at risk"} />
          </div>
          <p class="cover-sections">
            <span class="cs-lab">Four sections follow</span>
            ${NAV.map(([id, n, label, hue]) => html`
              <span class="cs-item" key=${id}>
                <span class="cs-n">${n}</span>${" " + label}
              </span>`)}
          </p>
          <button class="cta" onClick=${onEnter}>Open the controller</button>
        </div>

        <figure class="cover-map">
          <figcaption class="cover-map-cap">${
            `${int(summary.escalated)} escalated transactions · sized by exposure · ` +
            `coloured by exception class`
          }</figcaption>
          <div class="cover-map-panel">
            ${complete
              ? html`<${Treemap} cells=${cells} W=${COVER_W} H=${COVER_H}
                                 fit="none" floor=${0.42} highlight=${hi} animate=${!calm} />`
              : html`<div class="map-wait dark">${
                  `Loading all ${int(queue.rows)} escalated transactions…`
                }</div>`}
            ${complete && row && place ? html`
              <svg class="cover-leader" viewBox="0 0 100 100" preserveAspectRatio="none"
                   aria-hidden="true">
                <line x1=${place.from[0].toFixed(2)} y1=${place.from[1].toFixed(2)}
                      x2=${place.to[0].toFixed(2)} y2=${place.to[1].toFixed(2)}
                      stroke=${PAL.lime} stroke-width="2" vector-effect="non-scaling-stroke" />
              </svg>
              <div class=${"cover-callout" + (place.vertical ? " v" : "")}
                   style=${{ left: place.x.toFixed(2) + "%", top: place.y.toFixed(2) + "%" }}>
                <div class="cc-body" key=${hi}>
                  <div class="cc-lab">${
                    row.rank === 1 ? "Largest single exposure"
                                   : `Rank ${String(row.rank).padStart(2, "0")} by exposure`
                  }</div>
                  <div class="cc-val num">${money(row.exposure_cents)} ${CURRENCY}</div>
                  <div class="cc-meta mono">${"B_id " + row.b_id}</div>
                  <div class="cc-cls" style=${{ color: inkFor(row.exception_class) }}>${
                    row.exception_class.replace(/_/g, " ")
                  }</div>
                </div>
              </div>` : ""}
          </div>
        </figure>
      </div>
    </section>`;
}

function App() {
  const [view, setView] = useState("landing");
  const [bootPhase, setBootPhase] = useState("in");   // in -> out -> done
  const [tour, setTour] = useState(false);
  const [tourTaken, setTourTaken] = useState(false);
  const bootAt = useRef(0);
  const [wire, setWire] = useState({});               // url -> [received, total]
  const [batch, setBatch] = useState("eval");
  const [selected, setSelected] = useState(null);
  const [synthAvailable, setSynth] = useState(false);

  // Only the first batch is measured; later batches use the inline loading line.
  const onWire = useCallback((url, got, total, done) => {
    setWire((w) => (w[url] && w[url][2] ? w : { ...w, [url]: [got, total, !!done] }));
  }, []);
  const measure = bootPhase === "in" ? onWire : undefined;

  const summary = useJson(`${DATA}/summary_${batch}.json`, measure);
  // the queue index is measured by the boot bar through its own byte counter below
  const queue = useQueue(batch);
  const curve = useJson(`${DATA}/curve_${batch}.json`, measure);

  const loaded = !!(summary.data && queue.data && curve.data);

  /* Bar position: bytes actually received over bytes announced, across the three
   * files. A file's total is unknown until its headers arrive, so the denominator
   * grows early on — that is the real shape of the download, not a smoothed guess. */
  const bootPct = useMemo(() => {
    if (loaded) return 100;
    const rows = Object.values(wire);
    if (!rows.length) return 0;
    let got = 0, total = 0;
    for (const [g, t, done] of rows) { got += done ? t : g; total += t; }
    return total ? Math.min(99, (got / total) * 100) : 0;
  }, [wire, loaded]);

  // Shown once, on the first batch, for ~1.2s; then it fades and never returns.
  useEffect(() => { if (!bootAt.current) bootAt.current = Date.now(); }, []);
  useEffect(() => {
    if (bootPhase !== "in" || !loaded) return;
    const hold = prefersReducedMotion()
      ? 0 : Math.max(0, BOOT_MS - (Date.now() - (bootAt.current || Date.now())));
    const t = setTimeout(() => setBootPhase("out"), hold);
    return () => clearTimeout(t);
  }, [loaded, bootPhase]);
  useEffect(() => {
    if (bootPhase !== "out") return;
    const t = setTimeout(() => setBootPhase("done"), prefersReducedMotion() ? 0 : BOOT_FADE_MS);
    return () => clearTimeout(t);
  }, [bootPhase]);
  const [active, go, markActive] = useActiveSection(view === "controller" && loaded);

  /* The tour drives the page: each step says which view it needs and whether the
   * detail drawer should be open, and this is the only thing that acts on that. */
  const tourNavigate = useCallback((st) => {
    setView(st.view);
    setSelected(st.open && queue.data && queue.data.queue.length
      ? queue.data.queue[0].b_id : null);
  }, [queue.data]);

  const endTour = useCallback(() => {
    setTour(false);
    setTourTaken(true);
    setSelected(null);
    setView("controller");
  }, []);

  useEffect(() => { setTourTaken(tourSeen()); }, []);

  const enter = useCallback(() => {
    setView("controller");
    if (typeof window !== "undefined") window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    let live = true;
    fetch(`${DATA}/summary_synth.json`, { method: "GET" })
      .then((r) => live && setSynth(r.ok))
      .catch(() => live && setSynth(false));
    return () => { live = false; };
  }, []);

  /* Switching batch replaces every figure on the page. The wipe covers the swap so
   * nothing is seen half-updated, and the page returns to the top of the overview so
   * the new batch is read from the beginning rather than from wherever the reader
   * happened to be in the old queue. */
  const [curtain, setCurtain] = useState(null);
  const swapTimers = useRef([]);

  const applyBatch = useCallback((k) => {
    setBatch(k);
    setSelected(null);
    if (typeof window !== "undefined") {
      // "instant" and not scrollTo(0, 0): the stylesheet sets scroll-behavior: smooth
      window.scrollTo({ top: 0, left: 0, behavior: "instant" });
    }
    markActive("overview");
  }, [markActive]);

  const switchBatch = useCallback((k) => {
    if (k === batch) return;
    swapTimers.current.forEach(clearTimeout);
    swapTimers.current = [];
    const hue = hueForBatch(k);
    const label = (BATCHES.find((b) => b.key === k) || {}).label || k;
    if (prefersReducedMotion()) { applyBatch(k); return; }
    setCurtain({ hue, label, phase: "in" });
    swapTimers.current.push(setTimeout(() => {
      applyBatch(k);
      // Hold, covered, until the new batch has landed. Wiping away on a fixed timer
      // showed the loading state for however long the fetch outlasted it.
      setCurtain({ hue, label, phase: "hold" });
    }, CURTAIN_MS));
  }, [batch, applyBatch]);

  useEffect(() => () => swapTimers.current.forEach(clearTimeout), []);

  // Lift once the incoming batch is ready, or after a cap so a stalled fetch cannot
  // leave the screen covered indefinitely.
  useEffect(() => {
    if (!curtain || curtain.phase !== "hold") return;
    const lift = () => {
      setCurtain((c) => (c && c.phase === "hold" ? { ...c, phase: "out" } : c));
      swapTimers.current.push(setTimeout(() => setCurtain(null), CURTAIN_MS));
    };
    const t = setTimeout(lift, loaded ? 80 : 2000);
    return () => clearTimeout(t);
  }, [curtain, loaded]);

  const err = summary.error || queue.error || curve.error;

  /* One tree, several bodies.
   *
   * Every one of these used to be its own early `return`, which meant the wipe and the
   * tour were dropped from the tree the moment the app fell back to a loading state —
   * and a batch switch does exactly that while the new files are in flight, so the
   * curtain vanished mid-switch and came back. Whatever the body is, the overlays sit
   * beside it in a fixed position. */
  // printed under every section title; computed before the branches, so it has to
  // tolerate a summary that has not loaded yet
  const bLine = summary.data ? batchLine(summary.data) : "";

  let body;
  if (err) {
    body = html`
      <div class="wrap errpage">
        <h1 class="errtitle display">Data not loaded</h1>
        <p class="lede errlede">${err}</p>
        <p class="note u-mt16">
          This page reads static JSON from <code>web/data/</code> over HTTP. Opening <code>index.html</code> from the filesystem will not work, because browsers block <code>fetch</code> on <code>file://</code>. From the <code>web/</code> directory run:
        </p>
        <pre class="cmdbox">python -m http.server 8000</pre>
        <p class="note">
          then open <code>http://localhost:8000/</code>. If the data directory is missing, regenerate it with <code>python export.py</code>.
        </p>
      </div>`;
  } else if (bootPhase !== "done") {
    body = html`<${Boot} batch=${batch === "synth" ? "synthetic" : "BenchRec eval"}
                         records=${summary.data ? summary.data.records_processed : 0}
                         pct=${bootPct} out=${bootPhase === "out"} />`;
  } else if (!loaded) {
    // reached on a batch switch too, where the curtain is covering it
    body = html`<div class="wrap"><div class="loading">Loading…</div></div>`;
  } else {
    /* The tour drives the view from step to step, so it has to survive that switch —
     * rendered from two places it would unmount and restart at step one the moment the
     * cover gave way to the controller. */
    body = view === "landing"
    ? html`<${Landing} summary=${summary.data} queue=${queue.data} onEnter=${enter}
                       calm=${tour} complete=${queue.complete} />`
    : html`
      <${Fragment}>
        <${Ticker} summary=${summary.data} queue=${queue.data} />
        <${TopBar} summary=${summary.data} batch=${batch} setBatch=${switchBatch}
                 synthAvailable=${synthAvailable} onHome=${() => setView("landing")} />
        <${Rail} active=${active} onGo=${go} tourTaken=${tourTaken}
               onTour=${() => setTour(true)} />
        <div class="main">
        <${Overview} summary=${summary.data} queue=${queue.data}
                     complete=${queue.complete} batchLine=${bLine} />
        <${Summary} s=${summary.data} batchLine=${bLine} />
        <${Queue} queue=${queue.data.queue} total=${queue.data.rows}
                  complete=${queue.complete} selected=${selected} onSelect=${setSelected}
                  batchLine=${bLine} />
        <${Curve} curve=${curve.data} batchLine=${bLine} />
        <footer>
          <div class="wrap note-sm">
            Static review interface. All figures were produced by the Python pipeline and
            written to <code>web/data/</code> by <code>export.py</code>, which cross-checks
            them against <code>exposure.py</code> before writing. Nothing is recomputed and no
            model is called when this page renders. The clock is the system clock; every other
            value on screen comes from the exported JSON. Amounts are stored as integer cents
            and shown in ${CURRENCY}.
          </div>
        </footer>
        </div>
        ${selected
          ? html`<${Detail} bId=${selected} batch=${batch} onClose=${() => setSelected(null)} />`
          : ""}
      <//>`;
  }

  return html`
    <div>
      ${body}
      ${tour
        ? html`<${Tour} summary=${summary.data} curve=${curve.data}
                        navigate=${tourNavigate} onClose=${endTour} />`
        : ""}
      ${curtain
        ? html`<${Curtain} hue=${curtain.hue} phase=${curtain.phase}
                           label=${curtain.label} />`
        : ""}
    </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
