"""Visual check for the review interface.

Not a test suite — a feedback loop. It serves web/, drives a real browser over the
page, writes screenshots you can look at, and reports layout failures in plain terms
with the element and the number that broke.

    python check.py                 # 1440x900, 1280x800, 1920x1080
    python check.py --width 1366    # one width
    python check.py --keep          # leave the screenshots from previous runs

Screenshots land in screenshots/. Every run clears that directory first unless --keep.

Playwright drives the Chrome already installed on this machine (channel="chrome"), so
no browser is downloaded. If that fails, install one with:  python -m playwright install
"""

import argparse
import functools
import http.server
import os
import shutil
import socket
import socketserver
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
SHOTS = os.path.join(HERE, "screenshots")
WIDTHS = [(1440, 900), (1280, 800), (1920, 1080)]

# Views to capture. (name, how to get there)
SECTIONS = ["overview", "summary", "queue", "curve"]


# --------------------------------------------------------------------------- server
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(directory):
    """Serve `directory` on a free port in a background thread. Returns the port."""
    handler = functools.partial(_Quiet, directory=directory)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    httpd.allow_reuse_address = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


# --------------------------------------------------------------------------- probes
#
# Each probe runs in the page and returns a list of failures. A failure is a dict with
# a one-line `what` and whatever numbers make it actionable.

OVERFLOW_JS = r"""
() => {
  const doc = document.documentElement;
  const over = doc.scrollWidth - window.innerWidth;
  return over > 0
    ? [{ what: `document is ${over}px wider than the window`,
         detail: `scrollWidth ${doc.scrollWidth} vs innerWidth ${window.innerWidth}` }]
    : [];
}
"""

# Elements past the right edge. Anything inside a scroll/clip container is that
# container's business, not the page's, so it is skipped.
PAST_EDGE_JS = r"""
() => {
  const doc = document.documentElement, W = window.innerWidth;
  const label = (el) => {
    const cls = (el.getAttribute('class') || '').trim().split(/\s+/).filter(Boolean)
      .slice(0, 2).map((c) => '.' + c).join('');
    return el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + cls;
  };
  const clipped = (el) => {
    for (let p = el.parentElement; p && p !== doc; p = p.parentElement) {
      const s = getComputedStyle(p);
      if (s.overflowX !== 'visible' || s.overflow !== 'visible') return true;
      if (s.position === 'fixed') return true;
    }
    return false;
  };
  const out = [], seen = new Set();
  for (const el of doc.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width < 1 && r.height < 1) continue;
    if (r.right <= W + 0.5) continue;
    if (clipped(el)) continue;
    let covered = false;
    for (let p = el.parentElement; p; p = p.parentElement) if (seen.has(p)) covered = true;
    seen.add(el);
    if (covered) continue;
    out.push({ what: `${label(el)} extends ${Math.round(r.right - W)}px past the viewport`,
               detail: `left ${Math.round(r.left)}, right ${Math.round(r.right)}, ` +
                       `width ${Math.round(r.width)}` });
  }
  return out;
}
"""

# Text clipped by its own box. Elements that opt into ellipsis and carry a title
# attribute are showing the full value on hover, so they are excluded by name.
CLIPPED_TEXT_JS = r"""
() => {
  const label = (el) => {
    const cls = (el.getAttribute('class') || '').trim().split(/\s+/).filter(Boolean)
      .slice(0, 2).map((c) => '.' + c).join('');
    return el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + cls;
  };
  const out = [];
  for (const el of document.querySelectorAll('*')) {
    if (el.children.length) continue;                 // leaf elements carry the text
    const text = (el.textContent || '').trim();
    if (!text) continue;
    const s = getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden') continue;
    const cut = el.scrollWidth - el.clientWidth;
    if (cut <= 1) continue;
    if (el.hasAttribute('title')) continue;           // full value available on hover
    if (el.closest('.tscroll, .qscroll, .ticker')) continue;   // scrolls, not clipped
    if (el.ownerSVGElement) continue;                 // SVG text has no clipping box:
                                                      // scrollWidth is meaningless there
    out.push({ what: `${label(el)} clips its text by ${cut}px`,
               detail: JSON.stringify(text.slice(0, 44)) +
                       ` needs ${el.scrollWidth}px, has ${el.clientWidth}px` });
  }
  return out;
}
"""

NAV_JS = r"""
() => {
  const out = [];
  const items = [...document.querySelectorAll('.rail a .rlabel')];
  if (items.length !== 4) {
    out.push({ what: `expected 4 nav labels, found ${items.length}`, detail: '' });
    return out;
  }
  for (const el of items) {
    const cut = el.scrollWidth - el.clientWidth;
    const txt = el.textContent.trim();
    if (cut > 0 || getComputedStyle(el).textOverflow === 'ellipsis' && cut > 0) {
      out.push({ what: `nav label "${txt}" is truncated by ${cut}px`,
                 detail: `needs ${el.scrollWidth}px, has ${el.clientWidth}px` });
    }
  }
  return out;
}
"""

# Every queue amount must be fully readable, and the magnitude bar must sit behind it.
AMOUNTS_JS = r"""
() => {
  const out = [];
  const rows = [...document.querySelectorAll('.qrow')];
  if (!rows.length) return [{ what: 'no queue rows are mounted', detail: '' }];
  for (const row of rows) {
    const amt = row.querySelector('.amt');
    const bar = row.querySelector('.bar');
    const cell = row.querySelector('.amtcell');
    if (!amt) { out.push({ what: 'a queue row has no amount', detail: '' }); continue; }
    const txt = amt.textContent.trim();

    const cut = amt.scrollWidth - amt.clientWidth;
    if (cut > 1) out.push({ what: `amount ${txt} is clipped by ${cut}px`,
                            detail: `needs ${amt.scrollWidth}px, has ${amt.clientWidth}px` });

    const a = amt.getBoundingClientRect(), c = cell.getBoundingClientRect();
    if (a.left < c.left - 0.5 || a.right > c.right + 0.5) {
      out.push({ what: `amount ${txt} spills out of its cell`,
                 detail: `text ${Math.round(a.left)}..${Math.round(a.right)}, ` +
                         `cell ${Math.round(c.left)}..${Math.round(c.right)}` });
    }
    if (!bar) continue;

    // The bar is allowed to sit behind the text, but only behind: it must be painted
    // under it and must not tint the glyphs beyond what a background may.
    const zBar = Number(getComputedStyle(bar).zIndex) || 0;
    const zAmt = Number(getComputedStyle(amt).zIndex) || 0;
    if (!(zAmt > zBar)) {
      out.push({ what: `amount ${txt} is not painted above its bar`,
                 detail: `bar z-index ${zBar}, text z-index ${zAmt}` });
    }
    const op = Number(getComputedStyle(amt).opacity);
    if (op < 0.999) out.push({ what: `amount ${txt} is not at full opacity`,
                               detail: `opacity ${op}` });
    const bg = getComputedStyle(bar).backgroundImage;
    if (bg && bg !== 'none' && /rgba?\([^)]*\)\s+0%,\s*rgba?\([^)]*\)\s+0%/.test(bg)) {
      out.push({ what: `bar behind ${txt} has a hard edge`, detail: bg.slice(0, 60) });
    }
  }
  return out;
}
"""

PROBES = [
    ("no horizontal overflow", OVERFLOW_JS),
    ("nothing past the viewport edge", PAST_EDGE_JS),
    ("no text clipped by its own box", CLIPPED_TEXT_JS),
    ("nav labels are not truncated", NAV_JS),
    ("queue amounts fully visible above their bars", AMOUNTS_JS),
]

# Which probes make sense on which view. The rail and the queue only exist inside the
# controller, so running those probes on the cover would report noise.
SKIP = {
    "landing": {"nav labels are not truncated",
                "queue amounts fully visible above their bars"},
    "overview": {"queue amounts fully visible above their bars"},
    "summary": {"queue amounts fully visible above their bars"},
    "curve": {"queue amounts fully visible above their bars"},
}


# --------------------------------------------------------------------------- driver
def shoot(page, name, width, height):
    path = os.path.join(SHOTS, f"{width}x{height}-{name}.png")
    page.screenshot(path=path)
    return os.path.relpath(path, HERE)


def run_probes(page, view, failures, width, height):
    for title, js in PROBES:
        if title in SKIP.get(view, set()):
            continue
        for f in page.evaluate(js):
            failures.append((f"{width}x{height}", view, title, f["what"], f.get("detail", "")))


def check_width(pw, port, width, height, failures, written):
    browser = pw.chromium.launch(channel="chrome", args=["--hide-scrollbars"])
    page = browser.new_page(viewport={"width": width, "height": height},
                            reduced_motion="reduce")   # freeze the ambient loop
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")

    page.wait_for_selector(".cta", timeout=20000)
    page.wait_for_timeout(300)
    written.append(shoot(page, "01-landing", width, height))
    run_probes(page, "landing", failures, width, height)

    page.click(".cta")
    page.wait_for_selector("#curve", timeout=20000)
    page.wait_for_timeout(400)

    for i, sec in enumerate(SECTIONS, start=2):
        page.evaluate(
            """(id) => { const el = document.getElementById(id);
                 window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 120); }""",
            sec)
        page.wait_for_timeout(450)
        written.append(shoot(page, f"{i:02d}-{sec}", width, height))
        run_probes(page, sec, failures, width, height)

    # the detail panel, opened from the top queue row
    page.evaluate(
        """() => { const el = document.getElementById('queue');
             window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 120); }""")
    page.wait_for_timeout(350)
    page.click(".qrow")
    page.wait_for_selector(".drawer .body h3", timeout=20000)
    page.wait_for_timeout(500)
    written.append(shoot(page, "06-detail", width, height))
    run_probes(page, "detail", failures, width, height)

    for e in errors:
        failures.append((f"{width}x{height}", "page", "no script errors", e, ""))
    browser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, action="append",
                    help="check one width (repeatable); height follows the 16:10-ish default")
    ap.add_argument("--keep", action="store_true", help="keep screenshots from earlier runs")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright is not installed.  pip install playwright")

    sizes = [(w, round(w * 0.625 / 10) * 10) for w in args.width] if args.width else WIDTHS

    if os.path.isdir(SHOTS) and not args.keep:
        shutil.rmtree(SHOTS)
    os.makedirs(SHOTS, exist_ok=True)

    httpd, port = serve(WEB)
    print(f"serving web/ on 127.0.0.1:{port}")

    failures, written = [], []
    with sync_playwright() as pw:
        for w, h in sizes:
            print(f"  {w}x{h} ...", flush=True)
            check_width(pw, port, w, h, failures, written)
    httpd.shutdown()

    print()
    print(f"{len(written)} screenshots in {os.path.relpath(SHOTS, HERE)}/")
    for p in written:
        print("   ", p)

    print()
    if not failures:
        print("PASS - every assertion held at every width.")
        return 0

    print(f"FAIL - {len(failures)} problem(s).")
    by_check = {}
    for size, view, title, what, detail in failures:
        by_check.setdefault(title, []).append((size, view, what, detail))
    for title, rows in by_check.items():
        sizes_hit = sorted({r[0] for r in rows})
        print(f"\n  {title}  —  fails at {', '.join(sizes_hit)}")
        shown = 0
        for size, view, what, detail in rows:
            if shown >= 8:
                print(f"      ... and {len(rows) - shown} more")
                break
            print(f"      [{size} {view}] {what}")
            if detail:
                print(f"          {detail}")
            shown += 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
