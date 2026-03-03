(function () {
  const TEXT_PATTERNS = [/sign in again/i, /\bsign in\b/i, /sign in to/i];
  const HANDLED_ATTR = 'data-ms-autoclick-handled';
  let lastClickTs = 0;
  const MIN_CLICK_INTERVAL_MS = 2000;

  function isVisible(el) {
    if (!el) return false;
    if (el.offsetParent === null && el.getClientRects().length === 0) return false;
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity || '1') === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  function matchesSignInText(text) {
    if (!text) return false;
    return TEXT_PATTERNS.some((rx) => rx.test(text));
  }

  function candidateText(el) {
    return (el.innerText || el.textContent || el.getAttribute && el.getAttribute('aria-label') || '').trim();
  }

  function tryClickElement(el) {
    try {
      el.click();
      el.setAttribute(HANDLED_ATTR, '1');
      lastClickTs = Date.now();
      console.log('ms_sign_in-extension: auto-clicked element ->', candidateText(el));
      return true;
    } catch (err) {
      console.warn('ms_sign_in-extension: click failed', err);
      return false;
    }
  }

  function findAndClickOnce() {
    const selectors = 'button, input[type="button"], input[type="submit"], a[role="button"], [role="button"], a';
    const nodes = Array.from(document.querySelectorAll(selectors));
    for (const el of nodes) {
      if (el.getAttribute && el.getAttribute(HANDLED_ATTR)) continue;
      const text = candidateText(el);
      if (!matchesSignInText(text)) continue;
      if (!isVisible(el)) continue;
      const now = Date.now();
      if (now - lastClickTs < MIN_CLICK_INTERVAL_MS) return false;
      const clicked = tryClickElement(el);
      if (clicked) return true;
    }
    return false;
  }

  // Observe DOM changes and poll as fallback
  const observer = new MutationObserver(() => {
    findAndClickOnce();
  });

  let pollId = null;
  function start() {
    const root = document.body || document.documentElement;
    if (root) {
      observer.observe(root, { childList: true, subtree: true });
    }
    // Initial attempt
    findAndClickOnce();
    // Fallback polling (once per second)
    pollId = setInterval(() => {
      findAndClickOnce();
    }, 1000);
    console.log('ms_sign_in-extension: started');
  }

  function stop() {
    try { observer.disconnect(); } catch (e) {}
    if (pollId) clearInterval(pollId);
    console.log('ms_sign_in-extension: stopped');
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }

  window.addEventListener('beforeunload', stop);
})();