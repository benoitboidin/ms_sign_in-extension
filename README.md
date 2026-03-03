# MS Sign-in AutoClick

Small Chrome extension (Manifest V3) that auto-clicks "Sign in" prompts on https://teams.cloud.microsoft/.

Installation
- Open Chrome and go to chrome://extensions/
- Enable "Developer mode" (top-right).
- Click "Load unpacked" and select this folder: modules/ms_sign_in-extension

Behavior
- Injects content_script.js on pages under https://teams.cloud.microsoft/*
- Looks for visible buttons/links whose text matches "Sign in", "Sign in again", or similar and clicks the first match once per prompt.
- Uses MutationObserver + 1s polling fallback.
- Logs actions to the console (open DevTools to view).

Customization
- To add a CSS selector fallback, edit content_script.js and add preferred selectors or tighten TEXT_PATTERNS.
- To change polling interval, modify the setInterval delay in content_script.js.

Notes and safety
- The extension only runs on the configured host pattern to limit accidental clicks on other sites.
- If the extension mis-clicks, remove it from chrome://extensions/ or adjust the matching logic.

Files
- manifest.json
- content_script.js
- icon128.png (optional)