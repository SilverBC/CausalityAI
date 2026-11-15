# The Investigator's mascot

From the rebuild plan's assets. The app draws the current version inline as `GatorIcon` (src/components/icons.jsx); the files here are kept for reference, slides and app icons. The reference images that inspired it are not included (licence unknown).

| File | What it is |
|---|---|
| `gator-current-v5.svg` | **Current mascot.** Line-art alligator head, jaws open, deerstalker tipped ~45° back. Strokes use `currentColor` (white by default); the hat crown is filled with `--accent` so it covers the lines behind it. viewBox `-3 -4 52 52`. In React it is `GatorIcon` (v1 `frontend/src/components/icons.jsx`). |
| `gator-line-art-v4.svg` | v4: same line art, hat sitting flat with its brim forming the top of the head. |
| `gator-head-detective-v3.svg` | v3: solid cartoon head with gaping jaws (after the photo reference) and a deerstalker. |
| `gator-full-body-v2.svg` | v2: whole-alligator silhouette in a small deerstalker. |
| `gator-head-v1.svg` | v1: first open-jawed head, no hat. |

The SVGs are drawn in **white**, for the indigo button, so they look blank on a white background. Set `color` (and
`--accent`) on the `<svg>` or a parent to recolor them.

### mascot/png/

| File | Use |
|---|---|
| `gator-button-light-66px@1x/2x/3x.png` | The chat button exactly as in the app (66 px circle, glow shadow), light mode |
| `gator-button-dark-66px@1x/2x/3x.png` | Same, dark mode (lighter indigo button, dark alligator) |
| `gator-app-icon-light-192.png`, `-512.png` | Square-canvas round icons (PWA manifest, store mock-ups, slides) |
| `gator-app-icon-dark-192.png`, `-512.png` | Same in the dark-mode colors |
| `gator-transparent-white-512.png` | Alligator only, white, transparent background (for colored backgrounds) |
| `gator-transparent-ink-512.png` | Alligator only, dark ink, transparent background (for light backgrounds, docs, slides) |
| `all-versions-preview.png` | All five mascot versions side by side |
