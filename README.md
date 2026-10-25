# Handwritten Digit Recognition — PyTorch CNN in the browser

Static site: HTML + CSS + JavaScript. Inference runs in the browser with ONNX Runtime Web.

## Files
- `index.html`, `style.css`, `app.js`: the site
- `digits_cnn.onnx`: the trained CNN (exported from `digits_cnn.pt`); must sit next to `index.html`
- `model-data.js`: same model embedded as base64, a fallback for opening `index.html` by double-click (optional)
- `export_to_onnx.py`: how `digits_cnn.pt` becomes `digits_cnn.onnx`

## Run it
    python -m http.server 8000      # inside this folder
    # open http://localhost:8000

(Double-clicking `index.html` also works thanks to `model-data.js`, but a local server is the proper way.)
Internet is needed once to load ONNX Runtime Web and fonts from a CDN.

## Deploy
Upload the whole folder to GitHub Pages, Netlify, Vercel or Cloudflare Pages. No build step.
