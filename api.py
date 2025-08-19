# file: api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from router import predict  # returns {"emotion": {...}, "intent": {...}}

app = FastAPI(title="Emotion + Intent Detector")

class Pred(BaseModel):
    label: str
    confidence: float
    source: str

class Out(BaseModel):
    emotion: Pred
    intent: Pred

class In(BaseModel):
    text: str

@app.post("/predict", response_model=Out)
def _predict(inp: In):
    return predict(inp.text)

# ---------- Minimal UI ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Emotion + Intent Detector</title>
<style>
  :root{color-scheme:light dark}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:2rem;display:grid;place-items:start;background:#0b0b0c;color:#e6e6e6}
  main{max-width:720px;width:100%}
  h1{margin:0 0 1rem 0;font-size:1.5rem}
  form{display:flex;gap:.75rem;align-items:flex-start}
  textarea{flex:1;min-height:110px;padding:.75rem;border-radius:.5rem;border:1px solid #3a3a3a;background:#151518;color:inherit}
  button{padding:.7rem 1rem;border-radius:.5rem;border:0;background:#4f46e5;color:#fff;cursor:pointer}
  button:disabled{opacity:.6;cursor:wait}
  .card{margin-top:1rem;padding:1rem;border:1px solid #2a2a2a;border-radius:.75rem;background:#121317}
  .row{display:flex;gap:.5rem;flex-wrap:wrap}
  .pill{padding:.25rem .6rem;border-radius:999px;background:#1e2a3d;border:1px solid #2b3a52;font-size:.9rem}
  pre{margin:.75rem 0 0 0;white-space:pre-wrap;word-break:break-word}
  .meter{height:10px;border-radius:6px;background:#23242a;overflow:hidden}
  .meter > div{height:100%;background:#22c55e;width:0%}
  small{opacity:.7}
</style>
</head>
<body>
<main>
  <h1>Emotion + Intent Detector</h1>
  <form id="f">
    <textarea id="text" placeholder="Type a sentence… e.g., ‘I can’t stop smiling today!!’"></textarea>
    <button id="go" type="submit">Analyze</button>
  </form>

  <div class="card" id="result" style="display:none">
    <div class="row">
      <div class="pill" id="emotionLabel">emotion: —</div>
      <div class="pill" id="intentLabel">intent: —</div>
      <div class="pill" id="srcEm">src(em): —</div>
      <div class="pill" id="srcIn">src(in): —</div>
    </div>

    <div style="margin:.75rem 0 .25rem 0"><small>emotion confidence</small></div>
    <div class="meter"><div id="barEm"></div></div>

    <div style="margin:.75rem 0 .25rem 0"><small>intent confidence</small></div>
    <div class="meter"><div id="barIn"></div></div>

    <pre id="json"></pre>
  </div>
</main>

<script>
const form = document.getElementById('f');
const btn  = document.getElementById('go');
const ta   = document.getElementById('text');

const card = document.getElementById('result');
const emotionLabel = document.getElementById('emotionLabel');
const intentLabel  = document.getElementById('intentLabel');
const srcEm        = document.getElementById('srcEm');
const srcIn        = document.getElementById('srcIn');
const barEm        = document.getElementById('barEm');
const barIn        = document.getElementById('barIn');
const pre          = document.getElementById('json');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = ta.value.trim();
  if (!text) { alert('Please enter some text.'); return; }

  btn.disabled = true;
  card.style.display = 'block';
  emotionLabel.textContent = 'emotion: —';
  intentLabel.textContent = 'intent: —';
  srcEm.textContent = 'src(em): —';
  srcIn.textContent = 'src(in): —';
  barEm.style.width = '0%';
  barIn.style.width = '0%';
  pre.textContent = 'Analyzing…';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) {
      pre.textContent = 'Error: ' + res.status + ' ' + res.statusText;
      return;
    }
    const data = await res.json();

    emotionLabel.textContent = 'emotion: ' + (data.emotion?.label ?? '—');
    intentLabel.textContent  = 'intent: '  + (data.intent?.label  ?? '—');

    const emoConf = Math.round(((data.emotion?.confidence ?? 0) * 100));
    const intConf = Math.round(((data.intent?.confidence  ?? 0) * 100));
    barEm.style.width = emoConf + '%';
    barIn.style.width = intConf + '%';

    srcEm.textContent = 'src(em): ' + (data.emotion?.source ?? '—');
    srcIn.textContent = 'src(in): ' + (data.intent?.source  ?? '—');

    pre.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    pre.textContent = 'Network error: ' + err.message;
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""
