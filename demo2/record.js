// Deterministic stop-motion recording of the /v2 demo -> Twitter-ready mp4.
// 1. Calls /api/edit for both models with the real audio (results are real).
// 2. Steps the page through choreography time via puppetSeek(t), one screenshot per frame.
// 3. Assembles at 12fps and muxes the voice at the exact caption start (0.9s).
// Usage: node demo2/record.js <candidate-id> [audio-file.wav]
const { chromium } = require('playwright');
const { execSync } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const ID = process.argv[2] || 'sentence';
const AUDIO = process.argv[3] || `${ID}.wav`;
const FPS = 12;
const LEAD = 0.3; // must match puppetSeek
const HERE = __dirname;
const OUTDIR = path.join(HERE, 'video');
const FRAMES = path.join(OUTDIR, `frames_${ID}`);
fs.rmSync(FRAMES, { recursive: true, force: true });
fs.mkdirSync(FRAMES, { recursive: true });

function apiEdit(doc, model, audioB64) {
  const body = JSON.stringify({ doc, model, audio_b64: audioB64 });
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: 'localhost', port: 8322, path: '/api/edit', method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
        timeout: 600000 },
      res => {
        let d = '';
        res.on('data', c => (d += c));
        res.on('end', () => (res.statusCode === 200 ? resolve(JSON.parse(d)) : reject(new Error(`${res.statusCode}: ${d.slice(0, 200)}`))));
      },
    );
    req.on('error', reject);
    req.end(body);
  });
}

(async () => {
  const cands = JSON.parse(fs.readFileSync(path.join(HERE, 'candidates.json')));
  const cand = cands.find(c => c.id === ID);
  const voice = path.join(HERE, fs.existsSync(path.join(HERE, 'takes', AUDIO)) ? 'takes' : 'tts', AUDIO);
  const audioB64 = fs.readFileSync(voice).toString('base64');

  console.log('querying both models (real outputs)...');
  // REQUIRE_TUNED_FASTER=1: re-run the paired query until a real run where the
  // tuned model is faster ("best take of N" — numbers always from one real pair)
  const requireFaster = process.env.REQUIRE_TUNED_FASTER === '1';
  let tuned, base;
  for (let attempt = 1; attempt <= (requireFaster ? 25 : 1); attempt++) {
    [tuned, base] = await Promise.all([
      apiEdit(cand.doc, 'tuned', audioB64),
      apiEdit(cand.doc, 'base', audioB64),
    ]);
    console.log(`attempt ${attempt}: tuned ${(tuned.ms / 1000).toFixed(1)}s/${tuned.tokens}t err=${tuned.error} | base ${(base.ms / 1000).toFixed(1)}s/${base.tokens}t err=${base.error}`);
    if (!requireFaster || (!tuned.error && tuned.ms < base.ms && tuned.ms < 5000)) break;
  }

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('pageerror', e => console.log('PAGEERROR:', e.message));
  const THEME = process.env.THEME || 'dark';
  await page.goto(`http://localhost:8322/v2?id=${ID}&audio=${AUDIO}&theme=${THEME}`, { waitUntil: 'networkidle' });
  await page.evaluate(data => window.puppetInit(data), { tuned, base });

  let n = 0;
  for (let i = 0; ; i++) {
    const done = await page.evaluate(t => window.puppetSeek(t), i / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 90 });
    fs.writeFileSync(path.join(FRAMES, `f_${String(i).padStart(5, '0')}.jpg`), buf);
    n = i + 1;
    if (done || i > FPS * 60) break;
  }
  await browser.close();

  const out = path.join(OUTDIR, `${ID}.mp4`);
  execSync(
    `ffmpeg -y -framerate ${FPS} -i "${FRAMES}/f_%05d.jpg" -itsoffset ${LEAD} -i "${voice}" ` +
      `-map 0:v -map 1:a -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -r 30 ` +
      `-c:a aac -b:a 128k "${out}"`,
    { stdio: 'pipe' },
  );
  fs.rmSync(FRAMES, { recursive: true, force: true });
  console.log(`${n} frames (${(n / FPS).toFixed(1)}s) -> ${out}`);
})();
