const info   = document.getElementById('info');
const swatch = document.getElementById('swatch');

const rRange = document.getElementById('r');
const gRange = document.getElementById('g');
const bRange = document.getElementById('b');
const rNum   = document.getElementById('r_num');
const gNum   = document.getElementById('g_num');
const bNum   = document.getElementById('b_num');

const API_BASE = 'https://colornet-production.up.railway.app';

function clamp255(n){ n = Number(n); return Math.max(0, Math.min(255, n|0)); }

function rgbToHex(r,g,b){
  return '#' + [r,g,b].map(v => clamp255(v).toString(16).padStart(2,'0')).join('').toUpperCase();
}

function getRGB() {
  return { r: clamp255(rRange.value), g: clamp255(gRange.value), b: clamp255(bRange.value) };
}
function setNumbers({r,g,b}) {
  rNum.value = clamp255(r);
  gNum.value = clamp255(g);
  bNum.value = clamp255(b);
}

let t = null;
function queryDebounced(rgb) {
  clearTimeout(t);
  t = setTimeout(() => query(rgb), 90);
}

async function query(rgb) {
  const {r,g,b} = rgb || getRGB();
  swatch.style.background = `rgb(${r}, ${g}, ${b})`;
  info.textContent = 'Checking…';

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ r, g, b })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    const label = data["Prediction"];
    const p = data["P (White)"];
    swatch.style.color = (label === 'white') ? 'white' : 'black';
    swatch.textContent = `${label.toUpperCase()} TEXT`;
    info.textContent = `RGB(${r},${g},${b}), HEX ${rgbToHex(r,g,b)} • P(white)=${p}`;
  } catch (e) {
    info.textContent = 'Error contacting server';
    console.error('Predict failed:', e);
  }
}

[rRange, gRange, bRange].forEach((el, idx) => {
  el.addEventListener('input', () => {
    const rgb = getRGB();
    setNumbers(rgb);        
    queryDebounced(rgb);    
  });
  el.addEventListener('change', () => query(getRGB())); 
});

[rNum, gNum, bNum].forEach(el => {
  el.addEventListener('input', () => {
    const r = clamp255(rNum.value);
    const g = clamp255(gNum.value);
    const b = clamp255(bNum.value);
    rRange.value = r; gRange.value = g; bRange.value = b;
    query({r,g,b});
  });
});

(function init(){
  const start = getRGB();
  setNumbers(start);
  query(start);
})();
