"use strict";

const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
};

let file = null;
let lastResult = null;
let maxBytes = 15 * 1024 * 1024;

/* ---------- Theme ---------- */
$("theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  store.set("sacviet-theme", root.dataset.theme);
});

/* ---------- Before/after slider ---------- */
function initCompare(box) {
  const range = box.querySelector(".cmp-range");
  const set = (v) => box.style.setProperty("--pos", `${v}%`);
  range.addEventListener("input", () => set(range.value));
  set(range.value);
  box.setPosition = (v) => { range.value = v; set(v); };
}
document.querySelectorAll(".compare").forEach(initCompare);

function sweep(box) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const start = performance.now();
  const tick = (now) => {
    const t = (now - start) / 2600;
    if (t >= 1) { box.setPosition(50); return; }
    box.setPosition(50 + 38 * Math.sin(t * Math.PI * 2) * (1 - t));
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ---------- Status & examples ---------- */
async function loadStatus() {
  const chip = $("status-chip");
  try {
    const s = await (await fetch("/api/status")).json();
    chip.classList.add("ok");
    $("status-text").textContent = `${s.device}${s.sam ? " · SAM bật" : ""}`;
    chip.title = s.model;
    maxBytes = s.max_upload_mb * 1024 * 1024;
    $("max-mb").textContent = s.max_upload_mb;
    const select = $("model");
    s.models.forEach((m) => select.add(new Option(m.label, m.key, false, m.key === s.default_model)));
    if (s.models.length < 2) select.closest("label").hidden = true;
    renderGallery(s.examples);
    loadHero(s.examples);
  } catch (e) {
    chip.classList.add("bad");
    $("status-text").textContent = "Chưa kết nối máy chủ";
  }
}

function renderGallery(examples) {
  const cards = $("cards");
  examples.forEach((ex) => {
    const b = document.createElement("button");
    b.className = "card"; b.type = "button";
    b.innerHTML = `<img loading="lazy" alt="Ảnh tư liệu: ${ex.title}"><div><b></b><span></span></div>`;
    b.querySelector("img").src = ex.url;
    b.querySelector("b").textContent = ex.title;
    b.querySelector("span").textContent = "Ảnh tư liệu";
    b.addEventListener("click", async () => {
      const blob = await (await fetch(ex.url)).blob();
      setFile(new File([blob], `${ex.id}.jpg`, { type: blob.type }));
      $("studio").scrollIntoView({ behavior: "smooth" });
      run();
    });
    cards.append(b);
  });
}

async function loadHero(examples) {
  const ex = examples.find((e) => e.id === "pho_phuong_mien_bac_1973") || examples[0];
  if (!ex) { $("hero-loading").hidden = true; return; }
  try {
    const r = await (await fetch(`/api/example/${ex.id}?mode=modern`)).json();
    $("hero-gray").src = r.gray;
    $("hero-color").src = r.color;
    $("hero-color").onload = () => { $("hero-loading").hidden = true; sweep($("hero-compare")); };
    $("hero-caption").textContent = `${ex.title} · tô màu bằng VietColorNet`;
  } catch (e) {
    $("hero-loading").textContent = "Chưa tải được ảnh mẫu";
  }
}

/* ---------- Upload ---------- */
const drop = $("drop"), input = $("file");
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
input.addEventListener("change", () => input.files[0] && setFile(input.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));

function hint(text, error = false) {
  const h = $("hint");
  h.textContent = text;
  h.classList.toggle("err", error);
}

function setFile(f) {
  if (!/^image\/(jpeg|png|webp)$/.test(f.type)) { hint("Chỉ nhận ảnh JPG, PNG hoặc WebP.", true); return; }
  if (f.size > maxBytes) { hint(`Ảnh lớn hơn ${Math.round(maxBytes / 1048576)} MB, hãy chọn ảnh nhỏ hơn.`, true); return; }
  file = f;
  const preview = $("preview");
  preview.src = URL.createObjectURL(f);
  preview.hidden = false;
  drop.classList.add("has-image");
  drop.querySelector(".drop-text strong").textContent = f.name;
  $("run").disabled = false;
  hint("Sẵn sàng. Bấm “Tô màu ngay”.");
}

$("strength").addEventListener("input", (e) => { $("strength-out").textContent = `${Number(e.target.value).toFixed(2)}×`; });

/* ---------- Colorize ---------- */
let timer = null;
function setBusy(on) {
  $("busy").hidden = !on;
  $("empty").hidden = on || !!lastResult;
  $("out").hidden = on || !lastResult;
  $("run").disabled = on || !file;
  clearInterval(timer);
  if (on) {
    const t0 = Date.now();
    $("busy-time").textContent = "0s";
    timer = setInterval(() => { $("busy-time").textContent = `${Math.floor((Date.now() - t0) / 1000)}s`; }, 500);
  }
}

async function run() {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("mode", document.querySelector("input[name=mode]:checked").value);
  form.append("strength", $("strength").value);
  form.append("model", $("model").value || "");
  setBusy(true);
  try {
    const res = await fetch("/api/colorize", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Máy chủ gặp lỗi khi tô màu.");
    }
    lastResult = await res.json();
    showResult(lastResult);
    hint("Xong. Kéo thanh trượt để so sánh.");
  } catch (e) {
    hint(e.message, true);
  } finally {
    setBusy(false);
  }
}
$("run").addEventListener("click", run);

function showResult(r) {
  $("res-gray").src = r.gray;
  $("res-color").src = r.color;
  $("dl-color").href = r.color;
  const modeName = { natural: "Tự nhiên", modern: "Hiện đại", vivid: "Rực rỡ" }[r.mode];
  $("meta").textContent = `Hoàn tất trong ${r.seconds}s · ${r.width}×${r.height} px · ${modeName} · ${r.strength.toFixed(2)}× · ${r.model}`;
  $("res-color").onload = () => sweep($("result-compare"));
}

/* ---------- Download comparison & lightbox ---------- */
function loadImg(src) {
  return new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = src; });
}
$("dl-compare").addEventListener("click", async () => {
  if (!lastResult) return;
  const [g, c] = await Promise.all([loadImg(lastResult.gray), loadImg(lastResult.color)]);
  const canvas = document.createElement("canvas");
  canvas.width = g.width + c.width; canvas.height = Math.max(g.height, c.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(g, 0, 0); ctx.drawImage(c, g.width, 0);
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png"); a.download = "sac-viet-so-sanh.png"; a.click();
});

const lb = $("lightbox");
$("zoom").addEventListener("click", () => { if (lastResult) { $("lb-img").src = lastResult.color; lb.hidden = false; } });
$("lb-close").addEventListener("click", () => { lb.hidden = true; });
lb.addEventListener("click", (e) => { if (e.target === lb) lb.hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") lb.hidden = true; });

loadStatus();
