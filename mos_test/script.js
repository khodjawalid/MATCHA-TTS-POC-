// ================================
// CONFIG
// ================================
const SUBMIT_URL = "https://script.google.com/macros/s/AKfycbwbdq09yBFdYPBDUf5YB0P-qSRl76yZTVipNTg14Jd7tGGPooD9C15ctsXHqxBIUUqD/exec";

// 🔹 Liste EXACTE de tes fichiers (dans mos_test/audios/)
const AUDIO_FILES = [
  // Température (Axx = 4 audios par question)
  "A01_T1.wav","A01_T2.wav","A01_T3.wav","A01_T4.wav",
  "A02_T1.wav","A02_T2.wav","A02_T3.wav","A02_T4.wav",
  "A03_T1.wav","A03_T2.wav","A03_T3.wav","A03_T4.wav",

  // Steps (Bxx = 3 audios par question)
  "B01_S2.wav","B01_S4.wav","B01_S10.wav",
  "B02_S2.wav","B02_S4.wav","B02_S10.wav",
  "B03_S2.wav","B03_S4.wav","B03_S10.wav"
];

// Optionnel : afficher un texte par item.
// Exemple: TEXT_BY_GROUP["A01"] = 'Texte: "...."';
const TEXT_BY_GROUP = {
  // "A01": 'Texte : "..."',
  // "B01": 'Texte : "..."',
};

// ================================
// UTILS
// ================================
function parseName(name) {
  // A01_T1.wav  -> group=A01, cond=T1
  // B02_S10.wav -> group=B02, cond=S10
  const m = name.match(/^([AB]\d{2})_([TS]\d+)\.wav$/);
  if (!m) return null;
  return { group: m[1], cond: m[2], file: name };
}

function shuffle(arr) {
  // Fisher–Yates
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

function getOrCreateParticipantId() {
  const key = "mos_participant_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;

  const id = "P" + Math.random().toString(16).slice(2, 10);
  localStorage.setItem(key, id);
  return id;
}

// ================================
// BUILD QUESTIONS
// ================================
function buildQuestions() {
  const groups = {}; // group -> [{group,cond,file},...]

  for (const f of AUDIO_FILES) {
    const p = parseName(f);
    if (!p) continue;

    if (!groups[p.group]) groups[p.group] = [];
    groups[p.group].push(p);
  }

  // Transform into question list
  const questions = [];
  for (const group in groups) {
    const kind = group.startsWith("A") ? "A_temperature" : "B_steps";
    const items = groups[group].slice(); // copy

    // randomize order inside each group (position bias)
    shuffle(items);

    questions.push({
      group: group,
      kind: kind,
      text: TEXT_BY_GROUP[group] || "",
      items: items
    });
  }

  // randomize order of questions
  return shuffle(questions);
}

// ================================
// STATE
// ================================
const QUESTIONS = buildQuestions();
let idx = 0;
let results = [];
const participantId = getOrCreateParticipantId();

// ================================
// RENDER
// ================================
function render() {
  const q = QUESTIONS[idx];

  document.getElementById("question").innerHTML = `
    <h3>Question ${idx + 1} / ${QUESTIONS.length}</h3>

    <div class="legend">
      <div><b>Légende des scores (1–5)</b></div>
      <ul>
        <li><b>1</b> — Complètement artificiel</li>
        <li><b>2</b> — Plutôt artificiel</li>
        <li><b>3</b> — Il reste un doute</li>
        <li><b>4</b> — Plutôt naturel</li>
        <li><b>5</b> — Totalement naturel</li>
      </ul>
    </div>

    <p class="notice">
      ⚠️ Les différences entre les extraits audio peuvent être subtiles.
      Veuillez écouter attentivement et faire un effort de comparaison avant de répondre.
      Merci !
    </p>

    ${q.text ? `<p class="textref">${q.text}</p>` : ""}
  `;

  document.getElementById("audios").innerHTML = q.items.map((it, i) => `
    <div class="stim">
      <audio controls preload="none" src="audios/${it.file}"></audio>

      <div class="row">
        <span class="label">Extrait ${i + 1}</span>

        <select id="r${i}">
          <option value="">Score</option>
          <option value="1">1 — Complètement artificiel</option>
          <option value="2">2 — Plutôt artificiel</option>
          <option value="3">3 — Il reste un doute</option>
          <option value="4">4 — Plutôt naturel</option>
          <option value="5">5 — Totalement naturel</option>
        </select>
      </div>

      <input type="hidden" id="f${i}" value="${it.file}">
    </div>
  `).join("");
}

// ================================
// NEXT / SUBMIT
// ================================
function next() {
  const q = QUESTIONS[idx];
  const answers = [];

  for (let i = 0; i < q.items.length; i++) {
    const v = document.getElementById(`r${i}`).value;
    if (!v) {
      alert("Merci de noter tous les extraits avant de continuer.");
      return;
    }
    answers.push({
      file: document.getElementById(`f${i}`).value,
      score: Number(v)
    });
  }

  // We store group + kind so Apps Script can compute stats
  results.push({
    group: q.group,
    kind: q.kind,
    answers: answers
  });

  idx++;

  if (idx < QUESTIONS.length) {
    render();
    return;
  }

  // END: auto-submit
  const payload = {
    participant_id: participantId,
    results: results
  };

  document.body.innerHTML = `
    <h2>Merci !</h2>
    <p>Envoi automatique de vos réponses…</p>
    <p id="status">⏳ En cours…</p>
  `;

  fetch(SUBMIT_URL, {
    method: "POST",
    // IMPORTANT: text/plain évite le préflight CORS
    headers: { "Content-Type": "text/plain;charset=utf-8" },
    body: JSON.stringify(payload)
  })

    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data && data.ok) {
        document.getElementById("status").innerHTML =
          "✅ Réponses envoyées avec succès. Merci pour votre participation !";
      } else {
        document.getElementById("status").innerHTML =
          "Échec de l'envoi automatique. Merci de prévenir l'organisateur.";
      }
    })
    .catch(function () {
      document.getElementById("status").innerHTML =
        "Problème réseau : réponses non envoyées. Merci de prévenir l'organisateur.";
    });
}

// Expose next() globally for the button onclick
window.next = next;

// Start
render();
