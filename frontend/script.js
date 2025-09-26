const API = location.origin.replace(/\/$/, "");

let token = localStorage.getItem('token') || '';
let user = null;
let currentAppId = null;
let isAdmin = false;
let ws;

function headers() {
  return token ? { 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
}

async function register() {
  const email = document.getElementById('regEmail').value;
  const password = document.getElementById('regPwd').value;
  const admin = document.getElementById('regAdmin').checked;
  const res = await fetch(`${API}/api/auth/register`, { method: 'POST', headers: headers(), body: JSON.stringify({ email, password, is_admin: admin }) });
  const data = await res.json();
  alert('Registered ' + data.email);
}

async function login() {
  const email = document.getElementById('loginEmail').value;
  const password = document.getElementById('loginPwd').value;
  const form = new URLSearchParams();
  form.append('username', email);
  form.append('password', password);
  const res = await fetch(`${API}/api/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: form });
  const data = await res.json();
  token = data.access_token;
  localStorage.setItem('token', token);
  user = { email };
  isAdmin = document.getElementById('regAdmin').checked; // heuristic; server does not return role here
  document.querySelector('.app').style.display = 'block';
  connectWS();
}

async function createApplication() {
  const applicant_name = document.getElementById('applicantName').value;
  const dob = document.getElementById('applicantDob').value;
  const aadhaar_number = document.getElementById('aadhaarNumber').value;
  const res = await fetch(`${API}/api/applications`, { method: 'POST', headers: headers(), body: JSON.stringify({ applicant_name, dob, aadhaar_number }) });
  const app = await res.json();
  currentAppId = app.id;
  document.getElementById('appInfo').textContent = `Application #${app.id} created`;
}

async function uploadDocument() {
  if (!currentAppId) return alert('Create application first');
  const dt = document.getElementById('docType').value;
  const file = document.getElementById('fileInput').files[0];
  const form = new FormData();
  form.append('application_id', currentAppId);
  form.append('doc_type', dt);
  form.append('file', file);
  const res = await fetch(`${API}/api/upload`, { method: 'POST', headers: { 'Authorization': 'Bearer ' + token }, body: form });
  const out = await res.json();
  document.getElementById('uploadResult').textContent = JSON.stringify(out, null, 2);
}

async function startTraining() {
  const dataset_slug = document.getElementById('datasetSlug').value || null;
  const target_column = document.getElementById('targetColumn').value || null;
  const res = await fetch(`${API}/api/training/start`, { method: 'POST', headers: headers(), body: JSON.stringify({ dataset_slug, target_column }) });
  const job = await res.json();
  document.getElementById('trainLogs').textContent += `\nJob ${job.id} started...`;
}

async function predict() {
  if (!currentAppId) return alert('Create application first');
  const res = await fetch(`${API}/api/predict`, { method: 'POST', headers: headers(), body: JSON.stringify({ application_id: currentAppId }) });
  const data = await res.json();
  document.getElementById('predictOut').textContent = JSON.stringify(data, null, 2);
  speak(`Predicted probability of approval is ${Math.round(data.probability*100)} percent.`);
}

function downloadReport() {
  if (!currentAppId) return alert('Create application first');
  fetch(`${API}/api/report/${currentAppId}.pdf`, { headers: { 'Authorization': 'Bearer ' + token }})
    .then(r => r.blob())
    .then(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `application_${currentAppId}_report.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    });
}

function connectWS() {
  if (!token) return;
  ws = new WebSocket(`${location.origin.replace('http', 'ws')}/ws`);
  ws.onopen = () => { ws.send(JSON.stringify({ token })); };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.event === 'training_progress') {
      const pct = Math.min(100, Math.round((msg.progress || 0) * 100));
      document.getElementById('trainBar').style.width = pct + '%';
      document.getElementById('trainLogs').textContent += `\n${msg.model}: ${(msg.metrics && msg.metrics.roc_auc)||0}`;
    } else if (msg.event === 'training_completed') {
      document.getElementById('trainBar').style.width = '100%';
      document.getElementById('trainLogs').textContent += `\nCompleted. Best: ${msg.best_model}`;
    } else if (msg.event === 'document_status') {
      appendChat(`Doc ${msg.doc_id} -> ${msg.status}`);
    }
  };
}

// Chatbot using Web Speech API
const synth = window.speechSynthesis;
let recognizing = false;
let recognition;
if ('webkitSpeechRecognition' in window) {
  recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.lang = 'en-US';
  recognition.onresult = (event) => {
    const text = event.results[0][0].transcript;
    document.getElementById('chatInput').value = text;
    handleChat(text);
  };
}

function speak(text) {
  if (!synth) return;
  const utter = new SpeechSynthesisUtterance(text);
  synth.speak(utter);
}

function appendChat(text) {
  const log = document.getElementById('chatLog');
  const div = document.createElement('div');
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function handleChat(text) {
  const lower = text.toLowerCase();
  if (lower.includes('predict')) {
    predict();
    appendChat('Bot: Running prediction...');
  } else if (lower.includes('train')) {
    startTraining();
    appendChat('Bot: Starting training...');
  } else if (lower.includes('report')) {
    downloadReport();
    appendChat('Bot: Downloading report...');
  } else {
    appendChat('Bot: I can train, predict, upload docs, and get reports.');
    speak('I can train, predict, upload documents and get reports.');
  }
}

document.getElementById('btnRegister').onclick = register;
document.getElementById('btnLogin').onclick = login;
document.getElementById('btnCreateApp').onclick = createApplication;
document.getElementById('btnUpload').onclick = uploadDocument;
document.getElementById('btnStartTrain').onclick = startTraining;
document.getElementById('btnPredict').onclick = predict;
document.getElementById('btnReport').onclick = downloadReport;
document.getElementById('btnSend').onclick = () => handleChat(document.getElementById('chatInput').value);
document.getElementById('btnSpeak').onclick = () => recognition && recognition.start();

