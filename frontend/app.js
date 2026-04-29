/**
 * StructScan — app.js
 * Multi-page: Splash → Landing → Analysis (Camera/Image/Video)
 */

var S = {
  mode: null, ws: null, stream: null, pendingMode: null,
  alertCooldown: false, lastTime: Date.now(), frameCount: 0,
  videoFrames: [], videoPlayTimer: null, videoPlayIdx: 0,
};

function el(id) {
  var e = document.getElementById(id);
  if (!e) console.warn('Missing #' + id);
  return e;
}

// ── Severity config (new scale) ──────────────────────────────────────────────
var SEV_MAP = [
  [0,  20,  'SAFE',      '#16a34a'],
  [20, 30,  'RISK',      '#65a30d'],
  [30, 50,  'HIGH RISK', '#d97706'],
  [50, 75,  'DANGEROUS', '#ea580c'],
  [75, 101, 'CRITICAL',  '#dc2626'],
];

function getSev(score) {
  for (var i = 0; i < SEV_MAP.length; i++) {
    if (score >= SEV_MAP[i][0] && score < SEV_MAP[i][1])
      return { level: SEV_MAP[i][2], color: SEV_MAP[i][3] };
  }
  return { level: 'SAFE', color: '#16a34a' };
}

// ── Factors per crack type ───────────────────────────────────────────────────
var FACTORS = {
  'Hairline Crack': ['Surface shrinkage during curing','Thermal expansion/contraction','Minor vibration or impact','Age-related surface weathering'],
  'Linear Structural Crack': ['Excessive tensile stress','Foundation settlement','Overloading beyond design capacity','Reinforcement corrosion'],
  'Diagonal Shear Crack': ['Shear force exceeding capacity','Inadequate stirrups/ties','Point loading near support','Seismic activity'],
  'Spalling / Delamination': ['Rebar corrosion and rust expansion','Freeze-thaw cycles','Alkali-silica reaction','Inadequate cover depth'],
  'Vertical / Horizontal Crack': ['Differential settlement','Thermal movement','Shrinkage during curing','Lateral earth pressure'],
};

// ── Splash ───────────────────────────────────────────────────────────────────
window.addEventListener('load', function() {
  var fill = el('splashFill');
  if (fill) {
    setTimeout(function() { fill.style.width = '100%'; }, 100);
    setTimeout(showLanding, 3000);
  }
  checkHealth();
});

function showLanding() {
  var splash  = el('splash');
  var landing = el('landing');
  if (splash)  { splash.classList.add('out'); setTimeout(function(){ splash.style.display='none'; }, 700); }
  if (landing) landing.classList.remove('hidden');
}

// ── Health check ─────────────────────────────────────────────────────────────
function checkHealth() {
  fetch('/health').then(function(r){ return r.json(); }).then(function(h) {
    var dot  = el('navDot');
    var txt  = el('navStatusText');
    if (h.demo_mode) {
      if (dot) dot.className = 'busy';
      if (txt) txt.textContent = 'demo mode';
    } else {
      if (dot) dot.className = 'live';
      if (txt) txt.textContent = 'model loaded';
    }
  }).catch(function() {
    var dot = el('navDot');
    var txt = el('navStatusText');
    if (dot) dot.className = 'error';
    if (txt) txt.textContent = 'offline';
  });
}

// ── Open mode ─────────────────────────────────────────────────────────────────
window.openMode = function(mode) {
  el('landing').classList.add('hidden');
  var ap = el('analysisPage');
  ap.classList.remove('hidden');

  var titles = { camera: 'LIVE CAMERA ANALYSIS', image: 'IMAGE ANALYSIS', video: 'VIDEO ANALYSIS' };
  el('analysisTitle').textContent = titles[mode] || mode.toUpperCase();

  S.mode = null; // reset before setting
  setAnalysisBadge('READY', '');

  // Show/hide start button label
  el('btnStart').textContent = mode === 'camera' ? '◉ START CAMERA' :
                               mode === 'image'  ? '▣ UPLOAD IMAGE' : '▶ UPLOAD VIDEO';

  S.pendingMode = mode;
  log('Mode selected: ' + mode.toUpperCase(), 'info');
};

window.goBack = function() {
  hardStop();
  el('analysisPage').classList.add('hidden');
  el('landing').classList.remove('hidden');
};

// ── Start ─────────────────────────────────────────────────────────────────────
window.startAnalysis = function() {
  var mode = S.pendingMode;
  if (!mode) return;
  S.mode = mode;

  if (mode === 'camera') {
    startCamera();
  } else {
    var fi = el('fileInput');
    fi.accept = mode === 'image' ? 'image/*' : 'video/*,video/mp4,video/avi,video/mov';
    fi.value  = '';
    setTimeout(function(){ fi.click(); }, 60);
  }
};

document.addEventListener('DOMContentLoaded', function() {
  var fi2 = el('fileInput');
  if (fi2) fi2.addEventListener('change', function() {
  var f = el('fileInput').files[0];
  if (!f) return;
  if (S.pendingMode === 'image')      doImage(f);
  else if (S.pendingMode === 'video') doVideo(f);
    el('fileInput').value = '';
  });
});

// ── Hard Stop ─────────────────────────────────────────────────────────────────
window.hardStop = function() {
  S.mode = null;
  if (S.videoPlayTimer) { clearInterval(S.videoPlayTimer); S.videoPlayTimer = null; }
  if (S.ws) {
    S.ws.onclose = null; S.ws.onerror = null;
    try { S.ws.close(); } catch(e) {}
    S.ws = null;
  }
  if (S.stream) {
    S.stream.getTracks().forEach(function(t){ t.stop(); });
    S.stream = null;
    el('videoEl').srcObject = null;
  }
  var tl = el('videoTimeline');
  if (tl) { tl.style.display = 'none'; el('tlBar').innerHTML = ''; }
  showPlaceholder('Ready to analyse', 'Press Start to begin');
  el('feedScanLine').classList.remove('on');
  setAnalysisBadge('STOPPED', '');
  log('Analysis stopped');
};

// ── Feed helpers ──────────────────────────────────────────────────────────────
function showPlaceholder(msg, sub) {
  el('feedPlaceholder').style.display = 'flex';
  el('videoEl').style.display   = 'none';
  el('resultImg').style.display = 'none';
  if (msg) el('phMsg').textContent = msg;
  if (sub) el('phSub').textContent = sub;
}
function showVideo() {
  el('feedPlaceholder').style.display = 'none';
  el('videoEl').style.display   = 'block';
  el('resultImg').style.display = 'none';
  el('feedScanLine').classList.add('on');
}
function showResult(b64) {
  el('feedPlaceholder').style.display = 'none';
  el('videoEl').style.display   = 'none';
  var img = el('resultImg');
  img.style.display = 'block';
  img.src = 'data:image/jpeg;base64,' + b64;
  img.onload = function() {
    if (img.naturalWidth) el('metaDim').textContent = img.naturalWidth + '×' + img.naturalHeight;
  };
}
function setAnalysisBadge(text, cls) {
  el('aBadgeText').textContent = text;
  el('aBadgeDot').className = cls || '';
}

// ── Log ───────────────────────────────────────────────────────────────────────
function log(msg, type) {
  var scroll = el('logScroll');
  if (!scroll) return;
  var t   = new Date().toLocaleTimeString('en-GB', { hour12: false });
  var div = document.createElement('div');
  div.className = 'logEntry' + (type ? ' log' + type.charAt(0).toUpperCase() + type.slice(1) : '');
  div.textContent = '[' + t + '] ' + msg;
  scroll.insertBefore(div, scroll.firstChild);
  while (scroll.children.length > 50) scroll.removeChild(scroll.lastChild);
}

// ── Image ─────────────────────────────────────────────────────────────────────
function doImage(file) {
  log('Analysing: ' + file.name, 'info');
  setAnalysisBadge('ANALYSING', 'busy');
  showPlaceholder('Analysing image...', file.name);
  var fd = new FormData(); fd.append('file', file);
  fetch('/analyze/image', { method: 'POST', body: fd })
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.error) { log('Error: ' + d.error, 'danger'); setAnalysisBadge('ERROR','error'); showPlaceholder('Error', d.error); return; }
      updateResults(d);
      showResult(d.annotated_frame);
      setAnalysisBadge('DONE', '');
      log('Done — ' + d.label + ' ' + (d.severity_score||0).toFixed(0) + '%');
    })
    .catch(function(e) {
      log('Error: ' + e.message, 'danger');
      setAnalysisBadge('ERROR','error');
      showPlaceholder('Network error', e.message);
    });
}

// ── Video ─────────────────────────────────────────────────────────────────────
function doVideo(file) {
  var mb = (file.size/1048576).toFixed(1);
  log('Processing: ' + file.name + ' (' + mb + ' MB)', 'info');
  setAnalysisBadge('PROCESSING', 'busy');
  showPlaceholder('Analysing video...', file.name + ' · ' + mb + ' MB · ~1 frame / 2 sec');
  var fd = new FormData(); fd.append('file', file);
  fetch('/analyze/video', { method: 'POST', body: fd })
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.error) { log('Error: ' + d.error, 'danger'); setAnalysisBadge('ERROR','error'); showPlaceholder('Video error', d.error); return; }
      var frames = d.frames || [];
      log('Done — ' + d.video_duration_sec + 's · ' + frames.length + ' frames · ' + d.cracked_frames + ' cracked', 'info');
      if (!frames.length) { setAnalysisBadge('DONE',''); return; }
      var worst = frames.reduce(function(a,b){ return (b.severity_score||0)>(a.severity_score||0)?b:a; });
      updateResults(worst);
      showResult(worst.annotated_frame);
      setAnalysisBadge('DONE', '');
      buildTimeline(frames, d.video_duration_sec);
    })
    .catch(function(e) {
      log('Error: ' + e.message, 'danger');
      setAnalysisBadge('ERROR','error');
      showPlaceholder('Video error', e.message);
    });
}

// ── Timeline ──────────────────────────────────────────────────────────────────
function buildTimeline(frames, dur) {
  var tl  = el('videoTimeline');
  var bar = el('tlBar');
  if (!tl || !bar) return;
  bar.innerHTML = '';
  el('tlTitle').textContent = 'VIDEO TIMELINE — ' + frames.length + ' FRAMES · ' + dur + 's';
  tl.style.display = 'block';

  frames.forEach(function(frame, idx) {
    var sev   = getSev(frame.severity_score || 0);
    var seg   = document.createElement('button');
    seg.className = 'tlSeg';
    seg.style.cssText = 'border-color:' + sev.color + '44;color:' + sev.color;
    seg.innerHTML = frame.timestamp_sec + 's<br>' + (frame.severity_score||0).toFixed(0) + '%';
    seg.title = frame.label + ' · ' + sev.level + ' at ' + frame.timestamp_sec + 's';
    seg.addEventListener('click', function() {
      updateResults(frame);
      showResult(frame.annotated_frame);
      bar.querySelectorAll('.tlSeg').forEach(function(b){ b.classList.remove('active'); });
      seg.classList.add('active');
    });
    bar.appendChild(seg);
  });

  S.videoFrames = frames;

  // Auto-click worst
  var wi = 0;
  frames.forEach(function(f,i){ if((f.severity_score||0)>(frames[wi].severity_score||0)) wi=i; });
  if (bar.children[wi]) bar.children[wi].click();
}

window.toggleAutoPlay = function() {
  var btn = el('tlAutoPlay');
  if (S.videoPlayTimer) {
    clearInterval(S.videoPlayTimer); S.videoPlayTimer = null;
    if (btn) btn.textContent = '▶ AUTO-PLAY';
    return;
  }
  if (btn) btn.textContent = '⏹ STOP';
  S.videoPlayIdx = 0;
  S.videoPlayTimer = setInterval(function() {
    if (S.videoPlayIdx >= S.videoFrames.length) {
      clearInterval(S.videoPlayTimer); S.videoPlayTimer = null;
      if (btn) btn.textContent = '▶ AUTO-PLAY';
      return;
    }
    var frame = S.videoFrames[S.videoPlayIdx];
    updateResults(frame); showResult(frame.annotated_frame);
    var segs = el('tlBar').querySelectorAll('.tlSeg');
    segs.forEach(function(b,i){ b.classList.toggle('active', i===S.videoPlayIdx); });
    S.videoPlayIdx++;
  }, 1500);
};

// ── Camera ────────────────────────────────────────────────────────────────────
function startCamera() {
  log('Requesting camera...', 'info');
  setAnalysisBadge('CONNECTING', 'busy');
  showPlaceholder('Starting camera...', 'Requesting access');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    log('Camera not supported — use Chrome/Edge', 'danger');
    setAnalysisBadge('ERROR','error');
    showPlaceholder('Not supported', 'Use Chrome or Edge browser');
    S.mode = null; return;
  }

  navigator.mediaDevices.getUserMedia({
    video: { width:{ideal:640}, height:{ideal:480}, frameRate:{max:30} }, audio: false
  }).then(function(stream) {
    S.stream = stream;
    var v = el('videoEl');
    v.srcObject = stream;
    v.play().then(function() {
      showVideo();
      log('Camera live — connecting server...', 'info');
      connectWS();
    });
  }).catch(function(e) {
    log('Camera error: ' + e.message, 'danger');
    setAnalysisBadge('ERROR','error');
    showPlaceholder('Camera error', e.message);
    S.mode = null;
  });
}

function connectWS() {
  if (S.mode !== 'camera') return;
  var ws = new WebSocket('ws://' + location.host + '/ws/camera');

  ws.onopen = function() {
    S.ws = ws;
    setAnalysisBadge('LIVE', 'live');
    el('feedScanLine').classList.add('on');
    log('Real-time analysis active', 'info');
    sendFrame(ws);
  };

  ws.onmessage = function(e) {
    if (S.mode !== 'camera') return;
    try {
      var d = JSON.parse(e.data);
      if (!d.error) {
        updateResults(d);
        var img = el('resultImg');
        img.style.display = 'block';
        img.src = 'data:image/jpeg;base64,' + d.annotated_frame;
        el('feedPlaceholder').style.display = 'none';
        el('videoEl').style.display = 'none';
        S.frameCount++;
        var fps = (1000 / Math.max(Date.now() - S.lastTime, 1)).toFixed(1);
        S.lastTime = Date.now();
        el('metaFPS').textContent    = 'FPS: ' + fps;
        // Update live crack indicator
        var score = d.severity_score || 0;
        var liveBar = el('liveBar');
        if (liveBar) {
          liveBar.style.width = score + '%';
          liveBar.style.background = score > 75 ? '#dc2626' :
                                     score > 50 ? '#ea580c' :
                                     score > 30 ? '#d97706' :
                                     score > 20 ? '#65a30d' : '#16a34a';
        }
        el('metaFrames').textContent = 'Frames: ' + S.frameCount;
        el('stFPS').textContent      = fps;
      }
    } catch(err) {}
    if (S.mode === 'camera' && ws.readyState === WebSocket.OPEN) sendFrame(ws);
  };

  ws.onerror = function() { log('WebSocket error', 'danger'); setAnalysisBadge('ERROR','error'); };

  ws.onclose = function() {
    el('feedScanLine').classList.remove('on');
    if (S.mode === 'camera') {
      log('Reconnecting in 2s...', 'warn');
      setAnalysisBadge('RECONNECTING','busy');
      setTimeout(connectWS, 2000);
    }
  };
}

function sendFrame(ws) {
  if (S.mode !== 'camera' || !ws || ws.readyState !== WebSocket.OPEN) return;
  var v = el('videoEl');
  if (!v || !v.videoWidth || v.readyState < 2) { setTimeout(function(){ sendFrame(ws); }, 100); return; }
  var c = el('capCanvas');
  var ctx = c.getContext('2d');
  c.width = 640; c.height = 480;
  ctx.drawImage(v, 0, 0, 640, 480);
  c.toBlob(function(blob) {
    if (!blob || ws.readyState !== WebSocket.OPEN || S.mode !== 'camera') return;
    var reader = new FileReader();
    reader.onloadend = function() {
      if (ws.readyState === WebSocket.OPEN && S.mode === 'camera')
        ws.send(reader.result.split(',')[1]);
    };
    reader.readAsDataURL(blob);
  }, 'image/jpeg', 0.75);  // higher quality = better crack pixel detection
}

// ── Update results panel ──────────────────────────────────────────────────────
function updateResults(d) {
  var score = d.severity_score || 0;
  var label = d.label || '—';
  var ci    = d.crack_info || {};
  var sev   = getSev(score);

  // Severity card
  el('sevCard').style.borderLeftColor = sev.color;
  el('sevLevelText').textContent = sev.level;
  el('sevLevelText').style.color = sev.color;
  el('sevLabelText').textContent = label;
  el('sevPct').innerHTML = score.toFixed(0) + '<small>%</small>';
  el('sevFill').style.width = score + '%';
  el('sevFill').style.background = sev.color;
  el('sevMessage').textContent = d.alert_message || '';

  // Stats
  el('stConf').textContent  = ((d.confidence||0)*100).toFixed(1) + '%';
  el('stZones').textContent = (d.damaged_zones||0) + '/' + (d.total_zones||64);
  el('stType').textContent  = (ci.crack_type && ci.crack_type !== 'None') ? ci.crack_type.split(' ')[0] : '—';

  // Crack block
  var cracked = (label === 'Cracked');
  el('crackBlock').style.display = cracked ? 'block' : 'none';
  if (cracked) {
    el('crackRows').innerHTML = [
      ['Crack Type',       ci.crack_type      || '—'],
      ['Width Class',      ci.width_category  || '—'],
      ['Severity Hint',    ci.severity_hint   || '—'],
      ['Affected Area',    (ci.size_percent||0).toFixed(1) + '% of frame'],
      ['Est. Length',      (ci.length_px||0) + ' px'],
      ['Est. Width',       (ci.width_px||0) + ' px · ' + (ci.est_width_mm||0).toFixed(2) + ' mm'],
      ['Aspect Ratio',     (ci.aspect_ratio||0).toFixed(1) + ':1'],
      ['Orientation',      (ci.angle_deg||0) + '°'],
      ['Crack Regions',    ci.num_regions || 0],
      ['Structural Risk',  ci.is_structural ? '⚠ Yes' : 'No'],
      ['Activity',         ci.activity || '—'],
      ['Risk Level',       sev.level],
    ].map(function(r) {
      return '<div class="crackRow"><span class="crKey">' + r[0] + '</span>' +
             '<span class="crVal">' + r[1] + '</span></div>';
    }).join('');
  }

  // Surface + depth
  var sr = el('surfaceRow');
  if (sr && (d.surface_type || d.depth_estimate)) {
    sr.style.display = 'block';
    var si = el('surfaceInfo');
    if (si) {
      si.innerHTML = [
        ['Surface Type',   d.surface_type    || '—'],
        ['Depth Estimate', d.depth_estimate  || '—'],
        ['Width Category', ci.width_category || '—'],
        ['Est. Width',     (ci.est_width_mm || 0) + ' mm'],
        ['Activity',       ci.activity       || '—'],
        ['Structural?',    ci.is_structural  ? 'YES — engineer needed' : 'No — monitor only'],
      ].map(function(r){
        return '<div class="surfRow"><span class="surfKey">'+r[0]+'</span><span class="surfVal">'+r[1]+'</span></div>';
      }).join('');
    }
  }

  // Factors
  var fb = el('factorsBlock');
  if (cracked && ci.crack_type && FACTORS[ci.crack_type]) {
    fb.style.display = 'block';
    el('factorsList').innerHTML = FACTORS[ci.crack_type].map(function(f) {
      return '<div class="factorItem"><span class="factorDot"></span>' + f + '</div>';
    }).join('');
  } else {
    fb.style.display = 'none';
  }

  // Actions
  var al   = el('actionList');
  var sols = d.solutions || [];
  al.innerHTML = sols.length
    ? sols.map(function(s){ return '<li>' + s + '</li>'; }).join('')
    : '<li class="actionMuted">No issues detected — continue monitoring</li>';

  // Helplines
  var hls = d.helplines || [];
  var hb  = el('helplineBlock');
  hb.style.display = hls.length ? 'block' : 'none';
  if (hls.length) {
    el('helplineList').innerHTML = hls.map(function(h) {
      return '<div class="hlEntry"><div class="hlOrg">' + h.name + '</div>' +
             '<a class="hlNum" href="tel:' + h.number + '">' + h.number + '</a>' +
             '<div class="hlNote">' + h.note + '</div></div>';
    }).join('');
  }

  if (d.play_alert_sound) beep();
  if      (sev.level === 'CRITICAL')  log('🔴 CRITICAL — ' + score.toFixed(0) + '%', 'danger');
  else if (sev.level === 'DANGEROUS') log('🟠 DANGEROUS — ' + score.toFixed(0) + '%', 'warn');
  else if (sev.level === 'HIGH RISK') log('🟡 HIGH RISK — ' + score.toFixed(0) + '%', 'warn');
}

// ── Beep ──────────────────────────────────────────────────────────────────────
var audioCtx = null;
function beep() {
  if (S.alertCooldown) return;
  S.alertCooldown = true;
  setTimeout(function(){ S.alertCooldown = false; }, 8000);
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    [880,620,880,520].forEach(function(hz,i){
      var o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.connect(g); g.connect(audioCtx.destination);
      o.type='square'; o.frequency.value=hz;
      var t = audioCtx.currentTime + i*0.2;
      g.gain.setValueAtTime(0.15,t);
      g.gain.exponentialRampToValueAtTime(0.001,t+0.16);
      o.start(t); o.stop(t+0.18);
    });
  } catch(e){}
}
