// Pinned releases: keep the MediaPipe JS and WASM versions identical.
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32';
const TF = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
const DURATION = 1500, STEPS = 20, MAX_GAP = 150;
// Allow RAF timestamp jitter, but never process the same decoded frame twice.
const DETECT_INTERVAL = 1000 / 30 - 2;
// Short gaps only: require real observations on both sides, no extrapolation.
const MAX_MISSING_RATIO = 0.2;
let trackingSamples = [], lastStats = 0;
const trackingInfo = document.createElement('p');
trackingInfo.id = 'trackingInfo'; trackingInfo.className = 'note';
trackingInfo.textContent = '분석 FPS — · 한 손 추적 · 목표 최대 30회/초';
document.querySelector('.view').after(trackingInfo);
function resetTrackingStats() { trackingSamples = []; lastStats = 0; }
function updateTrackingStats(now, found) {
  trackingSamples.push({t:now,found});
  while (trackingSamples.length && trackingSamples[0].t < now-1000) trackingSamples.shift();
  if (now-lastStats < 500) return;
  lastStats = now;
  const n = trackingSamples.length;
  const span = n > 1 ? now-trackingSamples[0].t : 0;
  const fps = span ? (n-1)*1000/span : 0;
  const rate = Math.round(trackingSamples.filter(s=>s.found).length/n*100);
  trackingInfo.textContent = `분석 ${fps.toFixed(1)} FPS · 손 검출 ${rate}% · 한 손 추적`;
}
function expiredCapture(now) {
  if (!capture || now < capture.start) return false;
  const last = capture.frames.at(-1)?.t ?? capture.start;
  return now-last > MAX_GAP;
}
let detector, tf, stream, model, loading = false, training = false, predicting = false;
let capture = null, ring = [], lastVideo = -1, lastDetect = 0, lastPredict = 0, raf;
let data = [[], []];
let names = ['좌→우', '위→아래'], target = 10, trainedCounts = null;
function modelSummary() {
  if (!trainedCounts) return '아직 학습한 모델이 없습니다.';
  const counts = names.map((name,c) => `${name} ${trainedCounts[c]}회`).join(' · ');
  const pending = data.some((d,c) => d.length !== trainedCounts[c]);
  return `현재 모델: ${counts}로 학습${pending ? ' · 추가 데이터는 재학습 후 반영됩니다.' : ''}`;
}
function selectTab(mode, scroll = false) {
  for (const key of ['collect', 'predict']) {
    const selected = key === mode;
    $(key+'Panel').hidden = !selected;
    $(key+'Tab').setAttribute('aria-selected', String(selected));
    $(key+'Tab').tabIndex = selected ? 0 : -1;
  }
  if (mode === 'collect') { predicting = false; ring = []; clearPrediction(); controls(); }
  if (scroll) $('workspace').scrollIntoView({behavior:'smooth',block:'start'});
}
for (const key of ['collect', 'predict']) {
  $(key+'Tab').onclick = () => selectTab(key);
  $(key+'Tab').onkeydown = event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key) || capture || training || loading) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'collect' : event.key === 'End' ? 'predict' : key === 'collect' ? 'predict' : 'collect';
    selectTab(next); $(next+'Tab').focus();
  };
}
function changeTarget(value) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1 || n > 100) {
    $('target').value = target; say('목표 횟수는 1~100 사이의 정수로 입력하세요.'); return;
  }
  target = n; $('target').value = n; controls();
}
$('target').onchange = () => changeTarget($('target').value);
for (const n of [5,10,20]) $('goal'+n).onclick = () => changeTarget(n);
for (let c=0;c<2;c++) $('name'+c).onchange = () => {
  const name = $('name'+c).value.trim();
  if (!name || name === names[1-c]) {
    $('name'+c).value = names[c]; say('두 동작에 서로 다른 이름을 입력하세요.'); return;
  }
  names[c] = name; ring = []; clearPrediction(); controls();
};
const say = text => $('status').textContent = text;
function controls() {
  const busy = loading || training || !!capture;
  $('start').disabled = busy || !!stream;
  $('stop').disabled = !stream || training;
  for (let c = 0; c < 2; c++) {
    $('record'+c).disabled = busy || !stream;
    $('count'+c).textContent = data[c].length;
    $('goal'+c).textContent = target;
    $('name'+c).disabled = busy;
    $('label'+c).textContent = names[c];
    $('record'+c).setAttribute('aria-label', `${names[c]} 1회 기록`);
  }
  $('train').disabled = busy || data.some(d => d.length < target);
  $('target').disabled = busy;
  for (const n of [5,10,20]) $('goal'+n).disabled = busy;
  $('collectTab').disabled = busy; $('predictTab').disabled = busy;
  $('trainHelp').textContent = `각 동작 ${target}회 이상 모으면 학습할 수 있습니다. 현재 ${data[0].length}회 / ${data[1].length}회 수집.`;
  $('modelInfo').textContent = $('trainedCounts').textContent = modelSummary();
  $('reset').disabled = busy || (!model && !data[0].length && !data[1].length);
  $('predict').disabled = busy || !model || !stream;
  $('predict').textContent = predicting ? '예측 중지' : '실시간 예측 시작';
}
function clearPrediction(text = '동작 대기 중') {
  $('prediction').textContent = text;
  $('cameraPrediction').hidden = true;
  for (let c=0;c<2;c++) { $('p'+c).textContent = '—'; $('m'+c).value = 0; }
}
function loadTF() {
  return new Promise((resolve,reject) => {
    if (window.tf) return resolve(window.tf);
    const s = document.createElement('script'); s.src = TF; s.crossOrigin = 'anonymous';
    s.onload = () => resolve(window.tf);
    s.onerror = () => { s.remove(); reject(new Error('TensorFlow.js 다운로드 실패. 네트워크를 확인해 주세요.')); };
    document.head.append(s);
  });
}
async function initialize() {
  if (detector && tf) return;
  const [vision, tensorflow] = await Promise.all([import(`${MP}/vision_bundle.mjs`), loadTF()]);
  tf = tensorflow;
  try { await tf.setBackend('webgl'); await tf.ready(); }
  catch { await tf.setBackend('cpu'); await tf.ready(); }
  const files = await vision.FilesetResolver.forVisionTasks(`${MP}/wasm`);
  const options = {baseOptions:{modelAssetPath:MODEL,delegate:'GPU'},runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:0.6,minHandPresenceConfidence:0.6,minTrackingConfidence:0.6};
  try { detector = await vision.HandLandmarker.createFromOptions(files, options); }
  catch { options.baseOptions.delegate = 'CPU'; detector = await vision.HandLandmarker.createFromOptions(files, options); }
}
$('start').onclick = async () => {
  loading = true; controls(); say('카메라와 손 검출 모델을 준비합니다…');
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('HTTPS 주소에서 Safari 또는 Chrome으로 열어 주세요.');
    stream = await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:640},height:{ideal:480},frameRate:{ideal:30,max:30}}});
    video.srcObject = stream; await video.play(); await initialize();
    if (document.hidden) throw new Error('화면이 숨겨졌습니다. 돌아온 뒤 카메라를 다시 켜 주세요.');
    lastVideo = -1; lastDetect = 0; resetTrackingStats(); ring = []; say('준비 완료. 한 손을 보이고 동작을 기록하세요.'); $('cue').textContent = '한 손을 화면 안에 보여 주세요';
    stream.getVideoTracks()[0].onended = () => stopCamera();
    raf = requestAnimationFrame(loop);
  } catch (e) { stopCamera(); say(e.name === 'NotAllowedError' ? '카메라 권한이 필요합니다. 브라우저 사이트 설정에서 허용한 뒤 다시 켜 주세요.' : `시작 실패: ${e.message}`); }
  finally { loading = false; controls(); }
};
function stopCamera() {
  cancelAnimationFrame(raf); stream?.getTracks().forEach(t => t.stop()); stream = null;
  video.srcObject = null; capture = null; predicting = false; ring = [];
  resetTrackingStats(); trackingInfo.textContent = '분석 FPS — · 카메라 꺼짐';
  ctx.clearRect(0,0,canvas.width,canvas.height); $('cue').textContent = '카메라 꺼짐'; clearPrediction(); controls();
}
$('stop').onclick = () => { stopCamera(); say('카메라를 껐습니다. 수집한 좌표는 이 탭에 남아 있습니다.'); };
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) {stopCamera(); say('화면 전환으로 카메라를 껐습니다. 다시 켜 주세요.');} });
window.addEventListener('pagehide', stopCamera);
function abortCapture(message) { capture = null; ring = []; $('cue').textContent = '다시 기록해 주세요'; say(message); controls(); }
for (let c=0;c<2;c++) $('record'+c).onclick = () => {
  predicting = false; clearPrediction(); ring = [];
  capture = {label:c,start:performance.now()+3000,frames:[],attempts:0,missing:0}; controls(); say('정지 표현은 자세를 유지하고, 움직이는 표현은 기록 시간 안에 수행하세요.');
};
// Interpolate by timestamps, rather than treating variable phone FPS as fixed FPS.
function resample(frames) {
  const start = frames[0].t, end = frames.at(-1).t; let j=0;
  return Array.from({length:STEPS},(_,i) => {
    const t = start+(end-start)*i/(STEPS-1);
    while(j < frames.length-2 && frames[j+1].t < t) j++;
    const a=frames[j], b=frames[j+1], u=(t-a.t)/(b.t-a.t || 1);
    return a.v.map((v,k) => v+(b.v[k]-v)*u);
  });
}
// Each time step contains 63 wrist-relative shape coordinates and 2 trajectory values.
// Palm-size normalization reduces distance-to-camera effects; orientation is preserved.
// Keep wrist travel separately so centering the shape does not erase movement.
const FEATURES_PER_STEP = 65;
function features(sequence) {
  const origin = sequence[0];
  return sequence.flatMap(frame => {
    const palm = [5,9,17].map(i => Math.hypot(
      frame[i*3]-frame[0], frame[i*3+1]-frame[1], frame[i*3+2]-frame[2]
    ));
    const scale = Math.max(0.025, palm.reduce((a,b)=>a+b,0)/palm.length);
    const shape = frame.map((v,k) => (v-frame[k%3])/scale*0.5);
    // MediaPipe z is wrist-relative depth, not global camera depth: use x/y travel only.
    return [...shape, (frame[0]-origin[0])*3, (frame[1]-origin[1])*3];
  });
}
function draw(points) {
  if(canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {canvas.width=video.videoWidth;canvas.height=video.videoHeight;}
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!points) return;
  ctx.fillStyle='#54f5cf';
  points.forEach(p=>{ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,4,0,Math.PI*2);ctx.fill();});
}
function processFrame(points, now) {
  draw(points);
  if(capture && now < capture.start) { $('cue').textContent = `준비 ${Math.ceil((capture.start-now)/1000)}`; return; }
  if (capture) capture.attempts++;
  if(!points) {
    clearPrediction('손을 잠시 놓쳤습니다.');
    if(capture) {
      capture.missing++;
      if (expiredCapture(now)) abortCapture('손 추적이 0.15초 이상 끊겨 저장하지 않았습니다. 밝은 곳에서 다시 기록하세요.');
      else $('cue').textContent='손 추적 복구 대기 · 기록 중';
    } else {
      if (ring.length && now-ring.at(-1).t > MAX_GAP) ring=[];
      $('cue').textContent='한 손을 보여 주세요';
    }
    return;
  }
  // Mirror x so the feature coordinates match the mirrored preview.
  const frame = {t:now,v:points.flatMap(p=>[1-p.x,p.y,p.z])};
  if(capture) {
    const frames=capture.frames;
    if((frames.length && now-frames.at(-1).t>MAX_GAP) || (!frames.length && now-capture.start>MAX_GAP)) return abortCapture('처리 속도가 너무 느립니다. 다른 앱을 닫고 다시 시도하세요.');
    frames.push(frame); $('cue').textContent = `기록 중 · ${names[capture.label]}`;
    if(now-capture.start>=DURATION) {
      if(frames.length<12) return abortCapture('실제 관찰한 좌표가 부족합니다. 다른 앱을 닫고 다시 시도하세요.');
      if(capture.missing/capture.attempts > MAX_MISSING_RATIO) return abortCapture('손을 놓친 비율이 20%를 넘어 저장하지 않았습니다. 밝은 곳에서 다시 기록하세요.');
      const sequence=resample(frames);
      const label=capture.label, repaired=capture.missing>0; data[label].push(sequence); capture=null;
      $('training').textContent='데이터를 추가했습니다. 목표 횟수를 모은 뒤 재학습하세요.';
      $('cue').textContent='저장 완료'; say(`${names[label]} ${data[label].length}회 저장${repaired ? ' (짧은 누락 보간)' : ''}. 시작 위치로 돌아간 뒤 다시 눌러 주세요.`); controls();
    }
    return;
  }
  $('cue').textContent=predicting?'한 표현을 1.5초 동안 보여 주세요':'손 인식 중';
  if(!predicting) return;
  if(ring.length && now-ring.at(-1).t>MAX_GAP) {ring=[];clearPrediction('추적 대기 중');}
  ring.push(frame);
  while(ring.length>2 && ring[1].t<now-DURATION) ring.shift();
  if(ring.length<8 || now-ring[0].t<DURATION || now-lastPredict<250) return;
  lastPredict=now;
  const recent = trackingSamples;
  if (recent.length && recent.filter(s=>!s.found).length/recent.length > MAX_MISSING_RATIO) return clearPrediction('손 검출이 불안정합니다. 밝은 곳에서 다시 보여 주세요.');
  const sequence=resample(ring);
  const probs=tf.tidy(()=>Array.from(model.predict(tf.tensor2d([features(sequence)])).dataSync()));
  for(let c=0;c<2;c++) {$('p'+c).textContent=`${(probs[c]*100).toFixed(1)}%`;$('m'+c).value=probs[c];}
  const winner = probs[0]>probs[1] ? 0 : 1;
  $('prediction').textContent = names[winner];
  $('cameraPrediction').textContent = `${names[winner]} · ${(probs[winner]*100).toFixed(1)}%`;
  $('cameraPrediction').hidden = false;
}
function loop(now) {
  if(!stream) return;
  try {
    // Also reject a frozen camera, even if no new frame reaches processFrame.
    if (!training && expiredCapture(now)) abortCapture('손 추적 또는 카메라가 0.15초 이상 끊겨 저장하지 않았습니다. 다시 기록하세요.');
    if (!training && predicting && ring.length && now-ring.at(-1).t > MAX_GAP) {
      ring=[]; clearPrediction('추적 대기 중');
    }
    if(!training && video.readyState>=2 && video.currentTime!==lastVideo && now-lastDetect>=DETECT_INTERVAL) {
      lastVideo=video.currentTime; lastDetect=now;
      const points = detector.detectForVideo(video,now).landmarks[0];
      updateTrackingStats(now, !!points);
      processFrame(points,now);
    }
  } catch(e) {stopCamera();say(`손 추적 오류: ${e.message}. 카메라를 다시 켜 주세요.`);return;}
  raf=requestAnimationFrame(loop);
}
$('train').onclick=async()=>{
  training=true;predicting=false;ring=[];clearPrediction();controls();
  resetTrackingStats(); trackingInfo.textContent = '분석 FPS — · 학습 중 추적 일시 정지';
  model?.dispose(); model=null; trainedCounts=null; $('progress').value=0; controls(); let candidate,x,y;
  try {
    const rows=data.flatMap((sequences,label)=>sequences.map(s=>({x:features(s),y:label})));
    tf.util.shuffle(rows);
    x=tf.tensor2d(rows.map(r=>r.x)); y=tf.tensor2d(rows.map(r=>r.y===0?[1,0]:[0,1]));
    candidate=tf.sequential();
    candidate.add(tf.layers.dense({inputShape:[STEPS*FEATURES_PER_STEP],units:12,activation:'relu',kernelRegularizer:tf.regularizers.l2({l2:0.001})}));
    candidate.add(tf.layers.dense({units:2,activation:'softmax'}));
    candidate.compile({optimizer:tf.train.adam(0.003),loss:'categoricalCrossentropy',metrics:['accuracy']});
    await candidate.fit(x,y,{epochs:60,batchSize:8,shuffle:true,yieldEvery:'batch',callbacks:{onEpochEnd:async(epoch,logs)=>{
      if(!Number.isFinite(logs.loss)) throw new Error('학습 값이 불안정합니다. 데이터를 초기화하고 다시 수집해 주세요.');
      $('progress').value=epoch+1;
      $('training').textContent=`${epoch+1} / 60회 · 학습 데이터 정확도 ${((logs.acc??logs.accuracy??0)*100).toFixed(0)}%`;
      await tf.nextFrame();
    }}});
    model=candidate;candidate=null; trainedCounts=data.map(d=>d.length); selectTab('predict',true); say('학습 완료. 실시간 예측을 켜고 새로운 동작으로 시험하세요.');
    $('training').textContent+=' · 완료 (새 데이터 성능은 별도 확인 필요)';
  } catch(e) {$('training').textContent=`학습 실패: ${e.message}`;}
  finally {x?.dispose();y?.dispose();candidate?.dispose();training=false;ring=[];resetTrackingStats();controls();}
};
$('predict').onclick=()=>{predicting=!predicting;ring=[];clearPrediction(predicting?'1.5초 동안 손 모양을 유지하거나 동작을 보여 주세요.':'예측 중지');controls();};
$('reset').onclick=()=>{data=[[],[]];model?.dispose();model=null;trainedCounts=null;predicting=false;ring=[];$('progress').value=0;$('training').textContent='학습 데이터 대기 중';clearPrediction();say('데이터와 모델을 지웠습니다.');controls();};
controls();

