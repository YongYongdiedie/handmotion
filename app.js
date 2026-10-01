// Pinned releases: keep the MediaPipe JS and WASM versions identical.
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32';
const TF = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
const DURATION = 1500, STEPS = 20, MAX_GAP = 300;
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
    stream = await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:640},height:{ideal:480}}});
    video.srcObject = stream; await video.play(); await initialize();
    if (document.hidden) throw new Error('화면이 숨겨졌습니다. 돌아온 뒤 카메라를 다시 켜 주세요.');
    lastVideo = -1; ring = []; say('준비 완료. 한 손을 보이고 동작을 기록하세요.'); $('cue').textContent = '한 손을 화면 안에 보여 주세요';
    stream.getVideoTracks()[0].onended = () => stopCamera();
    raf = requestAnimationFrame(loop);
  } catch (e) { stopCamera(); say(e.name === 'NotAllowedError' ? '카메라 권한이 필요합니다. 브라우저 사이트 설정에서 허용한 뒤 다시 켜 주세요.' : `시작 실패: ${e.message}`); }
  finally { loading = false; controls(); }
};
function stopCamera() {
  cancelAnimationFrame(raf); stream?.getTracks().forEach(t => t.stop()); stream = null;
  video.srcObject = null; capture = null; predicting = false; ring = [];
  ctx.clearRect(0,0,canvas.width,canvas.height); $('cue').textContent = '카메라 꺼짐'; clearPrediction(); controls();
}
$('stop').onclick = () => { stopCamera(); say('카메라를 껐습니다. 수집한 좌표는 이 탭에 남아 있습니다.'); };
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) {stopCamera(); say('화면 전환으로 카메라를 껐습니다. 다시 켜 주세요.');} });
window.addEventListener('pagehide', stopCamera);
function abortCapture(message) { capture = null; ring = []; $('cue').textContent = '다시 기록해 주세요'; say(message); controls(); }
for (let c=0;c<2;c++) $('record'+c).onclick = () => {
  predicting = false; clearPrediction(); ring = [];
  capture = {label:c,start:performance.now()+3000,frames:[]}; controls(); say('손을 시작 위치에 놓으세요.');
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
// 20 × 63 displacement values, ordered in time. Subtract the FIRST frame,
// not each frame's wrist: per-frame centering would erase hand translation.
function features(sequence) { return sequence.flatMap(frame => frame.map((v,k) => (v-sequence[0][k])*3)); }
function moving(sequence) {
  const xs=sequence.map(f=>f[0]), ys=sequence.map(f=>f[1]);
  return Math.hypot(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys)) >= 0.12;
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
  if(!points) {
    ring=[]; clearPrediction('손을 찾지 못했습니다.');
    if(capture) abortCapture('손 추적이 끊겨 저장하지 않았습니다. 손 전체가 보이도록 다시 기록하세요.');
    else $('cue').textContent='한 손을 보여 주세요';
    return;
  }
  // Mirror x so the feature coordinates match the mirrored preview.
  const frame = {t:now,v:points.flatMap(p=>[1-p.x,p.y,p.z])};
  if(capture) {
    const frames=capture.frames;
    if((frames.length && now-frames.at(-1).t>MAX_GAP) || (!frames.length && now-capture.start>MAX_GAP)) return abortCapture('처리 속도가 너무 느립니다. 다른 앱을 닫고 다시 시도하세요.');
    frames.push(frame); $('cue').textContent = `기록 중 · ${names[capture.label]}`;
    if(now-capture.start>=DURATION) {
      if(frames.length<8) return abortCapture('좌표가 부족합니다. 다시 시도하세요.');
      const sequence=resample(frames);
      if(!moving(sequence)) return abortCapture('움직임이 너무 작습니다. 화면 너비나 높이의 1/4 정도 움직여 주세요.');
      const label=capture.label; data[label].push(sequence); capture=null;
      $('training').textContent='데이터를 추가했습니다. 목표 횟수를 모은 뒤 재학습하세요.';
      $('cue').textContent='저장 완료'; say(`${names[label]} ${data[label].length}회 저장. 시작 위치로 돌아간 뒤 다시 눌러 주세요.`); controls();
    }
    return;
  }
  $('cue').textContent=predicting?'새 동작을 보여 주세요':'손 인식 중';
  if(!predicting) return;
  if(ring.length && now-ring.at(-1).t>MAX_GAP) {ring=[];clearPrediction('추적 대기 중');}
  ring.push(frame);
  while(ring.length>2 && ring[1].t<now-DURATION) ring.shift();
  if(ring.length<8 || now-ring[0].t<DURATION || now-lastPredict<250) return;
  lastPredict=now;
  const sequence=resample(ring);
  if(!moving(sequence)) return clearPrediction('움직임 대기 중');
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
    if(!training && video.readyState>=2 && video.currentTime!==lastVideo && now-lastDetect>=65) {
      lastVideo=video.currentTime; lastDetect=now;
      processFrame(detector.detectForVideo(video,now).landmarks[0],now);
    }
  } catch(e) {stopCamera();say(`손 추적 오류: ${e.message}. 카메라를 다시 켜 주세요.`);return;}
  raf=requestAnimationFrame(loop);
}
$('train').onclick=async()=>{
  training=true;predicting=false;ring=[];clearPrediction();controls();
  model?.dispose(); model=null; trainedCounts=null; $('progress').value=0; controls(); let candidate,x,y;
  try {
    const rows=data.flatMap((sequences,label)=>sequences.map(s=>({x:features(s),y:label})));
    tf.util.shuffle(rows);
    x=tf.tensor2d(rows.map(r=>r.x)); y=tf.tensor2d(rows.map(r=>r.y===0?[1,0]:[0,1]));
    candidate=tf.sequential();
    candidate.add(tf.layers.dense({inputShape:[STEPS*63],units:12,activation:'relu',kernelRegularizer:tf.regularizers.l2({l2:0.001})}));
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
  finally {x?.dispose();y?.dispose();candidate?.dispose();training=false;ring=[];controls();}
};
$('predict').onclick=()=>{predicting=!predicting;ring=[];clearPrediction(predicting?'1.5초 동안 동작을 보여 주세요.':'예측 중지');controls();};
$('reset').onclick=()=>{data=[[],[]];model?.dispose();model=null;trainedCounts=null;predicting=false;ring=[];$('progress').value=0;$('training').textContent='학습 데이터 대기 중';clearPrediction();say('데이터와 모델을 지웠습니다.');controls();};
controls();
