/* ============================================================
   观相 · 智能面相分析
   基于 @vladmandic/face-api 人脸关键点检测，测量三停比例与
   五官形态，生成传统相学视角的解读报告（仅供文化娱乐参考）。
   ============================================================ */
(function () {
  'use strict';

  const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const previewBox = document.getElementById('previewBox');
  const canvas = document.getElementById('previewCanvas');
  const clearBtn = document.getElementById('clearBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const statusEl = document.getElementById('status');
  const reportEl = document.getElementById('report');

  let currentImage = null;      // 已载入的 HTMLImageElement
  let modelsLoaded = false;
  let loadingModels = null;

  /* ---------- 工具 ---------- */
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function round(v) { return Math.round(v); }
  function inRange(v, lo, hi) { return v >= lo && v <= hi; }

  function setStatus(msg, cls) {
    statusEl.textContent = msg;
    statusEl.className = 'status' + (cls ? ' ' + cls : '');
  }
  function setStatusLoading(msg) {
    statusEl.innerHTML = '<span class="spinner"></span>' + msg;
    statusEl.className = 'status';
  }

  /* ---------- 模型加载 ---------- */
  async function ensureModels() {
    if (modelsLoaded) return;
    if (loadingModels) return loadingModels;
    loadingModels = (async () => {
      if (typeof faceapi === 'undefined') {
        throw new Error('人脸识别库加载失败，请检查网络后刷新页面');
      }
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      modelsLoaded = true;
    })();
    try { await loadingModels; }
    finally { loadingModels = null; }
  }

  /* ---------- 文件读取 ---------- */
  function handleFile(file) {
    if (!file) return;
    if (!file.type || file.type.indexOf('image/') !== 0) {
      setStatus('请选择图片文件（JPG / PNG / WebP）', 'err');
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = function () {
      currentImage = img;
      previewBox.style.display = 'block';
      drawImageOnly();
      setStatus('已载入图片，点击「开始面相分析」', 'ok');
      analyzeBtn.disabled = false;
    };
    img.onerror = function () {
      setStatus('图片读取失败，请换一张试试', 'err');
    };
    img.src = url;
  }

  function clearImage() {
    currentImage = null;
    previewBox.style.display = 'none';
    analyzeBtn.disabled = true;
    if (fileInput) fileInput.value = '';
    reportEl.classList.remove('show');
    reportEl.innerHTML = '<h3>② 分析报告</h3><p style="color: var(--ink-soft);">上传照片并点击「开始面相分析」后，此处将显示解读结果。</p>';
    setStatus('');
  }

  /* ---------- 绘制 ---------- */
  function drawImageOnly() {
    if (!currentImage) return;
    const scale = Math.min(1, 620 / currentImage.naturalWidth);
    canvas.width = Math.round(currentImage.naturalWidth * scale);
    canvas.height = Math.round(currentImage.naturalHeight * scale);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
  }

  function drawOverlay(result) {
    const scale = canvas.width / currentImage.naturalWidth;
    const ctx = canvas.getContext('2d');
    const pts = result.landmarks.positions;
    const box = result.detection.box;

    // 人脸框
    ctx.strokeStyle = 'rgba(194,154,75,.95)';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x * scale, box.y * scale, box.width * scale, box.height * scale);

    // 68 个关键点
    ctx.fillStyle = 'rgba(166,58,46,.9)';
    pts.forEach(function (p) {
      ctx.beginPath();
      ctx.arc(p.x * scale, p.y * scale, 1.6, 0, Math.PI * 2);
      ctx.fill();
    });

    // 三停分界线：眉线（17-26 平均 y）、鼻尖线（点 33）
    let browY = 0;
    for (let i = 17; i <= 26; i++) browY += pts[i].y;
    browY /= 10;
    const noseTipY = pts[33].y;

    ctx.strokeStyle = 'rgba(166,58,46,.85)';
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 1.5;
    [browY, noseTipY].forEach(function (y) {
      ctx.beginPath();
      ctx.moveTo(box.x * scale, y * scale);
      ctx.lineTo((box.x + box.width) * scale, y * scale);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // 上/中/下停标签
    ctx.fillStyle = 'rgba(166,58,46,1)';
    ctx.font = 'bold 12px sans-serif';
    const labelX = (box.x + box.width + 4) * scale;
    ctx.fillText('上停', labelX, (box.y + 8) * scale);
    ctx.fillText('中停', labelX, (browY + 10) * scale);
    ctx.fillText('下停', labelX, (noseTipY + 12) * scale);
  }

  /* ---------- 指标计算 ---------- */
  function computeMetrics(result) {
    const pts = result.landmarks.positions;
    const box = result.detection.box;
    const P = function (i) { return pts[i]; };
    const dist = function (i, j) { return Math.hypot(pts[j].x - pts[i].x, pts[j].y - pts[i].y); };
    const midY = function (arr) { let s = 0; arr.forEach(function (i) { s += pts[i].y; }); return s / arr.length; };
    const midX = function (arr) { let s = 0; arr.forEach(function (i) { s += pts[i].x; }); return s / arr.length; };

    // 面部关键线
    const browY = midY([17, 18, 19, 20, 21, 22, 23, 24, 25, 26]);
    const eyeY = midY([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]);
    const noseTipY = P(33).y;
    const chinY = P(8).y;

    // 发际线近似：优先用人脸框顶部；框未含额头时按「上停≈中停」估算
    let hairlineY = box.y;
    if (hairlineY >= browY - 6) {
      hairlineY = browY - (noseTipY - browY) * 1.0;
    }

    const upper = Math.max(browY - hairlineY, 1);
    const middle = Math.max(noseTipY - browY, 1);
    const lower = Math.max(chinY - noseTipY, 1);
    const total = upper + middle + lower;

    // 宽度
    const fw = Math.max(box.width, 1);
    const fh = Math.max(chinY - hairlineY, 1);
    const ratio = fw / fh;
    const foreheadW = dist(0, 16);
    const cheekW = dist(2, 14);
    const jawW = dist(4, 12);
    const chinW = dist(6, 10);

    // 脸型
    let shape;
    if (chinW / foreheadW < 0.55) shape = '心形脸（瓜子脸）';
    else if (ratio >= 0.88) {
      const sq = Math.abs(foreheadW - jawW) / fw < 0.09 && Math.abs(cheekW - jawW) / fw < 0.09;
      shape = sq ? '方脸（国字脸）' : '圆脸';
    }
    else if (ratio >= 0.80) shape = '鹅蛋脸（椭圆脸）';
    else shape = '长脸';

    // 眉
    const browLength = (dist(17, 21) + dist(22, 26)) / 2;
    const browGap = dist(21, 22);
    const browSlope = ((P(21).y - P(17).y) + (P(22).y - P(26).y)) / 2; // 正=眉尾下垂，负=上扬

    // 眼
    const eyeWidth = (dist(36, 39) + dist(42, 45)) / 2;
    const eyeOpening = (dist(37, 41) + dist(43, 47)) / 2;
    const eyeOpenRatio = eyeOpening / Math.max(eyeWidth, 1);
    const eyeGap = dist(39, 42);
    const eyeGapRatio = eyeGap / Math.max(eyeWidth, 1);
    const eyeSpan = dist(36, 45);

    // 鼻
    const noseLength = dist(27, 33);
    const noseWidth = dist(31, 35);

    // 口
    const mouthWidth = dist(48, 54);
    const mouthHeight = dist(51, 57);
    const mouthRatio = mouthHeight / Math.max(mouthWidth, 1);
    const cornerLift = P(51).y - (P(48).y + P(54).y) / 2; // 正=嘴角上扬

    // 人中
    const philtrum = dist(33, 51);

    // 对称度（镜像配对误差）
    const faceLeft = Math.min.apply(null, pts.slice(0, 17).map(function (p) { return p.x; }));
    const faceRight = Math.max.apply(null, pts.slice(0, 17).map(function (p) { return p.x; }));
    const cx = (faceLeft + faceRight) / 2;
    const pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
      [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
      [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
      [31, 35], [32, 34], [48, 54], [49, 53], [50, 52], [59, 55], [58, 56], [60, 64], [61, 63], [67, 65]];
    let asym = 0;
    pairs.forEach(function (pair) {
      const L = P(pair[0]), R = P(pair[1]);
      asym += Math.abs((L.x + R.x) / 2 - cx) + Math.abs(L.y - R.y);
    });
    asym = asym / pairs.length;
    const symScore = clamp(round(100 - (asym / fw) * 130), 35, 100);

    // 评分
    const parts = [upper, middle, lower];
    const partMean = total / 3;
    const partDev = parts.reduce(function (s, v) { return s + Math.abs(v - partMean); }, 0) / 3 / partMean;
    const scoreBalance = clamp(round(98 - partDev * 320), 30, 98);

    let good = 0;
    if (inRange(eyeOpenRatio, 0.18, 0.45)) good++;
    if (inRange(mouthRatio, 0.16, 0.5)) good++;
    if (inRange(noseWidth / fw, 0.16, 0.30)) good++;
    if (inRange(browLength / fw, 0.28, 0.48)) good++;
    const scoreFeatures = round(52 + good * 11);

    const scoreYintang = clamp(round(((eyeGapRatio - 0.6) / 0.75) * 100), 35, 98);
    const scoreCaibo = clamp(round(55 + (noseWidth / fw - 0.16) * 260), 40, 96);
    const scoreDige = clamp(round(42 + (chinW / fw) * 120), 40, 96);

    const overall = round((scoreBalance + scoreFeatures + symScore + scoreYintang + scoreCaibo + scoreDige) / 6);

    return {
      fw: fw, fh: fh, ratio: ratio,
      upper: upper, middle: middle, lower: lower,
      upperR: upper / total, middleR: middle / total, lowerR: lower / total,
      foreheadW: foreheadW, cheekW: cheekW, jawW: jawW, chinW: chinW,
      shape: shape, browY: browY, eyeY: eyeY, noseTipY: noseTipY,
      browLength: browLength, browGap: browGap, browSlope: browSlope,
      eyeWidth: eyeWidth, eyeOpening: eyeOpening, eyeOpenRatio: eyeOpenRatio,
      eyeGap: eyeGap, eyeGapRatio: eyeGapRatio, eyeSpan: eyeSpan,
      noseLength: noseLength, noseWidth: noseWidth,
      mouthWidth: mouthWidth, mouthHeight: mouthHeight, mouthRatio: mouthRatio,
      cornerLift: cornerLift, philtrum: philtrum,
      symScore: symScore,
      scores: { balance: scoreBalance, features: scoreFeatures, symmetry: symScore, yintang: scoreYintang, caibo: scoreCaibo, dige: scoreDige },
      overall: overall
    };
  }

  /* ---------- 文案生成 ---------- */
  function shapeText(m) {
    switch (m.shape) {
      case '圆脸': return ['圆脸', '面相圆润，传统主亲和随和、人缘佳、重感情，福厚而稳健。'];
      case '方脸（国字脸）': return ['方脸', '下颚方正，传统主坚毅果敢、有担当、务实可靠，有威望之相。'];
      case '长脸': return ['长脸', '面型修长，传统主深沉内敛、思虑周密、重理想，善于谋略。'];
      case '心形脸（瓜子脸）': return ['心形脸', '额头开阔、下巴收尖，传统主聪慧灵动、才思敏捷、异性缘佳。'];
      case '鹅蛋脸（椭圆脸）': return ['鹅蛋脸', '轮廓柔和中正，传统为「中正端庄」之相，适应力与亲和力兼备。'];
      default: return [m.shape, '轮廓独具特点，传统以均衡为贵。'];
    }
  }

  function threePartsText(m) {
    const r = [m.upperR, m.middleR, m.lowerR];
    const mn = (m.upperR + m.middleR + m.lowerR) / 3;
    const balanced = r.every(function (v) { return Math.abs(v - mn) < 0.06; });
    const dominant = r.indexOf(Math.max.apply(null, r));
    const weakest = r.indexOf(Math.min.apply(null, r));
    const domName = ['上停（额头）', '中停（眉眼鼻）', '下停（人中下巴）'][dominant];
    const weakName = ['上停（额头）', '中停（眉眼鼻）', '下停（人中下巴）'][weakest];
    if (balanced) {
      return '三停匀称，比例均衡。传统认为此相一生运势平稳，早年、中年、晚年各有所成，属「中正」格局。';
    }
    return domName + '较为突出，' + weakName + '相对稍弱。传统认为前半生（上停主 15–30 岁）、中年（中停主 31–50 岁）与晚年（下停主 51 岁后）的侧重各有不同，宜扬长补短。';
  }

  function browText(m) {
    let s;
    if (m.browSlope < -2) s = '眉形上扬，近似「剑眉」，传统主果决进取、有主见、行动力强。';
    else if (m.browSlope > 2) s = '眉尾微垂，传统主性情温和、重感情、待人有耐心。';
    else s = '眉形平顺，传统主性情温和、处事稳重、情绪较为平稳。';
    if (m.browLength / m.fw > 0.42) s += ' 眉长过目，又主寿元与兄弟朋友缘厚。';
    return s;
  }

  function eyeText(m) {
    let s;
    if (m.eyeOpenRatio > 0.38) s = '眼形圆大、眼裂开阖，传统主聪慧开朗、善于表达，神采外露。';
    else if (m.eyeOpenRatio < 0.22) s = '眼形细长，传统主内敛沉着、心思细腻、善于观察。';
    else s = '眼形适中，传统主内外兼修、张弛有度。';
    if (m.eyeGapRatio > 1.15) s += ' 眼距开阔，主胸襟豁达、乐观随和。';
    else if (m.eyeGapRatio < 0.85) s += ' 眼距偏近，主专注执着、心思缜密。';
    return s;
  }

  function noseText(m) {
    const noseWide = m.noseWidth / m.fw > 0.26;
    const noseNarrow = m.noseWidth / m.fw < 0.19;
    let s;
    if (noseWide) s = '鼻翼较宽厚，传统称「财库丰」，主性格豪爽、掌财有方、气量宽宏。';
    else if (noseNarrow) s = '鼻翼偏窄，传统主精打细算、心思细腻、善于理财。';
    else s = '鼻翼适中，传统主财库平稳、收支有度。';
    if (m.noseLength / m.fh > 0.42) s += ' 鼻梁修长，主自尊心强、有主见。';
    else if (m.noseLength / m.fh < 0.30) s += ' 鼻型偏短，主敦厚踏实、为人质朴。';
    return s;
  }

  function mouthText(m) {
    let s;
    if (m.mouthRatio > 0.4) s = '唇形丰润，传统主食禄丰足、重情重义、表达力强。';
    else if (m.mouthRatio < 0.22) s = '唇形偏薄，传统主理性克制、能言善道（古称「薄唇善辩」）。';
    else s = '唇形适中，传统主言谈得体、进退有度。';
    if (m.cornerLift > 2) s += ' 嘴角上扬，主亲和乐观、人缘颇佳。';
    else if (m.cornerLift < -2) s += ' 嘴角微垂，传统主心思较重，宜多舒展情绪。';
    return s;
  }

  function chinText(m) {
    if (m.chinW / m.fw > 0.52) return '地阁方圆，下巴饱满，传统主晚景丰隆、有恒心毅力、福泽绵长。';
    if (m.chinW / m.fw < 0.38) return '下巴尖削，传统主灵活聪慧有余，而耐心与晚景宜多加经营。';
    return '下巴适中，传统主晚景平稳、个性均衡。';
  }

  function foreheadText(m) {
    const upperRatio = m.upper / (m.upper + m.middle + m.lower);
    if (upperRatio > 0.38 && m.foreheadW / m.fw > 0.8) return '天庭饱满开阔，传统主聪明、少年得志，读书与官运有助力。';
    if (upperRatio < 0.28) return '上停（额头）相对偏窄，传统认为早年运势需多靠自身努力，后发亦可至。';
    return '额头适中，传统主智慧与际遇平稳，脚踏实地则有所成。';
  }

  function philtrumText(m) {
    if (m.philtrum / m.fh > 0.16) return '人中深长清晰，传统主肾气充足、生育与寿元之兆较佳。';
    if (m.philtrum / m.fh < 0.09) return '人中偏浅短，传统认为宜注重养生保健、作息规律。';
    return '人中适中，传统主身心较为平衡。';
  }

  function cheekText(m) {
    const cheekMost = m.cheekW >= m.foreheadW - 2 && m.cheekW >= m.jawW - 2;
    if (cheekMost) return '颧骨较为明显，传统主权柄、有主见，宜「鼻颧相配」以显贵气。';
    return '颧骨平和，传统主性情随和、不喜争锋，人缘稳顺。';
  }

  function scoreBar(name, val) {
    return '<div class="score-row"><span class="name">' + name + '</span>' +
      '<span class="score-bar"><i style="width:' + val + '%"></i></span>' +
      '<span class="val">' + val + '</span></div>';
  }

  function renderReport(m) {
    const sh = shapeText(m);
    const bars = [
      ['三停匀称', m.scores.balance],
      ['五官端正', m.scores.features],
      ['面部对称', m.scores.symmetry],
      ['印堂开阔', m.scores.yintang],
      ['财帛丰隆', m.scores.caibo],
      ['地阁丰厚', m.scores.dige]
    ];
    let barsHtml = '';
    bars.forEach(function (b) { barsHtml += scoreBar(b[0], b[1]); });

    const chips = [m.shape];
    if (m.eyeGapRatio > 1.1) chips.push('印堂开阔');
    if (m.noseWidth / m.fw > 0.26) chips.push('鼻翼丰厚');
    if (m.chinW / m.fw > 0.52) chips.push('地阁方圆');
    if (m.cornerLift > 2) chips.push('嘴角上扬');
    if (m.browSlope < -2) chips.push('眉形上扬');
    if (m.symScore >= 85) chips.push('相貌端正');
    const chipsHtml = chips.map(function (c) { return '<span class="chip">' + c + '</span>'; }).join('');

    reportEl.innerHTML =
      '<div class="report-head">' +
        '<div>' +
          '<div class="score-label">综合相格指数（参考）</div>' +
          '<div class="score-big">' + m.overall + '</div>' +
        '</div>' +
        '<div>' +
          '<h3>' + sh[0] + '</h3>' +
          '<div class="verdict">' + sh[1] + '</div>' +
        '</div>' +
      '</div>' +

      '<div class="report-section"><h4>面相维度评分</h4><div class="score-grid">' + barsHtml + '</div></div>' +

      '<div class="report-section"><h4>三停格局</h4>' +
        '<div class="report-item"><span class="ri-title">三停比例</span>' +
        '<p>上停 ' + round(m.upperR * 100) + '% · 中停 ' + round(m.middleR * 100) + '% · 下停 ' + round(m.lowerR * 100) + '%</p>' +
        '<p>' + threePartsText(m) + '</p></div>' +
      '</div>' +

      '<div class="report-section"><h4>五官分论</h4>' +
        '<div class="report-item"><span class="ri-title">眉<small>保寿官 · 兄弟宫</small></span><p>' + browText(m) + '</p></div>' +
        '<div class="report-item"><span class="ri-title">眼<small>监察官</small></span><p>' + eyeText(m) + '</p></div>' +
        '<div class="report-item"><span class="ri-title">鼻<small>审辨官 · 财帛宫</small></span><p>' + noseText(m) + '</p></div>' +
        '<div class="report-item"><span class="ri-title">口<small>出纳官</small></span><p>' + mouthText(m) + '</p></div>' +
        '<div class="report-item"><span class="ri-title">耳<small>采听官</small></span><p>耳部不在正面关键点检测范围内，暂无法测量，建议参考传统「耳垂厚大、轮廓分明」之说自行对照。</p></div>' +
      '</div>' +

      '<div class="report-section"><h4>部位细论</h4>' +
        '<div class="report-item"><span class="ri-title">天庭<small>官禄宫</small></span><p>' + foreheadText(m) + '</p></div>' +
        '<div class="report-item"><span class="ri-title">印堂<small>命宫</small></span><p>' + (m.eyeGapRatio > 1.1 ? '两眉间距开阔，印堂平整，传统主心胸开阔、运势通达。' : m.eyeGapRatio < 0.85 ? '两眉间距偏近，印堂略窄，传统主心思细密，宜多开阔胸襟。' : '印堂适中，传统主气运平稳、心境平和。') + '</p></div>' +
        '<div class="report-item"><span class="ri-title">颧骨</span><p>' + cheekText(m) + '</p></div>' +
        '<div class="report-item"><span class="ri-title">人中</span><p>' + philtrumText(m) + '</p></div>' +
        '<div class="report-item"><span class="ri-title">地阁<small>奴仆宫</small></span><p>' + chinText(m) + '</p></div>' +
      '</div>' +

      '<div class="report-section"><h4>综合评语</h4>' +
        '<p style="color:var(--ink-soft);">' + sh[1] + ' ' + threePartsText(m) + ' 五官之中，' +
        (m.eyeOpenRatio > 0.38 ? '眼神' : m.noseWidth / m.fw > 0.26 ? '鼻相' : '眉形') +
        '较有特色。整体而言，面相讲究「形神兼备、三停相称、五官端正」，' +
        (m.symScore >= 80 ? '此相整体较为端正对称。' : '人人面部皆有轻微不对称，属正常现象。') +
        ' 相由心生，更重要的是由内而外的气度与修养。</p>' +
        '<div class="chips" style="margin-top:12px;">' + chipsHtml + '</div>' +
      '</div>' +

      '<div class="callout warn" style="margin-top:18px;">' +
        '<div class="callout-title">免责声明</div>' +
        '<p>本报告由 AI 根据人脸几何特征，结合传统相学文化自动生成，<strong>仅供娱乐与文化参考</strong>，不构成任何医学、心理、法律或投资建议，请理性看待。</p>' +
      '</div>';

    reportEl.classList.add('show');
  }

  /* ---------- 主流程 ---------- */
  async function analyze() {
    if (!currentImage) { setStatus('请先上传一张照片', 'err'); return; }
    if (typeof faceapi === 'undefined') { setStatus('人脸识别库未加载，请检查网络后刷新', 'err'); return; }
    analyzeBtn.disabled = true;
    try {
      setStatusLoading('正在加载人脸识别模型（首次约需数秒）…');
      await ensureModels();

      setStatusLoading('正在检测人脸与关键点…');
      const detection = await faceapi.detectSingleFace(
        currentImage,
        new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.35 })
      ).withFaceLandmarks();

      if (!detection) {
        setStatus('未检测到清晰正脸，请上传光线充足、正面、无遮挡的照片', 'err');
        analyzeBtn.disabled = false;
        return;
      }

      drawImageOnly();
      drawOverlay(detection);

      const metrics = computeMetrics(detection);
      renderReport(metrics);
      setStatus('分析完成 ✔', 'ok');

      // 滚动到报告
      if (window.innerWidth <= 900) {
        reportEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (e) {
      console.error(e);
      setStatus('分析失败：' + (e && e.message ? e.message : '请检查网络后重试'), 'err');
    } finally {
      analyzeBtn.disabled = !currentImage;
    }
  }

  /* ---------- 事件绑定 ---------- */
  dropzone.addEventListener('click', function () { fileInput.click(); });
  dropzone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', function () { handleFile(fileInput.files[0]); });

  ['dragenter', 'dragover'].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.add('drag'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.remove('drag'); });
  });
  dropzone.addEventListener('drop', function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  clearBtn.addEventListener('click', clearImage);
  analyzeBtn.addEventListener('click', analyze);
})();
