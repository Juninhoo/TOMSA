        const timeline = __TIMELINE_JSON__;
        const dt_step = timeline.dt_step;
        const t0_base = timeline.t0_base;
        const t_max = timeline.t_max;
        const numSteps = timeline.num_steps;
        const robotKeys = Object.keys(timeline.robots).sort();
        const DEG = Math.PI / 180.0;
        const FOV_F = 1.0 / Math.tan(35.0 * DEG);   // distância focal para FOV horizontal de 70 graus (u em [-1, 1])
        const HALF_V = 0.5625;                       // meia-altura do ecrã (16:9)

        let currentView = 'global';
        let currentRobot = '__ACTIVE_ROBOT__';
        let isPlaying = false;
        let speed = 1.0;
        let currentTime = t0_base;
        let animFrameId = null;
        let lastFrameTimestamp = null;
        let lastRender = 0;
        let lastActiveKey = '';

        // ---------- utilitários de interpolação ----------
        function curF() {
            return Math.min(numSteps - 1, Math.max(0, (currentTime - t0_base) / dt_step));
        }
        function at(arr, f) {
            const i = Math.floor(f);
            if (i >= arr.length - 1) return arr[arr.length - 1];
            const w = f - i;
            return arr[i] * (1 - w) + arr[i + 1] * w;
        }
        function isActive(r, f) { return f >= r.k0 && f <= r.k1; }
        function stateAt(r, f) {
            return {
                E: at(r.E, f), N: at(r.N, f), D: at(r.D, f),
                yaw: at(r.yaw, f), pitch: at(r.pitch, f), roll: at(r.roll, f),
                spd: at(r.spd, f), turn: at(r.turn, f)
            };
        }
        // Vetor no referencial do corpo (x frente, y direita, z baixo): R^T (ZYX) * (dN, dE, dD)
        function toBody(s, dN, dE, dD) {
            const psi = s.yaw * DEG, the = s.pitch * DEG, phi = s.roll * DEG;
            const cPsi = Math.cos(psi), sPsi = Math.sin(psi);
            const cThe = Math.cos(the), sThe = Math.sin(the);
            const cPhi = Math.cos(phi), sPhi = Math.sin(phi);
            return {
                x: cThe * cPsi * dN + cThe * sPsi * dE - sThe * dD,
                y: (sPhi * sThe * cPsi - cPhi * sPsi) * dN + (sPhi * sThe * sPsi + cPhi * cPsi) * dE + sPhi * cThe * dD,
                z: (cPhi * sThe * cPsi + sPhi * sPsi) * dN + (cPhi * sThe * sPsi - sPhi * cPsi) * dE + cPhi * cThe * dD
            };
        }
        function fmtElapsed(sec) {
            const m = Math.floor(sec / 60);
            return m + 'm ' + (sec - 60 * m).toFixed(1) + 's';
        }

        function computeSubmarineWireframe(e, n, d, yawDeg, pitchDeg) {
            const yaw = yawDeg * DEG, pitch = pitchDeg * DEG;
            const length = 8.0, wingspan = 3.5, height = 2.5;
            const fe = Math.sin(yaw) * Math.cos(pitch);
            const fn = Math.cos(yaw) * Math.cos(pitch);
            const fd = -Math.sin(pitch);
            const re = Math.cos(yaw), rn = -Math.sin(yaw), rd = 0.0;
            const ue = Math.sin(yaw) * Math.sin(pitch);
            const un = Math.cos(yaw) * Math.sin(pitch);
            const ud = -Math.cos(pitch);
            const nose_e = e + length * fe, nose_n = n + length * fn, nose_d = d + length * fd;
            const stern_e = e - 0.5 * length * fe, stern_n = n - 0.5 * length * fn, stern_d = d - 0.5 * length * fd;
            const port_e = stern_e - 0.5 * wingspan * re, port_n = stern_n - 0.5 * wingspan * rn, port_d = stern_d - 0.5 * wingspan * rd;
            const stbd_e = stern_e + 0.5 * wingspan * re, stbd_n = stern_n + 0.5 * wingspan * rn, stbd_d = stern_d + 0.5 * wingspan * rd;
            const rud_e = stern_e + height * ue, rud_n = stern_n + height * un, rud_d = stern_d + height * ud;
            return {
                x: [nose_e, stern_e, port_e, stbd_e, stern_e, rud_e, nose_e],
                y: [nose_n, stern_n, port_n, stbd_n, stern_n, rud_n, nose_n],
                z: [nose_d, stern_d, port_d, stbd_d, stern_d, rud_d, nose_d]
            };
        }

        // ---------- Vista Global 3D ----------
        function updateGlobalView(f) {
            const gd = document.getElementById('plot-global');
            if (!gd || !gd.data) return;
            const idx = [], X = [], Y = [], Z = [];
            const i0 = Math.floor(f);
            const flags = [];
            const mkIdx = [], mkOpacity = [], mkText = [], wfOpacity = [], wfIdx = [];

            robotKeys.forEach((id, i) => {
                const r = timeline.robots[id];
                const s = stateAt(r, f);
                const act = isActive(r, f);
                flags.push(act ? 1 : 0);

                // Rasto progressivo: só a parte já percorrida pelo robô
                const sIdx = Math.max(0, Math.floor(r.k0));
                const eIdx = Math.max(sIdx, Math.min(i0, Math.ceil(r.k1)));
                const tx = r.E.slice(sIdx, eIdx + 1), ty = r.N.slice(sIdx, eIdx + 1), tz = r.D.slice(sIdx, eIdx + 1);
                if (act) { tx.push(s.E); ty.push(s.N); tz.push(s.D); }
                idx.push(2 + 3 * i); X.push(tx); Y.push(ty); Z.push(tz);

                idx.push(3 + 3 * i); X.push([s.E]); Y.push([s.N]); Z.push([s.D]);
                const g = computeSubmarineWireframe(s.E, s.N, s.D, s.yaw, s.pitch);
                idx.push(4 + 3 * i); X.push(g.x); Y.push(g.y); Z.push(g.z);

                mkIdx.push(3 + 3 * i); mkOpacity.push(act ? 1.0 : 0.25);
                mkText.push([act ? ('<b>' + id + '</b><br>' + s.spd.toFixed(1) + ' m/s') : ('<b>' + id + '</b><br>inativo')]);
                wfIdx.push(4 + 3 * i); wfOpacity.push(act ? 1.0 : 0.25);
            });
            Plotly.restyle(gd, { x: X, y: Y, z: Z }, idx);

            const key = flags.join('');
            if (key !== lastActiveKey || (lastRender % 7) === 0) {
                lastActiveKey = key;
                Plotly.restyle(gd, { 'marker.opacity': mkOpacity, text: mkText }, mkIdx);
                Plotly.restyle(gd, { opacity: wfOpacity }, wfIdx);
            }
        }

        // ---------- Vista POV: câmara em perspetiva no submarino ativo ----------
        function projectPt(p) {
            if (p.x <= 0.3) return null;
            return [FOV_F * p.y / p.x, FOV_F * (-p.z) / p.x];
        }
        function updatePovView(f) {
            const gd = document.getElementById('plot-pov');
            if (!gd || !gd.data) return;
            const act = timeline.robots[currentRobot];
            if (!act) return;
            const s = stateAt(act, f);

            // Horizonte: direções horizontais do mundo no plano de imagem. g = "para baixo" no referencial do corpo.
            const the = s.pitch * DEG, phi = s.roll * DEG;
            const gx = -Math.sin(the), gy = Math.sin(phi) * Math.cos(the);
            let gz = Math.cos(phi) * Math.cos(the);
            if (Math.abs(gz) < 0.05) gz = (gz < 0 ? -0.05 : 0.05);
            const hv = (u) => (FOV_F * gx + gy * u) / gz;
            const vL = hv(-4), vR = hv(4);
            const upper = { x: [-4, 4, 4, -4], y: [vL, vR, vR + 10, vL + 10] };
            const lower = { x: [-4, 4, 4, -4], y: [vL, vR, vR - 10, vL - 10] };

            const idx = [0, 1, 2], X = [upper.x, lower.x, [-4, 4]], Y = [upper.y, lower.y, [vL, vR]];
            const mIdx = [], mText = [], mSize = [], mSym = [];

            robotKeys.forEach((id, i) => {
                const trI = 5 + 2 * i, mkI = 6 + 2 * i;
                const r = timeline.robots[id];
                if (id === currentRobot || !isActive(r, f)) {
                    idx.push(trI); X.push([]); Y.push([]);
                    idx.push(mkI); X.push([]); Y.push([]);
                    mIdx.push(mkI); mText.push([]); mSize.push(14); mSym.push('diamond');
                    return;
                }
                // Rasto do outro robô projetado na câmara (decimado)
                const sIdx = Math.max(0, Math.floor(r.k0));
                const eIdx = Math.max(sIdx, Math.min(Math.floor(f), Math.ceil(r.k1)));
                const stride = Math.max(1, Math.floor((eIdx - sIdx) / 350));
                const tx = [], ty = [];
                for (let k = sIdx; k <= eIdx; k += stride) {
                    const b = toBody(s, r.N[k] - s.N, r.E[k] - s.E, r.D[k] - s.D);
                    const q = projectPt(b);
                    if (q && Math.abs(q[0]) < 8 && Math.abs(q[1]) < 8) { tx.push(q[0]); ty.push(q[1]); }
                    else { tx.push(null); ty.push(null); }
                }
                idx.push(trI); X.push(tx); Y.push(ty);

                // Marcador do outro robô (ou indicador na margem do ecrã se estiver fora do campo de visão)
                const t = stateAt(r, f);
                const b = toBody(s, t.N - s.N, t.E - s.E, t.D - s.D);
                const dist = Math.sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
                const q = projectPt(b);
                const inFov = q && Math.abs(q[0]) <= 1.0 && Math.abs(q[1]) <= HALF_V;
                let px, py, label, size, sym;
                if (inFov) {
                    px = q[0]; py = q[1];
                    label = '<b>' + id + '</b><br>' + dist.toFixed(0) + ' m';
                    size = Math.max(12, Math.min(44, 2600 / Math.max(dist, 1)));
                    sym = 'diamond';
                } else {
                    let dx = q ? q[0] : b.y, dy = q ? q[1] : -b.z;
                    if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) { dx = 0; dy = -1; }
                    const sc = 0.93 / Math.max(Math.abs(dx) / 1.0, Math.abs(dy) / HALF_V);
                    px = dx * sc; py = dy * sc;
                    label = id + ' ' + dist.toFixed(0) + ' m' + (b.x > 0 ? '' : ' (atrás)');
                    size = 16; sym = 'circle-open';
                }
                idx.push(mkI); X.push([px]); Y.push([py]);
                mIdx.push(mkI); mText.push([label]); mSize.push(size); mSym.push(sym);
            });

            // HUD
            const hud = currentRobot + '  |  PROA ' + (((s.yaw % 360) + 360) % 360).toFixed(0) + ' graus  |  PROF ' +
                s.D.toFixed(1) + ' m  |  ' + s.spd.toFixed(2) + ' m/s  |  pitch ' + s.pitch.toFixed(1) + '  roll ' + s.roll.toFixed(1) +
                (isActive(act, f) ? '' : '  |  [FORA DE OPERACAO]');
            idx.push(4); X.push([-0.98]); Y.push([0.545]);
            Plotly.restyle(gd, { x: X, y: Y }, idx);
            Plotly.restyle(gd, { text: [[hud]] }, [4]);
            if (mIdx.length) {
                Plotly.restyle(gd, { text: mText, 'marker.size': mSize, 'marker.symbol': mSym }, mIdx);
            }
        }

        // ---------- Cockpit ----------
        function updateCockpitView(f) {
            const gd = document.getElementById('plot-cockpit');
            if (!gd || !gd.data) return;
            const actR = timeline.robots[currentRobot];
            if (!actR) return;
            const s = stateAt(actR, f);
            const active = isActive(actR, f);
            const yaw = ((s.yaw % 360.0) + 360.0) % 360.0;

            let minDist = null;
            const rs = [], thetas = [], texts = [], colors = [];
            robotKeys.forEach(id => {
                if (id === currentRobot) return;
                const r = timeline.robots[id];
                if (!isActive(r, f) || !active) return;
                const t = stateAt(r, f);
                const dN = t.N - s.N, dE = t.E - s.E, dD = t.D - s.D;
                const dist = Math.sqrt(dN * dN + dE * dE + dD * dD);
                if (minDist === null || dist < minDist) minDist = dist;
                const psi = s.yaw * DEG;
                const xb = Math.cos(psi) * dN + Math.sin(psi) * dE;
                const yb = -Math.sin(psi) * dN + Math.cos(psi) * dE;
                thetas.push(((Math.atan2(yb, xb) / DEG) % 360.0 + 360.0) % 360.0);
                rs.push(dist);
                texts.push(id + '<br>Dist: ' + dist.toFixed(1) + 'm');
                colors.push(dist < 10.0 ? '#e63946' : '#52b788');
            });

            const risk = (minDist !== null && minDist < 10.0);
            const statusColor = risk ? '#e63946' : '#2a9d8f';
            const statusMsg = risk ? 'PERIGO DE COLISÃO (< 10m)' : (minDist === null ? 'SEM OUTROS ROBÔS ATIVOS' : 'ROTA SEGURA');
            const sg = (v) => (v >= 0 ? '+' : '') + v.toFixed(1);

            Plotly.restyle(gd, {
                value: [yaw],
                'gauge.bar.color': [actR.color],
                title: [{ text: 'Roll: ' + sg(s.roll) + '° | Pitch: ' + sg(s.pitch) + '°' + (active ? '' : ' [INATIVO]'), font: { color: '#e0e1dd', size: 14 } }]
            }, [0]);
            Plotly.restyle(gd, { value: [s.D] }, [1]);
            Plotly.restyle(gd, { value: [s.spd] }, [2]);
            Plotly.restyle(gd, { r: [rs], theta: [thetas], hovertext: [texts], text: [texts.map(t => t.split('<br>')[0])], 'marker.color': [colors] }, [3]);
            Plotly.restyle(gd, { value: [s.turn] }, [4]);
            Plotly.restyle(gd, {
                value: [minDist === null ? 0 : minDist],
                'number.font.color': [statusColor],
                title: [{
                    text: '<b>' + statusMsg + '</b><br><span style="font-size:0.8em;color:#e0e1dd">Menor distância: ' + (minDist === null ? '-' : minDist.toFixed(1) + ' m') + '</span>',
                    font: { size: 16, color: statusColor }
                }]
            }, [5]);
        }

        // ---------- Gráfico 2D de distâncias: move o cursor de tempo ----------
        function updateProxView(f) {
            const gd = document.getElementById('plot-prox');
            if (!gd || !gd.layout || !gd.layout.shapes || gd.layout.shapes.length < 2) return;
            const tRel = f * dt_step;
            Plotly.relayout(gd, { 'shapes[1].x0': tRel, 'shapes[1].x1': tRel });
        }

        function updateFrame(f) {
            if (currentView === 'global') updateGlobalView(f);
            else if (currentView === 'pov') updatePovView(f);
            else if (currentView === 'cockpit') updateCockpitView(f);
            else if (currentView === 'prox') updateProxView(f);
        }

        // ---------- controlos ----------
        function setView(view) {
            currentView = view;
            document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-buttons button').forEach(b => {
                if (b.id.startsWith('btn-global') || b.id.startsWith('btn-pov') || b.id.startsWith('btn-cockpit') || b.id.startsWith('btn-prox'))
                    b.classList.remove('active');
            });
            const p = document.getElementById('panel-' + view);
            if (p) p.classList.add('active');
            const btn = document.getElementById('btn-' + view);
            if (btn) btn.classList.add('active');
            const targetPlot = document.getElementById('plot-' + view);
            if (targetPlot && window.Plotly) Plotly.Plots.resize(targetPlot);
            updateFrame(curF());
        }

        function setRobot(robotId) {
            currentRobot = robotId;
            document.getElementById('btn-r1').classList.toggle('active', robotId === 'Robot 1');
            document.getElementById('btn-r4').classList.toggle('active', robotId === 'Robot 4');
            document.getElementById('btn-r7').classList.toggle('active', robotId === 'Robot 7');
            const gd = document.getElementById('plot-global');
            if (gd && gd.data) {
                robotKeys.forEach((id, i) => {
                    const sel = (id === currentRobot);
                    Plotly.restyle(gd, { 'marker.size': [sel ? 8 : 6], 'marker.line.color': [sel ? '#ffffff' : timeline.robots[id].color] }, [3 + 3 * i]);
                    Plotly.restyle(gd, { 'line.color': [sel ? '#ffffff' : timeline.robots[id].color] }, [4 + 3 * i]);
                });
            }
            updateFrame(curF());
        }

        function setSpeed(val) {
            speed = val;
            const map = { 'spd-slow': 0.25, 'spd-norm': 1.0, 'spd-fast': 5.0, 'spd-10': 10.0, 'spd-20': 20.0 };
            Object.keys(map).forEach(k => document.getElementById(k).classList.toggle('active', val === map[k]));
            document.getElementById('spd-ultra').classList.toggle('active', val >= 40.0);
        }

        function togglePlay() {
            isPlaying = !isPlaying;
            document.getElementById('btn-play').innerHTML = isPlaying ? '&#9208; Pausa' : '&#9654; Play';
            if (isPlaying) {
                lastFrameTimestamp = null;
                animFrameId = requestAnimationFrame(runPlaybackLoop);
            } else if (animFrameId) {
                cancelAnimationFrame(animFrameId);
                animFrameId = null;
            }
        }

        function resetTrajectory() {
            currentTime = t0_base;
            document.getElementById('time-slider').value = t0_base;
            onSliderChange(t0_base);
        }

        function onSliderChange(val) {
            currentTime = parseFloat(val);
            document.getElementById('time-display').innerText = fmtElapsed(currentTime - t0_base);
            updateFrame(curF());
        }

        function runPlaybackLoop(timestamp) {
            if (!isPlaying) { lastFrameTimestamp = null; return; }
            if (!lastFrameTimestamp) lastFrameTimestamp = timestamp;
            const dtSec = Math.min(0.25, (timestamp - lastFrameTimestamp) / 1000.0);
            lastFrameTimestamp = timestamp;

            let effectiveSpeed = speed;
            if (speed >= 40.0) effectiveSpeed = (t_max - t0_base) / 2.0;   // missão completa em ~2 segundos

            currentTime += dtSec * effectiveSpeed;
            if (currentTime > t_max) currentTime = t0_base;

            document.getElementById('time-slider').value = currentTime;
            document.getElementById('time-display').innerText = fmtElapsed(currentTime - t0_base);

            if (timestamp - lastRender >= 33) {   // ~30 atualizações/s dos gráficos
                lastRender = timestamp;
                updateFrame(curF());
            }
            animFrameId = requestAnimationFrame(runPlaybackLoop);
        }

        // Atalhos de teclado
        window.addEventListener('keydown', (e) => {
            if (e.target && e.target.tagName === 'INPUT') return;
            if (e.key === '1') setRobot('Robot 1');
            if (e.key === '2') setRobot('Robot 4');
            if (e.key === '3') setRobot('Robot 7');
            if (e.key.toLowerCase() === 'v') {
                if (currentView === 'global') setView('pov');
                else if (currentView === 'pov') setView('cockpit');
                else if (currentView === 'cockpit') setView('prox');
                else setView('global');
            }
            if (e.key.toLowerCase() === 'r') resetTrajectory();
            if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        });

        // Inicialização com parâmetros de URL: ?t=segundos_desde_o_inicio&view=pov&robot=2&play=1 (ou #t=...)
        function initFromUrl() {
            const params = new URLSearchParams(window.location.search);
            const hash = window.location.hash;
            let rel = 0;
            if (params.has('t')) rel = parseFloat(params.get('t'));
            else if (hash && hash.startsWith('#t=')) rel = parseFloat(hash.substring(3));
            if (params.has('robot')) setRobot(['Robot 1', 'Robot 4', 'Robot 7'][parseInt(params.get('robot')) - 1] || currentRobot);
            if (params.has('view')) setView(params.get('view'));
            if (params.has('speed')) setSpeed(parseFloat(params.get('speed')));
            currentTime = t0_base + rel;
            document.getElementById('time-slider').value = currentTime;
            onSliderChange(currentTime);
            if ((params.get('play') === '1' || hash === '#play') && !isPlaying) togglePlay();
        }
        let initDone = false;
        function initOnce() { if (initDone) return; initDone = true; initFromUrl(); }
        window.addEventListener('load', () => setTimeout(initOnce, 300));
