import * as THREE from 'three/webgpu';
import { TOKAMAK } from '../reactor/tokamakMath.js';

export const REACTOR_MODES = {
  standby: {
    label: 'CRYO STANDBY',
    badgeClass: 'standby',
    bField: 3.2,
    heating: 0,
    qFactor: 4.6,
    plasmaIntensity: 0.0,
    coilEnergy: 0.22,
    divertorHeat: 0.0,
    nbiPower: 0.0,
    instability: 0.04,
    cryoTone: 1.0,
  },
  ramp: {
    label: 'MAGNETIC RAMP',
    badgeClass: '',
    bField: 8.4,
    heating: 38,
    qFactor: 3.9,
    plasmaIntensity: 0.58,
    coilEnergy: 0.92,
    divertorHeat: 0.28,
    nbiPower: 0.45,
    instability: 0.12,
    cryoTone: 0.35,
  },
  ignition: {
    label: 'D-T IGNITION',
    badgeClass: '',
    bField: 11.8,
    heating: 96,
    qFactor: 3.1,
    plasmaIntensity: 1.05,
    coilEnergy: 1.0,
    divertorHeat: 0.85,
    nbiPower: 0.9,
    instability: 0.18,
    cryoTone: 0.0,
  },
  overdrive: {
    label: 'ELM OVERDRIVE',
    badgeClass: 'overdrive',
    bField: 13.8,
    heating: 148,
    qFactor: 2.15,
    plasmaIntensity: 1.32,
    coilEnergy: 1.38,
    divertorHeat: 1.35,
    nbiPower: 1.45,
    instability: 0.58,
    cryoTone: 0.0,
  },
};

export const CAMERA_PRESETS = {
  hero: {
    pos: new THREE.Vector3(8.75, 3.85, 8.15),
    target: new THREE.Vector3(0.0, -0.32, 0.0),
  },
  core: {
    pos: new THREE.Vector3(4.55, 0.42, 4.10),
    target: new THREE.Vector3(1.55, -0.58, 1.25),
  },
  top: {
    pos: new THREE.Vector3(0.9, 10.6, 3.6),
    target: new THREE.Vector3(0.0, -0.25, 0.0),
  },
  nbi: {
    pos: new THREE.Vector3(-8.4, 2.35, 4.65),
    target: new THREE.Vector3(-3.65, -0.15, 1.55),
  },
  ride: {
    // Dynamic camera riding inside the torus channel
    dynamic: true,
  },
};

const CALLOUT_DEFS = [
  {
    id: 'solenoid',
    title: 'CENTRAL SOLENOID',
    sub: 'Nb₃Sn Inductive Core · 13 T',
    pos: new THREE.Vector3(0.0, 2.98, 0.0),
  },
  {
    id: 'tfcoil',
    title: 'TF D-SHAPED COILS',
    sub: '16× Superconducting 4.2 K',
    pos: new THREE.Vector3(2.45, 2.25, -1.95),
  },
  {
    id: 'plasma',
    title: 'D-T FUSION PLASMA',
    sub: 'WebGPU Helical Compute Ions',
    pos: new THREE.Vector3(2.35, 0.50, 1.55),
  },
  {
    id: 'divertor',
    title: 'TUNGSTEN DIVERTOR',
    sub: '32× Blackbody Exhaust Plates',
    pos: new THREE.Vector3(1.80, -1.52, 1.85),
  },
  {
    id: 'nbi',
    title: 'NEUTRAL BEAM INJECTOR',
    sub: '1 MeV Deuterium Heating',
    pos: new THREE.Vector3(-4.2, 0.85, 2.05),
  },
];

export class HudController {
  constructor(state, callbacks) {
    this.state = state;
    this.callbacks = callbacks;

    this.el = {
      backendValue: document.getElementById('backend-value'),
      particlesValue: document.getElementById('particles-value'),
      fpsValue: document.getElementById('fps-value'),
      stateBadge: document.getElementById('state-badge'),
      telTemp: document.getElementById('tel-temp'),
      telQgain: document.getElementById('tel-qgain'),
      telCurrent: document.getElementById('tel-current'),
      telDivertor: document.getElementById('tel-divertor'),
      stabilityPct: document.getElementById('stability-pct'),
      stabilityBar: document.getElementById('stability-bar'),
      sliderCutaway: document.getElementById('slider-cutaway'),
      valCutaway: document.getElementById('val-cutaway'),
      sliderBfield: document.getElementById('slider-bfield'),
      valBfield: document.getElementById('val-bfield'),
      sliderHeating: document.getElementById('slider-heating'),
      valHeating: document.getElementById('val-heating'),
      sliderQfactor: document.getElementById('slider-qfactor'),
      valQfactor: document.getElementById('val-qfactor'),
      toggleFlux: document.getElementById('toggle-flux'),
      toggleCallouts: document.getElementById('toggle-callouts'),
      toggleOrbit: document.getElementById('toggle-orbit'),
      toggleBackend: document.getElementById('toggle-backend'),
      calloutsLayer: document.getElementById('callouts-layer'),
      btnToggleHud: document.getElementById('btn-toggle-hud'),
      btnRestoreHud: document.getElementById('btn-restore-hud'),
    };

    this.calloutNodes = [];
    this._projVec = new THREE.Vector3();
    this._fpsFrames = 0;
    this._fpsAcc = 0;
    this._hudTimer = 0;

    this._buildCallouts();
    this._bindEvents();
    this.syncSlidersFromState();
  }

  setBackendLabel(isWebGPU, adapterInfo = '') {
    if (isWebGPU) {
      this.el.backendValue.textContent = 'WEBGPU · WGSL COMPUTE';
      this.el.backendValue.classList.add('accent-cyan');
      this.el.toggleBackend.textContent = 'TRY WEBGL2';
      if (adapterInfo) {
        document.getElementById('backend-chip').title = `WebGPU Adapter: ${adapterInfo}`;
      }
    } else {
      this.el.backendValue.textContent = 'WEBGL2 · TSL FALLBACK';
      this.el.backendValue.classList.remove('accent-cyan');
      this.el.backendValue.classList.add('accent-gold');
      this.el.toggleBackend.textContent = 'TRY WEBGPU';
    }
  }

  _buildCallouts() {
    this.el.calloutsLayer.innerHTML = '';
    for (const def of CALLOUT_DEFS) {
      const node = document.createElement('div');
      node.className = 'callout';
      node.innerHTML = `
        <span class="callout-dot"></span>
        <span class="callout-line"></span>
        <div class="callout-card">
          <div class="callout-title">${def.title}</div>
          <div class="callout-sub">${def.sub}</div>
        </div>
      `;
      this.el.calloutsLayer.appendChild(node);
      this.calloutNodes.push({ def, node });
    }
  }

  _bindEvents() {
    // Operating mode buttons
    document.querySelectorAll('#mode-buttons .mode-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        this.applyMode(mode);
      });
    });

    // Action buttons
    document.getElementById('btn-elm').addEventListener('click', () => {
      this.state.elmBurst = Math.min(1.55, this.state.elmBurst + 1.15);
      this.state.elmFlash = 1.0;
    });

    document.getElementById('btn-nbi').addEventListener('click', () => {
      this.state.nbiBoost = 1.1;
      this.state.target.heating = Math.min(150, this.state.target.heating + 25);
      this.syncSlidersFromState();
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
      this.state.target.cutaway = 0.42;
      this.applyMode('ignition');
      this.callbacks.onSelectCamera('hero');
      this._setActiveCamBtn('hero');
    });

    // Sliders
    this.el.sliderCutaway.addEventListener('input', (e) => {
      const pct = Number(e.target.value);
      this.state.target.cutaway = pct / 100;
      this.el.valCutaway.textContent = `${pct}%`;
    });

    this.el.sliderBfield.addEventListener('input', (e) => {
      const val = Number(e.target.value);
      this.state.target.bField = val;
      this.el.valBfield.textContent = `${val.toFixed(1)} T`;
    });

    this.el.sliderHeating.addEventListener('input', (e) => {
      const val = Number(e.target.value);
      this.state.target.heating = val;
      if (val > 5 && this.state.target.plasmaIntensity < 0.25) {
        this.state.target.plasmaIntensity = 0.75;
      }
      this.el.valHeating.textContent = `${val} MW`;
    });

    this.el.sliderQfactor.addEventListener('input', (e) => {
      const val = Number(e.target.value);
      this.state.target.qFactor = val;
      this.el.valQfactor.textContent = val.toFixed(2);
    });

    // GPU particle count selector
    document.querySelectorAll('#particle-count-btns .seg-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const count = Number(btn.dataset.count);
        document.querySelectorAll('#particle-count-btns .seg-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.el.particlesValue.textContent = count.toLocaleString();
        await this.callbacks.onParticleCountChange(count);
      });
    });

    // Toggles
    this.el.toggleFlux.addEventListener('click', () => {
      this.state.showFlux = !this.state.showFlux;
      this.el.toggleFlux.classList.toggle('active', this.state.showFlux);
    });

    this.el.toggleCallouts.addEventListener('click', () => {
      this.state.showCallouts = !this.state.showCallouts;
      this.el.toggleCallouts.classList.toggle('active', this.state.showCallouts);
      this.el.calloutsLayer.classList.toggle('hidden', !this.state.showCallouts);
    });

    this.el.toggleOrbit.addEventListener('click', () => {
      this.state.autoOrbit = !this.state.autoOrbit;
      this.el.toggleOrbit.classList.toggle('active', this.state.autoOrbit);
    });

    this.el.toggleBackend.addEventListener('click', () => {
      this.callbacks.onToggleBackend();
    });

    // Camera presets
    document.querySelectorAll('#camera-buttons .cam-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const cam = btn.dataset.cam;
        this._setActiveCamBtn(cam);
        this.callbacks.onSelectCamera(cam);
      });
    });

    // HUD Zen Mode
    const toggleZen = () => {
      const zen = document.body.classList.toggle('zen-mode');
      this.el.btnRestoreHud.classList.toggle('hidden', !zen);
    };
    this.el.btnToggleHud.addEventListener('click', toggleZen);
    this.el.btnRestoreHud.addEventListener('click', toggleZen);
    window.addEventListener('keydown', (e) => {
      if (e.key === 'h' || e.key === 'H') toggleZen();
    });
  }

  _setActiveCamBtn(cam) {
    document.querySelectorAll('#camera-buttons .cam-btn').forEach((b) => {
      b.classList.toggle('active', b.dataset.cam === cam);
    });
  }

  applyMode(modeKey) {
    const preset = REACTOR_MODES[modeKey];
    if (!preset) return;
    this.state.mode = modeKey;
    Object.assign(this.state.target, {
      bField: preset.bField,
      heating: preset.heating,
      qFactor: preset.qFactor,
      plasmaIntensity: preset.plasmaIntensity,
      coilEnergy: preset.coilEnergy,
      divertorHeat: preset.divertorHeat,
      nbiPower: preset.nbiPower,
      instability: preset.instability,
      cryoTone: preset.cryoTone,
    });

    if (modeKey === 'overdrive') {
      this.state.elmBurst = Math.max(this.state.elmBurst, 0.85);
      this.state.elmFlash = 0.7;
    }

    document.querySelectorAll('#mode-buttons .mode-btn').forEach((b) => {
      b.classList.toggle('active', b.dataset.mode === modeKey);
    });
    this.el.stateBadge.textContent = preset.label;
    this.el.stateBadge.className = `status-tag ${preset.badgeClass}`;
    this.syncSlidersFromState();
  }

  syncSlidersFromState() {
    const t = this.state.target;
    const cutPct = Math.round(t.cutaway * 100);
    this.el.sliderCutaway.value = String(cutPct);
    this.el.valCutaway.textContent = `${cutPct}%`;

    this.el.sliderBfield.value = t.bField.toFixed(1);
    this.el.valBfield.textContent = `${t.bField.toFixed(1)} T`;

    this.el.sliderHeating.value = String(Math.round(t.heating));
    this.el.valHeating.textContent = `${Math.round(t.heating)} MW`;

    this.el.sliderQfactor.value = t.qFactor.toFixed(2);
    this.el.valQfactor.textContent = t.qFactor.toFixed(2);
  }

  update(dt, camera) {
    // FPS counter
    this._fpsFrames++;
    this._fpsAcc += dt;
    if (this._fpsAcc >= 0.45) {
      const fps = Math.round(this._fpsFrames / this._fpsAcc);
      this.el.fpsValue.textContent = `${fps} FPS`;
      this._fpsFrames = 0;
      this._fpsAcc = 0;
    }

    // Throttle telemetry text updates to ~10 Hz
    this._hudTimer += dt;
    if (this._hudTimer >= 0.1) {
      this._hudTimer = 0;
      const s = this.state;
      const active = Math.min(1, s.plasmaIntensity);
      const tempMK = active * (22 + s.heating * 1.18 + s.bField * 2.1 + s.elmBurst * 24);
      const qGain = active * Math.max(0, (s.bField * 0.92 * (s.heating / 85 + 0.35)) / Math.max(1.2, s.qFactor * 0.34));
      const currentMA = active * (s.bField * 1.48 / Math.max(1.5, s.qFactor * 0.36));
      const divFlux = active * (1.8 + s.heating * 0.068 + s.elmBurst * 12.5);
      const stability = Math.max(
        18,
        Math.min(
          99,
          Math.round(
            98 - s.instability * 42 - s.elmBurst * 26 - Math.max(0, 2.2 - s.qFactor) * 18
          )
        )
      );

      this.el.telTemp.innerHTML = `${tempMK.toFixed(1)} <small>MK</small>`;
      this.el.telQgain.textContent = qGain.toFixed(1);
      this.el.telCurrent.innerHTML = `${currentMA.toFixed(1)} <small>MA</small>`;
      this.el.telDivertor.innerHTML = `${divFlux.toFixed(1)} <small>MW/m²</small>`;
      this.el.stabilityPct.textContent = `${stability}%`;
      this.el.stabilityBar.style.width = `${stability}%`;
      this.el.stabilityBar.style.background =
        stability < 55
          ? 'linear-gradient(90deg, #ff4769, #ff9947)'
          : 'linear-gradient(90deg, #b66dff, #38e8ff)';
    }

    // Project 3D component callouts to screen space
    if (this.state.showCallouts && !document.body.classList.contains('zen-mode')) {
      const w = window.innerWidth;
      const h = window.innerHeight;
      for (const item of this.calloutNodes) {
        this._projVec.copy(item.def.pos).project(camera);
        const visible = this._projVec.z > -1 && this._projVec.z < 1;
        if (!visible) {
          item.node.style.opacity = '0';
          continue;
        }
        const sx = (this._projVec.x * 0.5 + 0.5) * w;
        const sy = (-this._projVec.y * 0.5 + 0.5) * h;
        const inBounds = sx > 315 && sx < w - 340 && sy > 90 && sy < h - 95;
        item.node.style.opacity = inBounds ? '0.92' : '0';
        item.node.style.left = `${sx}px`;
        item.node.style.top = `${sy}px`;
      }
    }
  }
}
