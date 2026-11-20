import './style.css';
import * as THREE from 'three/webgpu';
import { pass } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { buildTokamak } from './reactor/buildTokamak.js';
import { TOKAMAK, millerPoint } from './reactor/tokamakMath.js';
import { PlasmaSystem } from './plasma/PlasmaSystem.js';
import { HudController, CAMERA_PRESETS, REACTOR_MODES } from './ui/HudController.js';

/**
 * Create a custom dark aerospace light-rig environment for crisp metallic reflections
 * on brushed titanium, superconducting copper, and sapphire glass.
 */
function buildStudioEnvironment(renderer) {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  envScene.background = new THREE.Color(0x050912);

  const addPanel = (pos, scale, col, rot = [0, 0, 0]) => {
    const mat = new THREE.MeshBasicMaterial({ color: col });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(scale[0], scale[1], scale[2]), mat);
    mesh.position.set(pos[0], pos[1], pos[2]);
    mesh.rotation.set(rot[0], rot[1], rot[2]);
    envScene.add(mesh);
  };

  // Overhead cool industrial light strips
  addPanel([0, 9, 0], [10, 0.4, 2.5], new THREE.Color(0xa6dcff).multiplyScalar(3.2));
  addPanel([0, 8.5, 6], [8, 0.3, 1.5], new THREE.Color(0x52b6ff).multiplyScalar(2.4));
  addPanel([0, 8.5, -6], [8, 0.3, 1.5], new THREE.Color(0xd2e9ff).multiplyScalar(2.2));

  // Warm amber/copper industrial bounce strips along horizon
  addPanel([9, 1.5, 4], [0.4, 3.2, 8], new THREE.Color(0xffb978).multiplyScalar(2.0));
  addPanel([-9, 2.2, -3], [0.4, 3.6, 8], new THREE.Color(0x66c2ff).multiplyScalar(2.3));

  // Subtle floor cyan glow reflection
  addPanel([0, -7, 0], [10, 0.3, 10], new THREE.Color(0x13324a).multiplyScalar(1.4));

  const envRT = pmrem.fromScene(envScene, 0.035);
  pmrem.dispose();
  return envRT.texture;
}

async function init() {
  const canvas = document.getElementById('viewport');
  const params = new URLSearchParams(window.location.search);
  const forceWebGL = params.get('backend') === 'webgl';

  // 1. Initialize Three.js WebGPU Renderer (auto-falls back to WebGL2 if unavailable)
  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: true,
    forceWebGL,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;

  await renderer.init();

  const isWebGPU = Boolean(renderer.backend && renderer.backend.isWebGPUBackend);
  let adapterInfo = '';
  if (isWebGPU && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      const info = adapter?.info;
      if (info) adapterInfo = [info.vendor, info.architecture, info.description].filter(Boolean).join(' / ');
    } catch {
      // Optional adapter info
    }
  }

  // 2. Scene, Camera & Controls
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x04070e);
  scene.fog = new THREE.FogExp2(0x04070e, 0.032);
  scene.environment = buildStudioEnvironment(renderer);
  scene.environmentIntensity = 0.92;

  const camera = new THREE.PerspectiveCamera(44, window.innerWidth / window.innerHeight, 0.08, 120);
  camera.position.copy(CAMERA_PRESETS.hero.pos);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 1.8;
  controls.maxDistance = 18.0;
  controls.maxPolarAngle = Math.PI * 0.56;
  controls.target.copy(CAMERA_PRESETS.hero.target);
  controls.update();

  // 3. Dramatic Studio & Dynamic Plasma Core Lighting
  const hemiLight = new THREE.HemisphereLight(0x88c8ff, 0x101828, 0.72);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xeef6ff, 2.4);
  keyLight.position.set(7, 10, 8);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x4cb8ff, 1.85);
  rimLight.position.set(-9, 5, -7);
  scene.add(rimLight);

  const warmFill = new THREE.DirectionalLight(0xffb574, 0.95);
  warmFill.position.set(4, -2, -8);
  scene.add(warmFill);

  // 3 Dynamic Plasma Point Lights placed inside the toroidal plasma channel
  const plasmaLights = [];
  for (let i = 0; i < 3; i++) {
    const phi = (i / 3) * Math.PI * 2 + 0.4;
    const pl = new THREE.PointLight(0x30dcff, 16, 11, 1.8);
    pl.position.set(Math.cos(phi) * TOKAMAK.R0, 0.05, Math.sin(phi) * TOKAMAK.R0);
    scene.add(pl);
    plasmaLights.push({ light: pl, basePhi: phi });
  }

  // 4. Build Mechanical Tokamak Reactor & WebGPU Compute Plasma System
  const tokamak = buildTokamak();
  scene.add(tokamak.root);

  const plasma = new PlasmaSystem(renderer, isWebGPU, 65536);
  scene.add(plasma.root);
  await plasma.initParticles(65536);

  // 5. WebGPU Node RenderPipeline with Selective HDR Bloom
  const renderPipeline = new THREE.RenderPipeline(renderer);
  const scenePass = pass(scene, camera);
  const scenePassColor = scenePass.getTextureNode('output');
  const bloomPass = bloom(scenePassColor, 0.36, 0.40, 0.78);
  renderPipeline.outputNode = scenePassColor.add(bloomPass);

  // 6. Reactive Simulation State
  const initMode = REACTOR_MODES.ignition;
  const state = {
    mode: 'ignition',
    cutaway: 0.42,
    bField: initMode.bField,
    heating: initMode.heating,
    qFactor: initMode.qFactor,
    plasmaIntensity: initMode.plasmaIntensity,
    coilEnergy: initMode.coilEnergy,
    divertorHeat: initMode.divertorHeat,
    nbiPower: initMode.nbiPower,
    instability: initMode.instability,
    cryoTone: initMode.cryoTone,
    elmBurst: 0.0,
    elmFlash: 0.0,
    nbiBoost: 0.0,
    showFlux: true,
    showCallouts: true,
    autoOrbit: true,
    activeCam: 'hero',
    pointerWorld: new THREE.Vector3(0, 99, 0),
    pointerStrength: 0.0,
    target: {
      cutaway: 0.42,
      bField: initMode.bField,
      heating: initMode.heating,
      qFactor: initMode.qFactor,
      plasmaIntensity: initMode.plasmaIntensity,
      coilEnergy: initMode.coilEnergy,
      divertorHeat: initMode.divertorHeat,
      nbiPower: initMode.nbiPower,
      instability: initMode.instability,
      cryoTone: initMode.cryoTone,
    },
  };

  // Camera transition controller
  let camTransition = null;
  let rideAngle = 0.0;

  const selectCamera = (presetKey) => {
    state.activeCam = presetKey;
    const preset = CAMERA_PRESETS[presetKey];
    if (!preset) return;
    if (preset.dynamic) {
      camTransition = null;
      controls.enabled = false;
    } else {
      controls.enabled = true;
      camTransition = {
        fromPos: camera.position.clone(),
        toPos: preset.pos.clone(),
        fromTarget: controls.target.clone(),
        toTarget: preset.target.clone(),
        t: 0,
        duration: 1.05,
      };
    }
  };

  // Cancel scripted transition when user manually drags/zooms
  controls.addEventListener('start', () => {
    camTransition = null;
  });

  // 7. Interactive 3D Pointer Perturbation on Midplane (y = 0)
  const raycaster = new THREE.Raycaster();
  const pointerNDC = new THREE.Vector2();
  const midPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const hitPoint = new THREE.Vector3();
  let pointerTargetStrength = 0;

  canvas.addEventListener('pointermove', (e) => {
    pointerNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
    pointerNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(pointerNDC, camera);
    if (raycaster.ray.intersectPlane(midPlane, hitPoint)) {
      const radialDist = Math.hypot(hitPoint.x, hitPoint.z);
      if (Math.abs(radialDist - TOKAMAK.R0) < 1.35) {
        state.pointerWorld.copy(hitPoint);
        pointerTargetStrength = 1.0;
        return;
      }
    }
    pointerTargetStrength = 0.0;
  });

  canvas.addEventListener('pointerleave', () => {
    pointerTargetStrength = 0.0;
  });

  // 8. Initialize HUD Controller
  const hud = new HudController(state, {
    onSelectCamera: selectCamera,
    onParticleCountChange: async (count) => {
      await plasma.initParticles(count);
    },
    onToggleBackend: () => {
      const next = isWebGPU ? '?backend=webgl' : '';
      window.location.href = window.location.pathname + next;
    },
  });
  hud.setBackendLabel(isWebGPU, adapterInfo);

  // Resize handler
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // Expose lightweight debug hook for automated inspection
  window.__HELIOS__ = {
    isWebGPU,
    adapterInfo,
    state,
    applyMode: (m) => hud.applyMode(m),
    selectCamera,
    triggerElm: () => {
      state.elmBurst = 1.35;
      state.elmFlash = 1.0;
    },
    setCutaway: (v) => {
      state.target.cutaway = v;
      hud.syncSlidersFromState();
    },
  };

  // 9. Main Render & Compute Loop
  let lastTime = performance.now() * 0.001;
  let elapsed = 0;
  const rideLook = new THREE.Vector3();
  const ridePos = new THREE.Vector3();

  renderer.setAnimationLoop(() => {
    const now = performance.now() * 0.001;
    const dt = Math.min(0.05, Math.max(0.001, now - lastTime));
    lastTime = now;
    elapsed += dt;

    // Smoothly interpolate reactor parameters toward targets
    const smooth = 1 - Math.exp(-dt * 4.5);
    const lerpField = (k) => {
      state[k] += (state.target[k] - state[k]) * smooth;
    };
    lerpField('cutaway');
    lerpField('bField');
    lerpField('heating');
    lerpField('qFactor');
    lerpField('plasmaIntensity');
    lerpField('coilEnergy');
    lerpField('divertorHeat');
    lerpField('instability');
    lerpField('cryoTone');

    // Decay transient pulses (ELM burst & NBI supercharge)
    const targetNbi = state.target.nbiPower + state.nbiBoost;
    state.nbiPower += (targetNbi - state.nbiPower) * (1 - Math.exp(-dt * 6.0));
    state.nbiBoost *= Math.exp(-dt * 1.4);

    // Sustained overdrive periodically spawns mini-ELM flares
    if (state.mode === 'overdrive') {
      const autoPulse = Math.max(0, Math.sin(elapsed * 2.6) * 0.45);
      state.elmBurst = Math.max(state.elmBurst * Math.exp(-dt * 0.9), autoPulse);
      state.elmFlash = Math.max(state.elmFlash * Math.exp(-dt * 1.4), autoPulse * 0.7);
    } else {
      state.elmBurst *= Math.exp(-dt * 0.85);
      state.elmFlash *= Math.exp(-dt * 1.35);
    }

    state.pointerStrength += (pointerTargetStrength - state.pointerStrength) * (1 - Math.exp(-dt * 8.0));

    // Dynamic plasma core illumination lighting the surrounding machinery (balanced to avoid washing out solenoid)
    const plasmaLum = state.plasmaIntensity * (1.0 + state.elmFlash * 0.65);
    for (let i = 0; i < plasmaLights.length; i++) {
      const entry = plasmaLights[i];
      const phi = entry.basePhi + elapsed * 0.35;
      entry.light.position.set(
        Math.cos(phi) * (TOKAMAK.R0 + 0.35),
        Math.sin(elapsed * 2.4 + i) * 0.25,
        Math.sin(phi) * (TOKAMAK.R0 + 0.35)
      );
      entry.light.intensity = plasmaLum * (9.5 + Math.sin(elapsed * 6.5 + i * 2.1) * 2.0);
      entry.light.color.setHSL(
        0.54 - state.elmFlash * 0.44,
        0.92,
        0.56 + state.elmFlash * 0.1
      );
    }

    // Camera animation: either Plasma Ride, Preset Transition, or OrbitControls
    if (state.activeCam === 'ride') {
      rideAngle += dt * 0.42;
      millerPoint(rideAngle, Math.sin(rideAngle * 2.0) * 0.35, 0.92, ridePos);
      millerPoint(rideAngle + 0.38, Math.sin((rideAngle + 0.38) * 2.0) * 0.25, 0.35, rideLook);
      camera.position.lerp(ridePos, 1 - Math.exp(-dt * 6.0));
      camera.lookAt(rideLook);
    } else if (camTransition) {
      camTransition.t += dt / camTransition.duration;
      const u = Math.min(1, camTransition.t);
      const ease = 1 - Math.pow(1 - u, 3);
      camera.position.lerpVectors(camTransition.fromPos, camTransition.toPos, ease);
      controls.target.lerpVectors(camTransition.fromTarget, camTransition.toTarget, ease);
      if (u >= 1) camTransition = null;
      controls.update();
    } else {
      controls.autoRotate = state.autoOrbit;
      controls.autoRotateSpeed = 0.52;
      controls.update();
    }

    // Update reactor mechanics & WebGPU compute plasma
    tokamak.update(dt, elapsed, state);
    plasma.update(dt, elapsed, state);
    hud.update(dt, camera);

    // Render frame through TSL RenderPipeline + Bloom
    renderPipeline.render();
  });
}

init().catch((err) => {
  console.error('HELIOS-IX initialization error:', err);
  const badge = document.getElementById('backend-value');
  if (badge) badge.textContent = 'INIT ERROR';
});
