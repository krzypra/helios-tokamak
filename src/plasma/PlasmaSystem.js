import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  uniform,
  instancedArray,
  instancedBufferAttribute,
  instanceIndex,
  float,
  vec2,
  vec3,
  vec4,
  color,
  mix,
  sin,
  cos,
  abs,
  pow,
  exp,
  clamp,
  smoothstep,
  length,
  dot,
  hash,
  mod,
  uv,
  positionLocal,
  normalView,
  positionViewDirection,
  time,
} from 'three/tsl';
import { TOKAMAK, millerCrossSection, millerPoint } from '../reactor/tokamakMath.js';

/**
 * Create a closed D-shaped Miller torus BufferGeometry for the volumetric plasma mantle.
 */
function createPlasmaTorusGeometry(rho = 0.88, phiSegments = 96, thetaSegments = 48) {
  const positions = [];
  const uvs = [];
  const indices = [];

  for (let i = 0; i <= phiSegments; i++) {
    const u = i / phiSegments;
    const phi = u * Math.PI * 2;
    const cosP = Math.cos(phi);
    const sinP = Math.sin(phi);

    for (let j = 0; j <= thetaSegments; j++) {
      const v = j / thetaSegments;
      const theta = v * Math.PI * 2;
      const cs = millerCrossSection(theta, rho);
      positions.push(cs.r * cosP, cs.y, cs.r * sinP);
      uvs.push(u, v);
    }
  }

  const stride = thetaSegments + 1;
  for (let i = 0; i < phiSegments; i++) {
    for (let j = 0; j < thetaSegments; j++) {
      const a = i * stride + j;
      const b = (i + 1) * stride + j;
      const c = (i + 1) * stride + (j + 1);
      const d = i * stride + (j + 1);
      indices.push(a, b, c, a, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * PlasmaSystem:
 *  - Real WebGPU Compute Shader ion simulation (65,536 - 131,072 particles)
 *    with automatic WebGL2 TSL GPU fallback.
 *  - Layered volumetric TSL helical plasma core shells.
 *  - Dynamic helical magnetic flux field cage reflecting safety factor q95.
 */
export class PlasmaSystem {
  constructor(renderer, isWebGPU, initialCount = 65536) {
    this.renderer = renderer;
    this.isWebGPU = isWebGPU;
    this.particleCount = initialCount;
    this.root = new THREE.Group();
    this.root.name = 'HeliosPlasmaSystem';

    // Shared reactive TSL uniforms
    this.uniforms = {
      uDelta: uniform(0.016),
      uTime: uniform(0.0),
      uBField: uniform(11.8),          // Toroidal magnetic field (Tesla)
      uHeating: uniform(0.64),         // Normalized heating power [0..1]
      uQFactor: uniform(3.1),          // Safety factor q95 (helical twist)
      uPlasmaIntensity: uniform(1.0),  // Master plasma ignition visibility [0..1.4]
      uInstability: uniform(0.18),     // MHD ballooning / helical ripple amplitude
      uElmBurst: uniform(0.0),         // Transient ELM strike burst [0..1.5]
      uPointerPos: uniform(new THREE.Vector3(0, 99, 0)),
      uPointerStrength: uniform(0.0),
      uFluxVisible: uniform(1.0),
    };

    this.particleObject = null;
    this.computeInit = null;
    this.computeUpdate = null;
    this.lastQBuilt = -1;

    this._buildVolumetricCore();
    this._buildFluxCage(3.1);
  }

  /**
   * Initialize or rebuild the GPU particle simulation for a given particle count.
   */
  async initParticles(count = this.particleCount) {
    this.particleCount = count;

    if (this.particleObject) {
      this.root.remove(this.particleObject);
      this.particleObject.material?.dispose?.();
      this.particleObject.geometry?.dispose?.();
      this.particleObject = null;
    }

    if (this.isWebGPU) {
      await this._buildWebGPUComputeParticles(count);
    } else {
      this._buildWebGLFallbackParticles(count);
    }
  }

  /**
   * Genuine WebGPU Compute Shader particle engine (runs WGSL compute dispatches every frame).
   */
  async _buildWebGPUComputeParticles(count) {
    // Storage buffers in GPU VRAM:
    // stateBuf: (phi, theta, rho, seed)
    // posBuf:   (x, y, z, energyTemp)
    const stateBuf = instancedArray(count, 'vec4');
    const posBuf = instancedArray(count, 'vec4');

    const R0 = float(TOKAMAK.R0);
    const aMinor = float(TOKAMAK.a);
    const kappa = float(TOKAMAK.kappa);
    const deltaTri = float(TOKAMAK.delta);
    const TWO_PI = float(Math.PI * 2);

    const {
      uDelta,
      uTime,
      uBField,
      uHeating,
      uQFactor,
      uPlasmaIntensity,
      uInstability,
      uElmBurst,
      uPointerPos,
      uPointerStrength,
    } = this.uniforms;

    // 1. Compute Initialization Kernel
    this.computeInit = Fn(() => {
      const idx = instanceIndex;
      const h1 = hash(idx.add(11));
      const h2 = hash(idx.add(1013));
      const h3 = hash(idx.add(9173));
      const h4 = hash(idx.add(31337));

      const phi = h1.mul(TWO_PI);
      const theta = h2.mul(TWO_PI);
      // Denser plasma toward magnetic axis, sparse in scrape-off layer
      const rho = pow(h3, float(0.68)).mul(0.98).add(0.03);
      const seed = h4;

      stateBuf.element(idx).assign(vec4(phi, theta, rho, seed));

      const r = R0.add(rho.mul(aMinor).mul(cos(theta.add(deltaTri.mul(sin(theta))))));
      const y = rho.mul(kappa).mul(aMinor).mul(sin(theta));
      const px = r.mul(cos(phi));
      const pz = r.mul(sin(phi));
      const temp = float(1.0).sub(rho).mul(0.85).add(seed.mul(0.15));

      posBuf.element(idx).assign(vec4(px, y, pz, temp));
    })().compute(count);

    // 2. Per-Frame Helical Magnetic Confinement + MHD / ELM Compute Kernel
    this.computeUpdate = Fn(() => {
      const idx = instanceIndex;
      const st = stateBuf.element(idx);
      const phi = st.x.toVar();
      const theta = st.y.toVar();
      const rho = st.z.toVar();
      const seed = st.w;

      // Local safety factor profile q(rho) = 1.05 + (q95 - 1.05) * rho^2
      const qLocal = float(1.05).add(uQFactor.sub(1.05).mul(rho).mul(rho));

      // Toroidal angular velocity scales with magnetic field B0, heating, and hot core profile
      const coreSpeedBoost = float(1.45).sub(rho.mul(0.65));
      const baseOmega = uBField.mul(0.11).add(uHeating.mul(1.35)).mul(coreSpeedBoost);
      const ionJitter = float(0.76).add(seed.mul(0.48));
      const dPhi = baseOmega.mul(ionJitter).mul(uDelta);

      // Helical magnetic field line following: dTheta = dPhi / q(rho)
      const dTheta = dPhi.div(qLocal);

      phi.assign(mod(phi.add(dPhi), TWO_PI));
      theta.assign(mod(theta.add(dTheta), TWO_PI));

      // Edge-Localized Mode (ELM) filament ejection toward the tungsten divertor X-point
      const isEdge = smoothstep(0.76, 0.94, rho);
      const elmKick = isEdge.mul(uElmBurst).mul(sin(phi.mul(4.0).add(uTime.mul(6.0)).add(seed.mul(6.28))).mul(0.5).add(0.5));
      const nextRho = rho.add(elmKick.mul(uDelta).mul(0.75)).toVar();

      // Relax perturbed ions back toward confined flux surface or recycle ejected ions
      const targetRho = pow(seed, float(0.68)).mul(0.96).add(0.04);
      nextRho.assign(mix(nextRho, targetRho, uDelta.mul(0.45)));

      If(nextRho.greaterThan(1.24), () => {
        nextRho.assign(float(0.08).add(seed.mul(0.45)));
      });

      stateBuf.element(idx).assign(vec4(phi, theta, nextRho, seed));

      // MHD helical ballooning wave displacement
      const mhdWave = sin(theta.mul(3.0).sub(phi.mul(2.0)).add(uTime.mul(4.8)).add(seed.mul(3.14)))
        .mul(uInstability.add(uElmBurst.mul(0.32)))
        .mul(nextRho)
        .mul(0.16);

      const effRho = clamp(nextRho.add(mhdWave), 0.02, 1.28);

      // Miller equilibrium D-section mapping
      const rCross = R0.add(effRho.mul(aMinor).mul(cos(theta.add(deltaTri.mul(sin(theta))))));
      const yCross = effRho.mul(kappa).mul(aMinor).mul(sin(theta)).toVar();

      // Pull outer ELM-ejected ions downward toward the bottom tungsten divertor (y ~ -1.38)
      const divertorPull = smoothstep(1.0, 1.22, effRho).mul(0.78);
      yCross.assign(mix(yCross, float(-1.34), divertorPull));

      const px = rCross.mul(cos(phi)).toVar();
      const pz = rCross.mul(sin(phi)).toVar();

      // Interactive 3D pointer magnetic field perturbation
      const pVec = vec3(px, yCross, pz);
      const toPointer = pVec.sub(uPointerPos);
      const pDist = length(toPointer);
      const pointerWave = exp(pDist.mul(pDist).mul(-1.35)).mul(uPointerStrength);
      const swirlOffset = vec3(
        sin(uTime.mul(9.0).add(phi.mul(6.0))).mul(0.24),
        cos(uTime.mul(8.0).add(theta.mul(5.0))).mul(0.24),
        sin(uTime.mul(7.0).sub(phi.mul(5.0))).mul(0.24)
      ).mul(pointerWave);

      px.addAssign(swirlOffset.x);
      yCross.addAssign(swirlOffset.y);
      pz.addAssign(swirlOffset.z);

      // Compute normalized particle temperature/energy:
      // 0.0..0.4 = magenta/violet edge, 0.4..0.85 = electric cyan, 0.85..1.1 = incandescent core, >1.1 = ELM orange strike
      const coreTemp = float(1.0).sub(clamp(effRho, 0.0, 1.0)).mul(0.78).add(uHeating.mul(0.28)).add(pointerWave.mul(0.35));
      const strikeTemp = float(1.22).add(elmKick.mul(0.35));
      const finalTemp = mix(coreTemp, strikeTemp, clamp(divertorPull.mul(1.6).add(elmKick.mul(0.55)), 0.0, 1.0));

      posBuf.element(idx).assign(vec4(px, yCross, pz, finalTemp));
    })().compute(count);

    await this.renderer.computeAsync(this.computeInit);

    // 3. Sprite Node Material reading directly from WebGPU compute posBuf
    const posAttr = posBuf.toAttribute();
    const spriteMat = this._createParticleSpriteMaterial(posAttr.xyz, posAttr.w);

    const sprite = new THREE.Sprite(spriteMat);
    sprite.count = count;
    sprite.frustumCulled = false;
    sprite.name = 'WebGPUComputePlasmaParticles';
    this.root.add(sprite);
    this.particleObject = sprite;
  }

  /**
   * Automatic WebGL2 TSL fallback (pure GPU analytic helical trajectories, zero CPU particle loop).
   */
  _buildWebGLFallbackParticles(count) {
    this.computeInit = null;
    this.computeUpdate = null;

    const seeds = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      seeds[i * 4 + 0] = Math.random() * Math.PI * 2;
      seeds[i * 4 + 1] = Math.random() * Math.PI * 2;
      seeds[i * 4 + 2] = Math.pow(Math.random(), 0.68) * 0.96 + 0.04;
      seeds[i * 4 + 3] = Math.random();
    }

    const seedAttr = new THREE.InstancedBufferAttribute(seeds, 4);
    const seedNode = instancedBufferAttribute(seedAttr);

    const { uTime, uBField, uHeating, uQFactor, uInstability, uElmBurst } = this.uniforms;

    const phi0 = seedNode.x;
    const theta0 = seedNode.y;
    const rho0 = seedNode.z;
    const rand = seedNode.w;

    const qLocal = float(1.05).add(uQFactor.sub(1.05).mul(rho0).mul(rho0));
    const speed = uBField.mul(0.11).add(uHeating.mul(1.35)).mul(float(1.4).sub(rho0.mul(0.6))).mul(float(0.75).add(rand.mul(0.5)));
    const phi = phi0.add(uTime.mul(speed));
    const theta = theta0.add(uTime.mul(speed).div(qLocal));

    const mhd = sin(theta.mul(3.0).sub(phi.mul(2.0)).add(uTime.mul(4.5)))
      .mul(uInstability.add(uElmBurst.mul(0.3)))
      .mul(rho0)
      .mul(0.15);
    const effRho = clamp(rho0.add(mhd), 0.03, 1.18);

    const rCross = float(TOKAMAK.R0).add(effRho.mul(TOKAMAK.a).mul(cos(theta.add(float(TOKAMAK.delta).mul(sin(theta))))));
    const yCross = effRho.mul(TOKAMAK.kappa * TOKAMAK.a).mul(sin(theta));
    const posNode = vec3(rCross.mul(cos(phi)), yCross, rCross.mul(sin(phi)));
    const tempNode = float(1.0).sub(clamp(effRho, 0.0, 1.0)).mul(0.8).add(uHeating.mul(0.25)).add(uElmBurst.mul(smoothstep(0.8, 1.0, rho0)).mul(0.45));

    const spriteMat = this._createParticleSpriteMaterial(posNode, tempNode);
    const sprite = new THREE.Sprite(spriteMat);
    sprite.count = count;
    sprite.frustumCulled = false;
    sprite.name = 'WebGLFallbackPlasmaParticles';
    this.root.add(sprite);
    this.particleObject = sprite;
  }

  /**
   * Shared TSL SpriteNodeMaterial mapping particle temperature to physically inspired plasma emission.
   */
  _createParticleSpriteMaterial(posNode, tempNode) {
    const { uPlasmaIntensity, uElmBurst } = this.uniforms;

    const mat = new THREE.SpriteNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    mat.positionNode = posNode;

    // Slight dynamic size boost for hot core & ELM strike ions
    const baseSize = mix(float(0.032), float(0.058), smoothstep(0.50, 1.25, tempNode));
    mat.scaleNode = baseSize.mul(clamp(uPlasmaIntensity, 0.05, 1.32));

    // Radial crisp glowing ion core inside the billboard quad
    const centered = uv().sub(0.5).mul(2.0);
    const r2 = dot(centered, centered);
    const halo = exp(r2.mul(-4.6)).mul(smoothstep(1.0, 0.10, r2));

    // Rich plasma physics emission spectrum: electric violet -> magenta -> azure cyan -> golden core -> fiery ELM
    const colEdge = color(0x8c1dff);
    const colMagenta = color(0xf736d4);
    const colCyan = color(0x00d8ff);
    const colCore = color(0xfff0c2);
    const colElm = color(0xff5a12);

    const c1 = mix(colEdge, colMagenta, smoothstep(0.05, 0.30, tempNode));
    const c2 = mix(c1, colCyan, smoothstep(0.30, 0.68, tempNode));
    const c3 = mix(c2, colCore, smoothstep(0.76, 1.05, tempNode));
    const finalCol = mix(c3, colElm, smoothstep(1.08, 1.30, tempNode));

    const brightness = float(0.95).add(uElmBurst.mul(0.35)).mul(uPlasmaIntensity);
    mat.colorNode = finalCol.mul(halo).mul(brightness);
    mat.opacityNode = halo.mul(clamp(uPlasmaIntensity.mul(0.62), 0.0, 0.78));

    return mat;
  }

  /**
   * Layered volumetric translucent TSL plasma shells inside the vacuum vessel.
   */
  _buildVolumetricCore() {
    const { uTime, uHeating, uPlasmaIntensity, uInstability, uElmBurst, uQFactor } = this.uniforms;

    // Outer plasma mantle (aurora-like magenta/cyan Fresnel sheath with helical filaments)
    const outerGeo = createPlasmaTorusGeometry(0.86, 96, 48);
    const outerMat = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    {
      const fresnel = pow(float(1.0).sub(abs(dot(normalView, positionViewDirection))), 1.75);
      const u = uv().x.mul(Math.PI * 2);
      const v = uv().y.mul(Math.PI * 2);

      // Multi-band helical plasma filaments matching safety factor q
      const helix1 = sin(v.mul(3.0).sub(u.mul(uQFactor.mul(2.0))).add(uTime.mul(3.8))).mul(0.5).add(0.5);
      const helix2 = sin(v.mul(5.0).sub(u.mul(uQFactor.mul(3.0))).sub(uTime.mul(2.4))).mul(0.5).add(0.5);
      const turbulence = helix1.mul(0.65).add(helix2.mul(0.35));

      const edgeCol = mix(color(0x9a20ff), color(0x00d8ff), turbulence);
      const hotCol = mix(edgeCol, color(0xff7e30), uElmBurst.mul(0.62));
      const wave = float(0.68).add(turbulence.mul(0.58)).add(uInstability.mul(0.26));

      outerMat.colorNode = hotCol.mul(fresnel.mul(0.85).add(0.16)).mul(wave).mul(uPlasmaIntensity.mul(0.74));
      outerMat.opacityNode = fresnel.mul(0.30).add(0.05).mul(clamp(uPlasmaIntensity, 0.0, 1.0));
    }

    this.outerMantle = new THREE.Mesh(outerGeo, outerMat);
    this.outerMantle.name = 'PlasmaOuterMantle';
    this.root.add(this.outerMantle);

    // Inner incandescent ignition core (compact hot torus axis)
    const innerGeo = createPlasmaTorusGeometry(0.44, 84, 36);
    const innerMat = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.FrontSide,
    });

    {
      const viewDot = abs(dot(normalView, positionViewDirection));
      const coreGlow = pow(viewDot, 1.45);
      const u = uv().x.mul(Math.PI * 2);
      const v = uv().y.mul(Math.PI * 2);
      const ripple = sin(u.mul(6.0).sub(v.mul(2.0)).add(uTime.mul(5.5))).mul(0.24).add(0.86);

      const colCyan = color(0x2ce4ff);
      const colWhiteGold = color(0xffefc4);
      const coreCol = mix(colCyan, colWhiteGold, clamp(uHeating.mul(0.88).add(uElmBurst.mul(0.42)), 0.0, 1.0));

      innerMat.colorNode = coreCol.mul(coreGlow.mul(1.05).add(0.18)).mul(ripple).mul(uPlasmaIntensity.mul(0.82));
      innerMat.opacityNode = coreGlow.mul(0.28).mul(clamp(uPlasmaIntensity, 0.0, 1.0));
    }

    this.innerCore = new THREE.Mesh(innerGeo, innerMat);
    this.innerCore.name = 'PlasmaIgnitionCore';
    this.root.add(this.innerCore);
  }

  /**
   * Build glowing helical magnetic flux cage wrapping the plasma separatrix.
   * Rebuilds cleanly whenever safety factor q95 changes appreciably.
   */
  _buildFluxCage(qVal) {
    if (this.fluxGroup) {
      this.root.remove(this.fluxGroup);
      this.fluxGroup.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
      });
    }

    this.lastQBuilt = qVal;
    this.fluxGroup = new THREE.Group();
    this.fluxGroup.name = 'MagneticFluxCage';

    const { uTime, uFluxVisible, uPlasmaIntensity, uElmBurst } = this.uniforms;

    const fluxMat = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    {
      const pulse = pow(sin(uv().x.mul(48.0).sub(uTime.mul(6.5))).mul(0.5).add(0.5), 3.0);
      const baseCol = mix(color(0x1fd8ff), color(0xff8e42), uElmBurst.mul(0.75));
      const intensity = float(0.22).add(pulse.mul(0.95)).mul(uFluxVisible).mul(uPlasmaIntensity.mul(0.55).add(0.45));
      fluxMat.colorNode = baseCol.mul(intensity);
      fluxMat.opacityNode = float(0.18).add(pulse.mul(0.45)).mul(uFluxVisible);
    }

    // 6 interleaved helical field lines tracing real dTheta = dPhi / q along rho = 0.99
    const strandCount = 6;
    const turns = 2;
    const steps = 240;

    for (let s = 0; s < strandCount; s++) {
      const pts = [];
      const thetaOffset = (s / strandCount) * Math.PI * 2;
      for (let i = 0; i <= steps; i++) {
        const f = i / steps;
        const phi = f * Math.PI * 2 * turns;
        const theta = thetaOffset + phi / qVal;
        pts.push(millerPoint(phi, theta, 0.99));
      }
      const curve = new THREE.CatmullRomCurve3(pts, false);
      const tubeGeo = new THREE.TubeGeometry(curve, 220, 0.012, 5, false);
      const mesh = new THREE.Mesh(tubeGeo, fluxMat);
      this.fluxGroup.add(mesh);
    }

    this.root.add(this.fluxGroup);
  }

  /**
   * Update uniforms and run WebGPU compute pass every animation frame.
   */
  update(dt, elapsed, state) {
    const safeDt = Math.min(dt, 0.05);
    this.uniforms.uDelta.value = safeDt;
    this.uniforms.uTime.value = elapsed;
    this.uniforms.uBField.value = state.bField;
    this.uniforms.uHeating.value = state.heating / 150;
    this.uniforms.uQFactor.value = state.qFactor;
    this.uniforms.uPlasmaIntensity.value = state.plasmaIntensity;
    this.uniforms.uInstability.value = state.instability;
    this.uniforms.uElmBurst.value = state.elmBurst;
    this.uniforms.uFluxVisible.value = state.showFlux ? 1.0 : 0.0;
    this.uniforms.uPointerPos.value.copy(state.pointerWorld);
    this.uniforms.uPointerStrength.value = state.pointerStrength;

    // Rebuild helical magnetic flux cage if safety factor q changed significantly
    if (Math.abs(state.qFactor - this.lastQBuilt) > 0.14) {
      this._buildFluxCage(state.qFactor);
    }

    this.fluxGroup.visible = state.showFlux;
    const showPlasma = state.plasmaIntensity > 0.015;
    this.outerMantle.visible = showPlasma;
    this.innerCore.visible = showPlasma;
    if (this.particleObject) {
      this.particleObject.visible = showPlasma;
    }

    // Dispatch real WebGPU Compute Shader on GPU
    if (this.isWebGPU && this.computeUpdate && showPlasma) {
      this.renderer.compute(this.computeUpdate);
    }
  }
}
