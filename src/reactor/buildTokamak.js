import * as THREE from 'three/webgpu';
import {
  uniform,
  float,
  vec2,
  color,
  mix,
  sin,
  cos,
  abs,
  fract,
  pow,
  clamp,
  smoothstep,
  length,
  positionLocal,
  positionWorld,
  uv,
  time,
} from 'three/tsl';
import { TOKAMAK, millerCrossSection, sectorCutawayMask } from './tokamakMath.js';

/**
 * Create a closed 2D D-shaped ring ExtrudeGeometry (outer D minus inner D hole)
 * positioned in the XY plane so local X = R (major radius) and local Y = height.
 */
function createDCoilGeometry(rhoInner, rhoOuter, depth, bevel = 0.028, segments = 64) {
  const outerShape = new THREE.Shape();
  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const p = millerCrossSection(theta, rhoOuter);
    if (i === 0) outerShape.moveTo(p.r, p.y);
    else outerShape.lineTo(p.r, p.y);
  }

  const innerHole = new THREE.Path();
  for (let i = segments; i >= 0; i--) {
    const theta = (i / segments) * Math.PI * 2;
    const p = millerCrossSection(theta, rhoInner);
    if (i === segments) innerHole.moveTo(p.r, p.y);
    else innerHole.lineTo(p.r, p.y);
  }
  outerShape.holes.push(innerHole);

  const geo = new THREE.ExtrudeGeometry(outerShape, {
    depth,
    bevelEnabled: bevel > 0,
    bevelThickness: bevel,
    bevelSize: bevel * 0.85,
    bevelSegments: 2,
    curveSegments: 1,
  });
  geo.translate(0, 0, -depth * 0.5);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Create a solid double-walled D-torus vacuum vessel sector spanning [phiStart, phiEnd].
 * Includes inner first-wall, outer shell, and capped cutaway end-walls so cross-sections look solid.
 */
function createVesselSectorGeometry(phiStart, phiEnd, rhoInner = 1.12, rhoOuter = 1.24, phiSteps = 5, thetaSteps = 44) {
  const positions = [];
  const uvs = [];
  const indices = [];

  const pushQuad = (a, b, c, d, ua, ub, uc, ud) => {
    const base = positions.length / 3;
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, d.x, d.y, d.z);
    uvs.push(ua[0], ua[1], ub[0], ub[1], uc[0], uc[1], ud[0], ud[1]);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  };

  const pt = (phi, theta, rho) => {
    const cs = millerCrossSection(theta, rho);
    return new THREE.Vector3(cs.r * Math.cos(phi), cs.y, cs.r * Math.sin(phi));
  };

  const thetaMin = -Math.PI * 0.42;
  const thetaMax = Math.PI * 1.42;

  for (let i = 0; i < phiSteps; i++) {
    const f0 = i / phiSteps;
    const f1 = (i + 1) / phiSteps;
    const p0 = phiStart + (phiEnd - phiStart) * f0;
    const p1 = phiStart + (phiEnd - phiStart) * f1;

    for (let j = 0; j < thetaSteps; j++) {
      const t0 = thetaMin + (thetaMax - thetaMin) * (j / thetaSteps);
      const t1 = thetaMin + (thetaMax - thetaMin) * ((j + 1) / thetaSteps);
      const v0 = j / thetaSteps;
      const v1 = (j + 1) / thetaSteps;

      const o00 = pt(p0, t0, rhoOuter);
      const o10 = pt(p1, t0, rhoOuter);
      const o11 = pt(p1, t1, rhoOuter);
      const o01 = pt(p0, t1, rhoOuter);
      pushQuad(o00, o01, o11, o10, [f0, v0], [f0, v1], [f1, v1], [f1, v0]);

      const i00 = pt(p0, t0, rhoInner);
      const i10 = pt(p1, t0, rhoInner);
      const i11 = pt(p1, t1, rhoInner);
      const i01 = pt(p0, t1, rhoInner);
      pushQuad(i00, i10, i11, i01, [f0, v0], [f1, v0], [f1, v1], [f0, v1]);
    }
  }

  for (let j = 0; j < thetaSteps; j++) {
    const t0 = thetaMin + (thetaMax - thetaMin) * (j / thetaSteps);
    const t1 = thetaMin + (thetaMax - thetaMin) * ((j + 1) / thetaSteps);

    pushQuad(
      pt(phiStart, t0, rhoInner),
      pt(phiStart, t1, rhoInner),
      pt(phiStart, t1, rhoOuter),
      pt(phiStart, t0, rhoOuter),
      [0, 0], [0, 1], [1, 1], [1, 0]
    );
    pushQuad(
      pt(phiEnd, t0, rhoOuter),
      pt(phiEnd, t1, rhoOuter),
      pt(phiEnd, t1, rhoInner),
      pt(phiEnd, t0, rhoInner),
      [0, 0], [0, 1], [1, 1], [1, 0]
    );
  }

  for (let i = 0; i < phiSteps; i++) {
    const p0 = phiStart + (phiEnd - phiStart) * (i / phiSteps);
    const p1 = phiStart + (phiEnd - phiStart) * ((i + 1) / phiSteps);
    pushQuad(
      pt(p0, thetaMin, rhoOuter),
      pt(p1, thetaMin, rhoOuter),
      pt(p1, thetaMin, rhoInner),
      pt(p0, thetaMin, rhoInner),
      [0, 0], [1, 0], [1, 1], [0, 1]
    );
    pushQuad(
      pt(p0, thetaMax, rhoInner),
      pt(p1, thetaMax, rhoInner),
      pt(p1, thetaMax, rhoOuter),
      pt(p0, thetaMax, rhoOuter),
      [0, 0], [1, 0], [1, 1], [0, 1]
    );
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Create a closed-end horizontal torus arc segment (used for split cutaway PF rings).
 */
function createCappedRingArc(radius, tube, startAngle, arcLength, radialSegs = 20, tubularSegs = 64) {
  const group = new THREE.Group();
  const torus = new THREE.TorusGeometry(radius, tube, radialSegs, tubularSegs, arcLength);
  // Rotate from XY plane into XZ horizontal plane:
  // Point (R cos a, R sin a, 0) rotated by +PI/2 around X becomes (R cos a, 0, R sin a)
  torus.rotateX(Math.PI * 0.5);
  torus.rotateY(-startAngle);
  return torus;
}

/**
 * Build the complete procedural HELIOS-IX Tokamak mechanical reactor and reactor hall.
 */
export function buildTokamak() {
  const root = new THREE.Group();
  root.name = 'HeliosTokamakRoot';

  const uniforms = {
    uCoilEnergy: uniform(0.85),
    uDivertorHeat: uniform(0.72),
    uPlasmaGlow: uniform(0.95),
    uNbiPower: uniform(0.8),
    uCryoTone: uniform(0.0),
    uElmFlash: uniform(0.0),
  };

  // ---------------------------------------------------------------------------
  // 1. TSL PROCEDURAL PBR MATERIALS (CLEAN, CRISP AEROSPACE METALS)
  // ---------------------------------------------------------------------------

  // Machined Aerospace Titanium (clean metallic slate with fine UV micro-grain, no zebra banding)
  const titaniumMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.86,
    roughness: 0.30,
  });
  {
    const fineGrain = sin(positionLocal.x.mul(140.0).add(positionLocal.y.mul(140.0))).mul(0.5).add(0.5);
    const seam = smoothstep(0.975, 0.995, abs(sin(uv().x.mul(24.0))));
    titaniumMat.colorNode = mix(color(0x364357), color(0x4b5c74), fineGrain.mul(0.45)).mul(
      float(1.0).sub(seam.mul(0.25))
    );
    titaniumMat.roughnessNode = mix(float(0.24), float(0.38), fineGrain.mul(0.5).add(seam.mul(0.5)));
  }

  // Dark Structural Cryostat Alloy
  const darkAlloyMat = new THREE.MeshStandardNodeMaterial({
    color: 0x1d2635,
    metalness: 0.84,
    roughness: 0.36,
  });

  // Bright Machined Stainless / Nickel Trim (bolts, flanges, clamps, rotors)
  const steelTrimMat = new THREE.MeshStandardNodeMaterial({
    color: 0x9fb3cc,
    metalness: 0.94,
    roughness: 0.18,
  });

  // Superconducting Braided Copper Coils with TSL traveling current pulse
  const copperCoilMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.92,
    roughness: 0.22,
  });
  {
    const braid = sin(uv().x.mul(160.0)).mul(sin(uv().y.mul(80.0))).mul(0.5).add(0.5);
    const baseCopper = mix(color(0xc86b32), color(0xf0a058), braid.mul(0.5));
    copperCoilMat.colorNode = baseCopper;

    // Subtle crisp superconducting flux pulse
    const wavePhase = positionWorld.y.mul(1.8).add(positionWorld.x.mul(1.3)).sub(time.mul(3.2));
    const pulse = pow(sin(wavePhase).mul(0.5).add(0.5), 7.0);
    const pulseColor = mix(color(0x28e4ff), color(0xff8e36), uniforms.uElmFlash.mul(0.85));
    copperCoilMat.emissiveNode = pulseColor.mul(pulse.mul(0.65).add(0.04)).mul(uniforms.uCoilEnergy);
  }

  // First-Wall Beryllium-Tungsten Tiled Armor (crisp beveled tile grid + subtle inner plasma glow)
  const armorMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.86,
    roughness: 0.32,
  });
  {
    const tileU = fract(uv().x.mul(8.0));
    const tileV = fract(uv().y.mul(20.0));
    const seamU = smoothstep(0.0, 0.09, tileU).mul(smoothstep(1.0, 0.91, tileU));
    const seamV = smoothstep(0.0, 0.09, tileV).mul(smoothstep(1.0, 0.91, tileV));
    const tileMask = seamU.mul(seamV);

    armorMat.colorNode = mix(color(0x141a24), color(0x4b5b70), tileMask);
    armorMat.roughnessNode = mix(float(0.62), float(0.26), tileMask);

    const radialDist = length(vec2(positionWorld.x, positionWorld.z)).sub(TOKAMAK.R0);
    const heatShimmer = sin(positionWorld.y.mul(3.5).add(time.mul(2.2))).mul(0.15).add(0.85);
    const innerTint = mix(color(0x2482ff), color(0x52ebff), heatShimmer);
    const plasmaBounce = smoothstep(1.16, 0.72, abs(radialDist)).mul(0.22).mul(uniforms.uPlasmaGlow);
    armorMat.emissiveNode = innerTint.mul(plasmaBounce).mul(tileMask.mul(0.75).add(0.25));
  }

  // Tungsten Divertor Blackbody Thermal Radiation Shader
  const divertorMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.84,
    roughness: 0.36,
  });
  {
    const rDist = length(vec2(positionWorld.x, positionWorld.z));
    // Sharp inner & outer V-target plasma strike zones
    const strikeInner = smoothstep(0.28, 0.0, abs(rDist.sub(2.24)));
    const strikeOuter = smoothstep(0.30, 0.0, abs(rDist.sub(2.76)));
    const shimmer = sin(positionWorld.x.mul(6.0).add(time.mul(4.5)))
      .mul(cos(positionWorld.z.mul(6.0).sub(time.mul(3.8))))
      .mul(0.10)
      .add(0.90);
    const heat = clamp(
      strikeInner.add(strikeOuter).mul(uniforms.uDivertorHeat.add(uniforms.uElmFlash.mul(0.9))).mul(shimmer),
      0.0,
      1.6
    );

    const coldTungsten = color(0x2d3440);
    const cherryRed = color(0xc41c06);
    const moltenOrange = color(0xff6a12);
    const incandescent = color(0xffe7a8);

    const glow1 = mix(color(0x000000), cherryRed, smoothstep(0.06, 0.45, heat));
    const glow2 = mix(glow1, moltenOrange, smoothstep(0.42, 0.92, heat));
    const glow3 = mix(glow2, incandescent, smoothstep(0.90, 1.48, heat));

    divertorMat.colorNode = mix(coldTungsten, color(0x4f3830), clamp(heat, 0.0, 1.0));
    divertorMat.emissiveNode = glow3.mul(float(1.55));
  }

  // Cryogenic & Magnetic Status Strip Material
  const statusStripMat = new THREE.MeshBasicNodeMaterial();
  {
    const pulse = sin(time.mul(2.8).add(positionWorld.y.mul(2.4))).mul(0.22).add(0.78);
    const cryoBlue = color(0x52ceff);
    const ignitionCyan = color(0x28ecff);
    const elmAlert = color(0xff5277);
    const baseCol = mix(ignitionCyan, cryoBlue, uniforms.uCryoTone);
    const finalCol = mix(baseCol, elmAlert, uniforms.uElmFlash);
    statusStripMat.colorNode = finalCol.mul(pulse.mul(1.35).mul(uniforms.uCoilEnergy.mul(0.55).add(0.4)));
  }

  // Sapphire Observation Viewport Glass
  const sapphireGlassMat = new THREE.MeshPhysicalNodeMaterial({
    color: 0x8ae2ff,
    metalness: 0.04,
    roughness: 0.05,
    transparent: true,
    opacity: 0.36,
    clearcoat: 1.0,
    clearcoatRoughness: 0.05,
  });

  // Neutral Beam Particle Cone Material
  const nbiBeamMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });
  {
    const along = uv().y;
    const radial = abs(uv().x.sub(0.5)).mul(2.0);
    const core = pow(float(1.0).sub(radial), 2.1);
    const streak = sin(along.mul(26.0).sub(time.mul(18.0))).mul(0.32).add(0.68);
    const fadeEnds = smoothstep(0.0, 0.14, along).mul(smoothstep(1.0, 0.72, along));
    const beamCol = mix(color(0x22d8ff), color(0xfff4d8), core.mul(0.72));
    nbiBeamMat.colorNode = beamCol.mul(core.mul(streak).mul(fadeEnds).mul(uniforms.uNbiPower).mul(2.1));
    nbiBeamMat.opacityNode = core.mul(fadeEnds).mul(clamp(uniforms.uNbiPower.mul(0.9), 0.0, 0.95));
  }

  // Diagnostic Laser Interferometry Beam Material
  const laserMat = new THREE.MeshBasicNodeMaterial({
    color: 0xff3b7f,
    transparent: true,
    opacity: 0.82,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  // Reactor Hall Industrial Floor (concentric machined pit & containment safety ring)
  const floorMat = new THREE.MeshStandardNodeMaterial({
    metalness: 0.74,
    roughness: 0.42,
  });
  {
    const p = vec2(positionWorld.x, positionWorld.z);
    const r = length(p);
    const ringLines = smoothstep(0.95, 0.992, abs(sin(r.mul(3.8))));
    const radialAngle = p.y.atan(p.x);
    const spokes = smoothstep(0.965, 0.995, abs(sin(radialAngle.mul(16.0))));
    const grid = clamp(ringLines.add(spokes.mul(smoothstep(2.0, 2.5, r))), 0.0, 1.0);
    const pitFade = smoothstep(15.5, 4.5, r);

    floorMat.colorNode = mix(color(0x0a0f19), color(0x1c283c), grid.mul(pitFade));
    floorMat.roughnessNode = mix(float(0.52), float(0.24), grid.mul(pitFade));

    const safetyRing = smoothstep(0.09, 0.0, abs(r.sub(5.55)));
    const innerRing = smoothstep(0.06, 0.0, abs(r.sub(4.05)));
    const ringCol = mix(color(0x24c6ff), color(0xff6b35), uniforms.uElmFlash);
    floorMat.emissiveNode = ringCol.mul(safetyRing.mul(0.75).add(innerRing.mul(0.32))).mul(
      uniforms.uCoilEnergy.mul(0.55).add(0.38)
    );
  }

  // ---------------------------------------------------------------------------
  // 2. CENTRAL SOLENOID COLUMN
  // ---------------------------------------------------------------------------
  const solenoidGroup = new THREE.Group();
  solenoidGroup.name = 'CentralSolenoid';
  root.add(solenoidGroup);

  const coreSpine = new THREE.Mesh(new THREE.CylinderGeometry(0.84, 0.84, 5.8, 36), darkAlloyMat);
  solenoidGroup.add(coreSpine);

  const pancakeCount = 10;
  const pancakeHeight = 0.40;
  const pancakeGap = 0.085;
  const startY = -((pancakeCount - 1) * (pancakeHeight + pancakeGap)) * 0.5;
  const pancakeOuterGeo = new THREE.CylinderGeometry(1.14, 1.14, pancakeHeight, 36);
  const pancakeCopperGeo = new THREE.CylinderGeometry(1.185, 1.185, pancakeHeight * 0.56, 36);
  const flangeGeo = new THREE.CylinderGeometry(1.23, 1.23, 0.042, 36);

  for (let i = 0; i < pancakeCount; i++) {
    const y = startY + i * (pancakeHeight + pancakeGap);
    const casing = new THREE.Mesh(pancakeOuterGeo, titaniumMat);
    casing.position.y = y;
    solenoidGroup.add(casing);

    const winding = new THREE.Mesh(pancakeCopperGeo, copperCoilMat);
    winding.position.y = y;
    solenoidGroup.add(winding);

    if (i < pancakeCount - 1) {
      const flange = new THREE.Mesh(flangeGeo, steelTrimMat);
      flange.position.y = y + pancakeHeight * 0.5 + pancakeGap * 0.5;
      solenoidGroup.add(flange);
    }
  }

  const busbarGeo = new THREE.CylinderGeometry(0.04, 0.04, 5.5, 10);
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const bar = new THREE.Mesh(busbarGeo, steelTrimMat);
    bar.position.set(Math.cos(a) * 1.26, 0, Math.sin(a) * 1.26);
    solenoidGroup.add(bar);
  }

  for (const sign of [-1, 1]) {
    const crownY = sign * 2.92;
    const crownHub = new THREE.Mesh(new THREE.CylinderGeometry(1.38, 1.22, 0.42, 32), titaniumMat);
    crownHub.position.y = crownY;
    solenoidGroup.add(crownHub);

    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.95, 1.18, 0.34, 24), darkAlloyMat);
    cap.position.y = crownY + sign * 0.34;
    solenoidGroup.add(cap);

    const glowRing = new THREE.Mesh(new THREE.TorusGeometry(1.34, 0.028, 12, 48), statusStripMat);
    glowRing.rotation.x = Math.PI * 0.5;
    glowRing.position.y = crownY + sign * 0.14;
    solenoidGroup.add(glowRing);

    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * Math.PI * 2;
      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.72, 10), steelTrimMat);
      pipe.rotation.z = Math.PI * 0.5;
      pipe.rotation.y = -a;
      pipe.position.set(Math.cos(a) * 1.55, crownY + sign * 0.04, Math.sin(a) * 1.55);
      solenoidGroup.add(pipe);
    }
  }

  const solenoidFluxRings = [];
  for (const yOff of [-1.25, 0, 1.25]) {
    const ringGroup = new THREE.Group();
    ringGroup.position.y = yOff;
    const arcGeo = new THREE.TorusGeometry(1.31, 0.018, 8, 28, Math.PI * 0.42);
    for (let s = 0; s < 3; s++) {
      const seg = new THREE.Mesh(arcGeo, statusStripMat);
      seg.rotation.x = Math.PI * 0.5;
      seg.rotation.z = (s * Math.PI * 2) / 3;
      ringGroup.add(seg);
    }
    solenoidGroup.add(ringGroup);
    solenoidFluxRings.push(ringGroup);
  }

  // ---------------------------------------------------------------------------
  // 3. 16 SCULPTED D-SHAPED TOROIDAL FIELD (TF) COILS & VACUUM VESSEL SECTORS
  // ---------------------------------------------------------------------------
  const tfCoils = [];
  const vesselSectors = [];

  const tfCasingGeo = createDCoilGeometry(1.35, 1.62, 0.22, 0.026, 60);
  // Prominently visible recessed braided copper superconducting track
  const tfCopperGeo = createDCoilGeometry(1.28, 1.46, 0.28, 0.014, 60);
  const tfTrimRingGeo = createDCoilGeometry(1.59, 1.64, 0.11, 0.008, 60);

  const dPhi = (Math.PI * 2) / TOKAMAK.coilCount;

  for (let i = 0; i < TOKAMAK.coilCount; i++) {
    const phi = i * dPhi;
    const mask = sectorCutawayMask(phi);
    const dir = new THREE.Vector3(Math.cos(phi), 0, Math.sin(phi));

    const coilPivot = new THREE.Group();
    coilPivot.rotation.y = -phi;

    const casingMesh = new THREE.Mesh(tfCasingGeo, titaniumMat);
    const copperMesh = new THREE.Mesh(tfCopperGeo, copperCoilMat);
    const trimMesh = new THREE.Mesh(tfTrimRingGeo, steelTrimMat);
    coilPivot.add(casingMesh, copperMesh, trimMesh);

    const lug = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.62, 0.34), darkAlloyMat);
    lug.position.set(TOKAMAK.R0 + 1.66 * TOKAMAK.a, 0, 0);
    coilPivot.add(lug);

    const statusLed = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.40, 0.36), statusStripMat);
    statusLed.position.set(TOKAMAK.R0 + 1.88 * TOKAMAK.a, 0, 0);
    coilPivot.add(statusLed);

    for (const sign of [-1, 1]) {
      const tie = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.14, 0.18), darkAlloyMat);
      tie.position.set(1.82, sign * 2.18, 0);
      tie.rotation.z = sign * -0.18;
      coilPivot.add(tie);
    }

    root.add(coilPivot);
    tfCoils.push({ group: coilPivot, phi, dir, mask });

    const sectorMidPhi = phi + dPhi * 0.5;
    const sectorMask = sectorCutawayMask(sectorMidPhi);
    const sectorDir = new THREE.Vector3(Math.cos(sectorMidPhi), 0, Math.sin(sectorMidPhi));

    const sectorGeo = createVesselSectorGeometry(phi + 0.014, phi + dPhi - 0.014, 1.12, 1.24, 4, 38);
    const sectorMesh = new THREE.Mesh(sectorGeo, armorMat);
    root.add(sectorMesh);
    vesselSectors.push({ mesh: sectorMesh, phi: sectorMidPhi, dir: sectorDir, mask: sectorMask });
  }

  // ---------------------------------------------------------------------------
  // 4. 32 TUNGSTEN DIVERTOR CASSETTE PLATES
  // ---------------------------------------------------------------------------
  const divertorGroup = new THREE.Group();
  divertorGroup.name = 'TungstenDivertor';
  root.add(divertorGroup);

  const divWingGeo = new THREE.BoxGeometry(0.46, 0.09, 0.28);
  const divDomeGeo = new THREE.CylinderGeometry(0.10, 0.15, 0.25, 12);
  const divBaseGeo = new THREE.BoxGeometry(0.80, 0.12, 0.30);

  for (let i = 0; i < TOKAMAK.divertorCount; i++) {
    const phi = (i / TOKAMAK.divertorCount) * Math.PI * 2;
    const cassette = new THREE.Group();
    cassette.position.set(Math.cos(phi) * 2.50, -1.48, Math.sin(phi) * 2.50);
    cassette.rotation.y = -phi;

    const innerWing = new THREE.Mesh(divWingGeo, divertorMat);
    innerWing.position.set(-0.27, 0.14, 0);
    innerWing.rotation.z = -0.58;
    cassette.add(innerWing);

    const outerWing = new THREE.Mesh(divWingGeo, divertorMat);
    outerWing.position.set(0.27, 0.14, 0);
    outerWing.rotation.z = 0.58;
    cassette.add(outerWing);

    const dome = new THREE.Mesh(divDomeGeo, divertorMat);
    dome.position.set(0, 0.04, 0);
    dome.rotation.x = Math.PI * 0.5;
    cassette.add(dome);

    const base = new THREE.Mesh(divBaseGeo, darkAlloyMat);
    base.position.set(0, -0.10, 0);
    cassette.add(base);

    divertorGroup.add(cassette);
  }

  // ---------------------------------------------------------------------------
  // 5. 6 SPLIT-CUTAWAY POLOIDAL FIELD (PF) OUTER SUPERCONDUCTING RING COILS
  // ---------------------------------------------------------------------------
  // Cutaway arc centered on phi = 0.24 * PI (width = 0.72 * PI -> 130 deg clear view!)
  const cutCenter = 0.24 * Math.PI;
  const cutHalf = 0.34 * Math.PI;
  const frontStart = cutCenter - cutHalf;
  const frontArc = cutHalf * 2;
  const backStart = cutCenter + cutHalf;
  const backArc = Math.PI * 2 - frontArc;
  const frontDir = new THREE.Vector3(Math.cos(cutCenter), 0, Math.sin(cutCenter));

  const pfConfigs = [
    { radius: 2.15, y: 2.78, tube: 0.15, dir: 1, splitCutaway: false },
    { radius: 3.92, y: 2.22, tube: 0.17, dir: 1, splitCutaway: true },
    { radius: 4.58, y: 1.08, tube: 0.18, dir: 1, splitCutaway: true },
    { radius: 4.58, y: -1.08, tube: 0.18, dir: -1, splitCutaway: true },
    { radius: 3.92, y: -2.22, tube: 0.17, dir: -1, splitCutaway: false },
    { radius: 2.15, y: -2.78, tube: 0.15, dir: -1, splitCutaway: false },
  ];

  const pfRings = [];
  for (const cfg of pfConfigs) {
    const ringGroup = new THREE.Group();
    ringGroup.position.y = cfg.y;

    const buildArcSection = (startA, lenA, isFront) => {
      const sub = new THREE.Group();
      const segs = isFront ? 28 : 72;

      const housingGeo = createCappedRingArc(cfg.radius, cfg.tube, startA, lenA, 18, segs);
      sub.add(new THREE.Mesh(housingGeo, titaniumMat));

      const copperGeo = createCappedRingArc(cfg.radius + cfg.tube * 0.56, cfg.tube * 0.44, startA, lenA, 12, segs);
      sub.add(new THREE.Mesh(copperGeo, copperCoilMat));

      const innerCopperGeo = createCappedRingArc(cfg.radius - cfg.tube * 0.56, cfg.tube * 0.36, startA, lenA, 10, segs);
      sub.add(new THREE.Mesh(innerCopperGeo, copperCoilMat));

      // Add solid machined steel end-caps at cutaway seams
      if (cfg.splitCutaway) {
        const capGeo = new THREE.CylinderGeometry(cfg.tube * 1.18, cfg.tube * 1.18, 0.08, 16);
        for (const edgeAngle of [startA, startA + lenA]) {
          const cap = new THREE.Mesh(capGeo, steelTrimMat);
          cap.position.set(Math.cos(edgeAngle) * cfg.radius, 0, Math.sin(edgeAngle) * cfg.radius);
          cap.rotation.z = Math.PI * 0.5;
          cap.rotation.y = -edgeAngle + Math.PI * 0.5;
          sub.add(cap);
        }
      }

      // Clamp brackets along the arc
      const clampGeo = new THREE.BoxGeometry(cfg.tube * 2.55, cfg.tube * 2.3, 0.15);
      for (let k = 0; k < 16; k++) {
        const a = (k / 16) * Math.PI * 2;
        // Check if angle lies inside [startA, startA + lenA]
        let rel = (a - startA) % (Math.PI * 2);
        if (rel < 0) rel += Math.PI * 2;
        if (rel > 0.06 && rel < lenA - 0.06) {
          const clampMesh = new THREE.Mesh(clampGeo, darkAlloyMat);
          clampMesh.position.set(Math.cos(a) * cfg.radius, 0, Math.sin(a) * cfg.radius);
          clampMesh.rotation.y = -a;
          sub.add(clampMesh);
        }
      }
      return sub;
    };

    let frontSub = null;
    if (cfg.splitCutaway) {
      ringGroup.add(buildArcSection(backStart, backArc, false));
      frontSub = buildArcSection(frontStart, frontArc, true);
      ringGroup.add(frontSub);
    } else {
      ringGroup.add(buildArcSection(0, Math.PI * 2, false));
    }

    root.add(ringGroup);
    pfRings.push({ group: ringGroup, frontSub, baseY: cfg.y, dir: cfg.dir });
  }

  // ---------------------------------------------------------------------------
  // 6. 2 NEUTRAL BEAM INJECTORS (NBI), SPINNING TURBO PUMPS & DIAGNOSTIC LASERS
  // ---------------------------------------------------------------------------
  const nbiAssemblies = [];
  const turboRotors = [];
  const cryoPistons = [];

  // Positioned so one NBI dramatically frames the left side of the Hero shot!
  const nbiAngles = [0.84 * Math.PI, 1.48 * Math.PI];
  for (let idx = 0; idx < nbiAngles.length; idx++) {
    const phi = nbiAngles[idx];
    const nbiGroup = new THREE.Group();
    nbiGroup.rotation.y = -phi;

    const unit = new THREE.Group();
    unit.position.set(5.25, 0.0, 0.55);
    unit.rotation.y = -0.30;
    nbiGroup.add(unit);

    // Main vacuum acceleration duct
    const duct = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.50, 2.35, 24), titaniumMat);
    duct.rotation.z = Math.PI * 0.5;
    unit.add(duct);

    // Copper electromagnetic accelerator rings & steel radiator fins
    for (let f = -3; f <= 3; f++) {
      const fin = new THREE.Mesh(
        new THREE.TorusGeometry(0.55, 0.055, 10, 28),
        f % 2 === 0 ? copperCoilMat : steelTrimMat
      );
      fin.rotation.y = Math.PI * 0.5;
      fin.position.x = f * 0.28;
      unit.add(fin);
    }

    // Rear ion source housing box with glowing diagnostic strips
    const sourceBox = new THREE.Mesh(new THREE.BoxGeometry(1.08, 1.38, 1.24), darkAlloyMat);
    sourceBox.position.set(1.48, 0, 0);
    unit.add(sourceBox);

    const sourceStrip = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.12, 1.28), statusStripMat);
    sourceStrip.position.set(1.12, 0, 0);
    unit.add(sourceStrip);

    // Sapphire inspection window on top of duct
    const viewportFrame = new THREE.Mesh(new THREE.CylinderGeometry(0.30, 0.34, 0.16, 20), steelTrimMat);
    viewportFrame.position.set(-0.25, 0.52, 0);
    unit.add(viewportFrame);
    const viewportGlass = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.06, 20), sapphireGlassMat);
    viewportGlass.position.set(-0.25, 0.58, 0);
    unit.add(viewportGlass);

    // Spinning Turbomolecular Vacuum Pump
    const pumpHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.44, 20), darkAlloyMat);
    pumpHousing.position.set(0.35, -0.76, 0);
    unit.add(pumpHousing);

    const rotor = new THREE.Group();
    rotor.position.set(0.35, -0.54, 0);
    rotor.add(new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.12, 14), steelTrimMat));
    const bladeGeo = new THREE.BoxGeometry(0.28, 0.03, 0.06);
    for (let b = 0; b < 10; b++) {
      const blade = new THREE.Mesh(bladeGeo, steelTrimMat);
      const ba = (b / 10) * Math.PI * 2;
      blade.position.set(Math.cos(ba) * 0.19, 0, Math.sin(ba) * 0.19);
      blade.rotation.y = -ba;
      blade.rotation.x = 0.45;
      rotor.add(blade);
    }
    unit.add(rotor);
    turboRotors.push(rotor);

    // Reciprocating Cryogenic Valve Piston
    const pistonHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.52, 14), titaniumMat);
    pistonHousing.position.set(0.55, 0.45, 0.58);
    unit.add(pistonHousing);
    const pistonRod = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.62, 12), steelTrimMat);
    pistonRod.position.set(0.55, 0.72, 0.58);
    unit.add(pistonRod);
    cryoPistons.push({ mesh: pistonRod, baseY: 0.72, phase: idx * Math.PI });

    // Glowing Neutral Beam Particle Stream shooting into the torus plasma
    const beamCone = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.12, 3.0, 20, 1, true), nbiBeamMat);
    beamCone.rotation.z = Math.PI * 0.5;
    beamCone.position.set(-2.30, 0, 0);
    unit.add(beamCone);

    // Diagnostic Laser Interferometry Beam
    const laserRay = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 3.5, 8), laserMat);
    laserRay.rotation.z = Math.PI * 0.5;
    laserRay.position.set(-2.45, 0.36, 0);
    unit.add(laserRay);

    // Heavy floor stanchion
    const stanchion = new THREE.Mesh(new THREE.BoxGeometry(0.52, 2.85, 0.62), darkAlloyMat);
    stanchion.position.set(1.45, -2.12, 0);
    unit.add(stanchion);

    root.add(nbiGroup);
    nbiAssemblies.push({ group: unit, baseX: 5.25, idx });
  }

  // ---------------------------------------------------------------------------
  // 7. GRAVITY SUPPORTS, CRYOSTAT PEDESTAL & REACTOR HALL FLOOR
  // ---------------------------------------------------------------------------
  const supportGroup = new THREE.Group();
  root.add(supportGroup);

  for (let i = 0; i < 8; i++) {
    const phi = (i / 8) * Math.PI * 2 + Math.PI / 16;
    const legGroup = new THREE.Group();
    legGroup.rotation.y = -phi;

    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.38, 1.55, 0.42), darkAlloyMat);
    leg.position.set(3.25, -2.78, 0);
    leg.rotation.z = -0.22;
    legGroup.add(leg);

    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.16, 0.68), titaniumMat);
    foot.position.set(3.46, TOKAMAK.floorY + 0.08, 0);
    legGroup.add(foot);

    supportGroup.add(legGroup);
  }

  const pedestalRing = new THREE.Mesh(new THREE.CylinderGeometry(4.15, 4.45, 0.34, 64), darkAlloyMat);
  pedestalRing.position.y = TOKAMAK.floorY + 0.17;
  supportGroup.add(pedestalRing);

  const innerPitCap = new THREE.Mesh(new THREE.CylinderGeometry(1.58, 1.72, 0.46, 36), titaniumMat);
  innerPitCap.position.y = TOKAMAK.floorY + 0.23;
  supportGroup.add(innerPitCap);

  const floorMesh = new THREE.Mesh(new THREE.CircleGeometry(26, 96), floorMat);
  floorMesh.rotation.x = -Math.PI * 0.5;
  floorMesh.position.y = TOKAMAK.floorY;
  supportGroup.add(floorMesh);

  // ---------------------------------------------------------------------------
  // 8. ANIMATION & CUTAWAY / EXPLODE UPDATE CONTROLLER
  // ---------------------------------------------------------------------------
  function update(dt, elapsed, state) {
    uniforms.uCoilEnergy.value = state.coilEnergy;
    uniforms.uDivertorHeat.value = state.divertorHeat;
    uniforms.uPlasmaGlow.value = state.plasmaIntensity;
    uniforms.uNbiPower.value = state.nbiPower;
    uniforms.uCryoTone.value = state.cryoTone;
    uniforms.uElmFlash.value = state.elmFlash;

    const cut = state.cutaway;
    // 0.00 -> 0.45: Clean museum cutaway opens front window cleanly
    const frontOpen = Math.min(1, cut / 0.42);
    const frontEased = frontOpen * frontOpen * (3 - 2 * frontOpen);
    // 0.50 -> 1.00: Dramatic full CAD explosion of entire reactor
    const fullExplode = Math.max(0, (cut - 0.50) / 0.50);
    const explodeEased = fullExplode * fullExplode * (3 - 2 * fullExplode);

    // Update 16 TF D-Coils
    for (const c of tfCoils) {
      if (c.mask > 0.5) {
        // Front cutaway coils cleanly retract and hide so the cross-section window is crystal clear
        const push = frontEased * 2.4;
        c.group.position.set(c.dir.x * push, frontEased * 0.25, c.dir.z * push);
        const fadeScale = Math.max(0.001, 1.0 - Math.max(0, (frontEased - 0.25) / 0.55));
        c.group.scale.setScalar(fadeScale);
        c.group.visible = fadeScale > 0.05;
      } else {
        // Remaining 12 TF coils stay locked in cutaway, then explode radially at 50%..100%
        const push = explodeEased * 1.95;
        c.group.position.set(c.dir.x * push, 0, c.dir.z * push);
        c.group.scale.setScalar(1.0);
        c.group.visible = true;
      }
    }

    // Update 16 Double-Walled Vacuum Vessel Sectors
    for (const s of vesselSectors) {
      if (s.mask > 0.45) {
        const push = frontEased * 1.8;
        s.mesh.position.set(s.dir.x * push, 0, s.dir.z * push);
        s.mesh.visible = frontEased < 0.38;
      } else {
        const push = explodeEased * 1.15;
        s.mesh.position.set(s.dir.x * push, 0, s.dir.z * push);
        s.mesh.visible = true;
      }
    }

    // Update 6 PF Ring Coils (front cutaway arc opens cleanly; full rings lift in explode mode)
    for (const pf of pfRings) {
      pf.group.position.y = pf.baseY + pf.dir * explodeEased * 1.55;
      if (pf.frontSub) {
        const arcPush = frontEased * 2.2;
        pf.frontSub.position.set(frontDir.x * arcPush, 0, frontDir.z * arcPush);
        pf.frontSub.visible = frontEased < 0.35;
      }
    }

    // Pull back NBI injectors in explode mode
    for (const nbi of nbiAssemblies) {
      nbi.group.position.x = nbi.baseX + explodeEased * 1.45;
    }

    // Spin turbomolecular vacuum pump rotors
    const pumpSpeed = 4.0 + state.coilEnergy * 12.0;
    for (let i = 0; i < turboRotors.length; i++) {
      turboRotors[i].rotation.y += dt * pumpSpeed * (i % 2 === 0 ? 1 : -1);
    }

    // Reciprocate cryogenic valve pistons
    for (const p of cryoPistons) {
      p.mesh.position.y = p.baseY + Math.sin(elapsed * (1.8 + state.coilEnergy * 2.4) + p.phase) * 0.09;
    }

    // Rotate central solenoid electromagnetic flux rings
    for (let i = 0; i < solenoidFluxRings.length; i++) {
      const dir = i % 2 === 0 ? 1 : -1;
      solenoidFluxRings[i].rotation.y += dt * (0.5 + state.coilEnergy * 2.2) * dir;
    }
  }

  return {
    root,
    uniforms,
    update,
  };
}
