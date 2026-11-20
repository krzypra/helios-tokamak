import * as THREE from 'three/webgpu';

/**
 * Shared Miller Equilibrium Tokamak Geometry Constants.
 * Every mechanical and plasma subsystem derives its coordinates from here
 * so coils, vacuum vessel, divertor, flux lines, and WebGPU compute ions align cleanly.
 */
export const TOKAMAK = {
  R0: 2.68,       // Major radius (m)
  a: 0.90,        // Minor horizontal plasma radius (m)
  kappa: 1.52,    // Vertical elongation (D-section height factor)
  delta: 0.38,    // Triangularity (flattens inboard wall near central solenoid)
  coilCount: 16,  // Number of Toroidal Field (TF) D-coils
  divertorCount: 32,
  floorY: -3.55,
};

/**
 * Compute 3D point on a Miller D-shaped toroidal surface.
 * @param {number} phi Toroidal angle [0, 2*PI)
 * @param {number} theta Poloidal angle [0, 2*PI)
 * @param {number} rho Normalized minor radius (1.0 = plasma separatrix)
 * @param {THREE.Vector3} [out]
 */
export function millerPoint(phi, theta, rho = 1.0, out = new THREE.Vector3()) {
  const { R0, a, kappa, delta } = TOKAMAK;
  const r = R0 + rho * a * Math.cos(theta + delta * Math.sin(theta));
  const y = rho * kappa * a * Math.sin(theta);
  out.set(r * Math.cos(phi), y, r * Math.sin(phi));
  return out;
}

/**
 * Compute 2D (R, Y) cross-section point on the Miller D-contour.
 */
export function millerCrossSection(theta, rho = 1.0) {
  const { R0, a, kappa, delta } = TOKAMAK;
  return {
    r: R0 + rho * a * Math.cos(theta + delta * Math.sin(theta)),
    y: rho * kappa * a * Math.sin(theta),
  };
}

/**
 * Compute cutaway weight [0..1] for a sector at toroidal angle phi.
 * Front-right sector (around phi = 0.22 * PI, facing the default hero camera)
 * opens first during cutaway, while all sectors expand at high explode values.
 */
export function sectorCutawayMask(phi) {
  // Center of cutaway opening faces (+X, +Z) at ~42 degrees
  const target = 0.24 * Math.PI;
  let diff = Math.atan2(Math.sin(phi - target), Math.cos(phi - target));
  const absDiff = Math.abs(diff);
  // Smooth mask: 1.0 inside +/- 62 deg arc, 0 outside +/- 78 deg
  const inner = 0.34 * Math.PI;
  const outer = 0.44 * Math.PI;
  if (absDiff <= inner) return 1.0;
  if (absDiff >= outer) return 0.0;
  const t = (absDiff - inner) / (outer - inner);
  return 1.0 - t * t * (3 - 2 * t);
}
