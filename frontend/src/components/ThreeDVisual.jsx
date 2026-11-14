import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export default function ThreeDVisual({ stageIndex = 0, pulseTrigger = 0 }) {
  const canvasRef = useRef(null);
  const [hasWebGL, setHasWebGL] = useState(true);
  const pulseRef = useRef(0);

  // Trigger pulse effect whenever pulseTrigger changes and is > 0
  useEffect(() => {
    if (pulseTrigger > 0) {
      pulseRef.current = 1.0;
    }
  }, [pulseTrigger]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const isMobile = window.innerWidth < 768;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
        powerPreference: 'high-performance',
      });
    } catch (e) {
      console.warn('WebGL initialization failed, falling back to CSS visual:', e);
      setHasWebGL(false);
      return;
    }

    setHasWebGL(true);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 0, 6.8);

    const silkVertexShader = `
      uniform float uTime;
      uniform float uMorph;
      uniform float uPulse;
      uniform vec2 uPointer;
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vec3 pos = position;
        float waveA = sin(pos.x * 2.2 + uTime * 1.45) * cos(pos.y * 2.0 + uTime * 1.15);
        float waveB = sin(pos.z * 2.6 - uTime * 1.7 + uMorph * 3.14159);
        float amp = 0.085 + uMorph * 0.09 + uPulse * 0.16;
        pos += normal * (waveA * 0.65 + waveB * 0.35) * amp;
        pos.x += uPointer.x * 0.12 * sin(uv.x * 6.2831);
        pos.y += uPointer.y * 0.12 * cos(uv.x * 6.2831);
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        vViewPosition = -mvPosition.xyz;
        vNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * mvPosition;
      }
    `;

    const silkFragmentShader = `
      uniform vec3 uColorA;
      uniform vec3 uColorB;
      uniform vec3 uRimColor;
      uniform float uTime;
      uniform float uPulse;
      varying vec3 vNormal;
      varying vec3 vViewPosition;
      varying vec2 vUv;
      void main() {
        vec3 normal = normalize(vNormal);
        vec3 viewDir = normalize(vViewPosition);
        float fresnel = pow(1.0 - abs(dot(normal, viewDir)), 2.2);
        float flow = 0.5 + 0.5 * sin(vUv.x * 12.566 - uTime * 1.6);
        vec3 baseColor = mix(uColorA, uColorB, flow);
        vec3 finalColor = mix(baseColor, uRimColor, fresnel * 0.78 + uPulse * 0.22);
        gl_FragColor = vec4(finalColor, 0.9);
      }
    `;

    const sculptureGroup = new THREE.Group();
    sculptureGroup.position.set(isMobile ? 0 : 1.65, 0.08, 0);
    scene.add(sculptureGroup);

    const tubularSegs = isMobile ? 96 : 160;
    const radialSegs = isMobile ? 16 : 24;

    const materials = [];
    const geometries = [];

    function createShaderRibbon(rx, ry, rz, phase, radius, colA, colB, rimCol) {
      const pts = [];
      for (let i = 0; i <= 96; i++) {
        const t = (i / 96) * Math.PI * 2;
        pts.push(
          new THREE.Vector3(
            Math.sin(t + phase) * rx + Math.sin(t * 2) * 0.38,
            Math.cos(t * 1.5 + phase) * ry + Math.sin(t * 3) * 0.22,
            Math.cos(t - phase) * rz
          )
        );
      }
      const curve = new THREE.CatmullRomCurve3(pts, true);
      const geo = new THREE.TubeGeometry(curve, tubularSegs, radius, radialSegs, true);
      geometries.push(geo);

      const uniforms = {
        uTime: { value: 0 },
        uMorph: { value: 0 },
        uPulse: { value: 0 },
        uPointer: { value: new THREE.Vector2(0, 0) },
        uColorA: { value: new THREE.Color(colA) },
        uColorB: { value: new THREE.Color(colB) },
        uRimColor: { value: new THREE.Color(rimCol) },
      };

      const mat = new THREE.ShaderMaterial({
        vertexShader: silkVertexShader,
        fragmentShader: silkFragmentShader,
        uniforms,
        transparent: true,
        side: THREE.DoubleSide,
      });
      materials.push(mat);

      const mesh = new THREE.Mesh(geo, mat);
      sculptureGroup.add(mesh);
      return uniforms;
    }

    const uniformsPrimary = createShaderRibbon(1.85, 1.25, 1.15, 0.0, 0.22, '#6D263D', '#D6AD70', '#FAF5EF');
    const uniformsSecondary = createShaderRibbon(2.15, 1.05, 1.35, 2.1, 0.13, '#C97861', '#6D263D', '#D6AD70');
    const uniformsTertiary = createShaderRibbon(1.55, 1.4, 0.95, 4.2, 0.095, '#D6AD70', '#C97861', '#FAF5EF');

    // Delicate Orbital Ring
    const ringGeo = new THREE.TorusGeometry(2.45, 0.022, 16, 120);
    geometries.push(ringGeo);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xd6ad70,
      transparent: true,
      opacity: 0.65,
    });
    materials.push(ringMat);
    const orbitalRing = new THREE.Mesh(ringGeo, ringMat);
    orbitalRing.rotation.set(Math.PI / 3.1, 0.25, 0);
    sculptureGroup.add(orbitalRing);

    // 360 Spline-Following Communication Signal Particles
    const pCount = isMobile ? 160 : 360;
    const splineStreams = [0, 1, 2].map((idx) => {
      const pts = [];
      const ph = (idx * Math.PI * 2) / 3;
      for (let i = 0; i <= 48; i++) {
        const u = (i / 48) * Math.PI * 2;
        pts.push(
          new THREE.Vector3(
            Math.sin(u + ph) * (2.3 + idx * 0.24),
            Math.cos(u * 1.5 + ph) * (1.4 + idx * 0.14),
            Math.cos(u - ph) * 1.6
          )
        );
      }
      return new THREE.CatmullRomCurve3(pts, true);
    });

    const paletteColors = [
      new THREE.Color('#6D263D'), // Rich Burgundy
      new THREE.Color('#D6AD70'), // Champagne Gold
      new THREE.Color('#C97861'), // Soft Terracotta
      new THREE.Color('#3B292B'), // Warm Espresso
      new THREE.Color('#6D263D'), // Rich Burgundy
    ];

    const pMeta = [];
    const pPos = new Float32Array(pCount * 3);
    const pCol = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i++) {
      const streamIdx = i % splineStreams.length;
      const progress = Math.random();
      const speed = 0.045 + Math.random() * 0.075;
      const radius = 0.07 + Math.random() * 0.18;
      const angle = Math.random() * Math.PI * 2;
      const col = paletteColors[i % paletteColors.length];
      pMeta.push({ streamIdx, progress, speed, radius, angle });
      const pt = splineStreams[streamIdx].getPointAt(progress);
      pPos[i * 3] = pt.x + Math.cos(angle) * radius;
      pPos[i * 3 + 1] = pt.y + Math.sin(angle) * radius;
      pPos[i * 3 + 2] = pt.z;
      pCol[i * 3] = col.r;
      pCol[i * 3 + 1] = col.g;
      pCol[i * 3 + 2] = col.b;
    }

    const pGeo = new THREE.BufferGeometry();
    geometries.push(pGeo);
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
    const pMat = new THREE.PointsMaterial({
      size: isMobile ? 0.072 : 0.064,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
      depthWrite: false,
    });
    materials.push(pMat);

    const particleSystem = new THREE.Points(pGeo, pMat);
    particleSystem.position.copy(sculptureGroup.position);
    scene.add(particleSystem);

    const tempVec = new THREE.Vector3();

    // Interaction tracking
    let pointerX = 0;
    let pointerY = 0;
    let scrollProgress = 0;
    let scrollVelocity = 0;
    let lastScrollY = window.scrollY;

    const handlePointerMove = (e) => {
      pointerX = (e.clientX / window.innerWidth) * 2 - 1;
      pointerY = -(e.clientY / window.innerHeight) * 2 + 1;
    };

    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      const maxScroll = Math.max(
        document.documentElement.scrollHeight - window.innerHeight,
        1
      );
      scrollProgress = Math.min(Math.max(currentScrollY / maxScroll, 0), 1);
      scrollVelocity = (currentScrollY - lastScrollY) / 100;
      lastScrollY = currentScrollY;
    };

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('resize', handleResize);

    const clock = new THREE.Clock();
    let animId;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      const t = clock.elapsedTime;

      // Pulse decay
      pulseRef.current = Math.max(0, pulseRef.current - dt * 1.25);
      const pulse = pulseRef.current;

      const stageNorm = stageIndex / 3;

      // Update GLSL Uniforms
      [uniformsPrimary, uniformsSecondary, uniformsTertiary].forEach((u) => {
        if (!reducedMotion) u.uTime.value = t;
        u.uMorph.value += (stageNorm - u.uMorph.value) * 0.06;
        u.uPulse.value = pulse;
        u.uPointer.value.set(pointerX, pointerY);
      });

      // Scroll-choreographed sculpture trajectory
      const mobileNow = window.innerWidth < 768;
      const baseX = mobileNow ? 0 : 1.65;
      const targetX = mobileNow
        ? 0
        : scrollProgress < 0.25
        ? baseX - scrollProgress * 4.6
        : scrollProgress < 0.65
        ? -1.4 + (scrollProgress - 0.25) * 3.8
        : Math.sin(scrollProgress * Math.PI * 2) * 1.1;

      const targetY =
        Math.sin(scrollProgress * Math.PI * 3) * 0.3 +
        (reducedMotion ? 0 : Math.sin(t * 0.9) * 0.14);

      const targetScale = (mobileNow ? 0.66 : 0.98 - scrollProgress * 0.14) + pulse * 0.08;

      sculptureGroup.position.x += (targetX - sculptureGroup.position.x) * 0.055;
      sculptureGroup.position.y += (targetY - sculptureGroup.position.y) * 0.055;
      sculptureGroup.scale.setScalar(sculptureGroup.scale.x + (targetScale - sculptureGroup.scale.x) * 0.07);

      const rotY = pointerX * 0.55 + scrollProgress * Math.PI * 1.6 + stageNorm * 0.6 + (reducedMotion ? 0 : t * 0.2);
      const rotX = -pointerY * 0.35 + scrollProgress * 0.65;
      sculptureGroup.rotation.y += (rotY - sculptureGroup.rotation.y) * 0.06;
      sculptureGroup.rotation.x += (rotX - sculptureGroup.rotation.x) * 0.06;

      if (!reducedMotion) {
        orbitalRing.rotation.z -= dt * 0.22;
        orbitalRing.rotation.x += dt * 0.14;
      }

      // Particle Stream Motion
      particleSystem.position.x += (targetX + pointerX * 0.25 - particleSystem.position.x) * 0.055;
      particleSystem.position.y += (-pointerY * 0.2 - particleSystem.position.y) * 0.055;

      if (!reducedMotion) {
        const dir = scrollVelocity < -0.15 ? -1 : 1;
        const speedBoost = 1 + stageNorm * 0.65 + pulse * 1.4 + Math.min(Math.abs(scrollVelocity) * 0.35, 2.0);
        const attr = pGeo.attributes.position;
        for (let i = 0; i < pCount; i++) {
          const m = pMeta[i];
          m.progress = (m.progress + dt * m.speed * speedBoost * dir + 1) % 1;
          splineStreams[m.streamIdx].getPointAt(m.progress, tempVec);
          const swirl = t * 1.2 + m.angle;
          const spread = m.radius + pulse * 0.85;
          attr.array[i * 3] = tempVec.x + Math.cos(swirl) * spread;
          attr.array[i * 3 + 1] = tempVec.y + Math.sin(swirl) * spread;
          attr.array[i * 3 + 2] = tempVec.z + Math.sin(swirl * 0.75) * spread;
        }
        attr.needsUpdate = true;
      }

      camera.position.x += (pointerX * 0.35 - camera.position.x) * 0.04;
      camera.position.y += (pointerY * 0.25 - scrollProgress * 0.28 - camera.position.y) * 0.04;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleResize);

      geometries.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
      if (renderer) renderer.dispose();
    };
  }, [stageIndex]);

  return (
    <div id="webgl-container" aria-hidden="true" className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      <canvas ref={canvasRef} id="webgl-stage" className={!hasWebGL ? 'hidden' : ''} />
      {/* Graceful Fallback if WebGL is Disabled or Fails */}
      <div id="webgl-fallback" className={`${hasWebGL ? 'hidden' : ''} absolute inset-0 bg-ivory editorial-grain`}>
        <div className="absolute -top-24 right-0 h-[520px] w-[520px] rounded-full bg-coral/20 blur-3xl" />
        <div className="absolute top-1/3 -left-24 h-[460px] w-[460px] rounded-full bg-slatecobalt/20 blur-3xl" />
        <div className="absolute bottom-0 right-1/4 h-[440px] w-[440px] rounded-full bg-amber/25 blur-3xl" />
      </div>
    </div>
  );
}
