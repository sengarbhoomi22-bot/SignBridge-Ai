import React from 'react';
import KineticHeading from './common/KineticHeading';

export default function VisualJourney({ onSetStage }) {
  return (
    <section id="visual-journey" className="relative py-24 lg:py-32 bg-ivory/90 backdrop-blur-md border-y border-subtleborder editorial-grain">
      <div className="max-w-7xl mx-auto px-4 sm:px-8">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
          <div className="max-w-2xl">
            <span className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-coral">
              The Four-Act Metaphor
            </span>
            <KineticHeading
              as="h2"
              lines={['Gesture becomes wave.', 'Wave becomes voice.']}
              italics={['wave.', 'voice.']}
              italicClass="italic font-normal text-coral"
              className="mt-3 font-serif font-semibold text-4xl sm:text-6xl leading-[1.02] text-midnight"
            />
          </div>
          <p className="max-w-md text-xs sm:text-sm leading-relaxed text-midnight/80 bg-white border border-subtleborder rounded-2xl p-4 shadow-sm">
            <strong>Creative Brand Metaphor:</strong> This interactive sequence illustrates how SignBridge AI bridges visual signing and spoken language—without fabricating live AI predictions.
          </p>
        </div>

        {/* 4-Act Interactive Visual Bridge Grid with Custom Animated SVG Kinematics */}
        <div className="mt-14 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Act 1: GESTURE */}
          <article
            onMouseEnter={() => onSetStage && onSetStage(0)}
            className="act-card group rounded-[2rem] bg-white border border-subtleborder p-6 sm:p-7 shadow-editorial transition-all duration-300 hover:-translate-y-1.5 hover:border-coral cursor-pointer"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-semibold uppercase tracking-widest text-coral">Act 01 · Gesture</span>
              <span className="h-2.5 w-2.5 rounded-full bg-coral" />
            </div>
            {/* Looping SVG: Expressive Hand Trajectory & Joint Nodes */}
            <div className="my-6 h-36 rounded-2xl bg-ivory border border-subtleborder flex items-center justify-center p-4">
              <svg viewBox="0 0 200 110" className="w-full h-full" fill="none" aria-hidden="true">
                <path d="M22 82 C 55 24, 95 92, 130 34 C 148 12, 168 44, 180 26" stroke="#6D263D" strokeWidth="2.5" className="svg-flow-line" />
                <path d="M30 92 C 70 46, 105 98, 145 48" stroke="#D6AD70" strokeWidth="1.5" strokeOpacity="0.85" />
                <circle cx="56" cy="48" r="5" fill="#6D263D" className="svg-pulse-node" />
                <circle cx="96" cy="68" r="4.5" fill="#3B292B" className="svg-pulse-node" style={{ animationDelay: '0.4s' }} />
                <circle cx="130" cy="34" r="6" fill="#D6AD70" className="svg-pulse-node" style={{ animationDelay: '0.8s' }} />
                <circle cx="180" cy="26" r="5" fill="#C97861" className="svg-pulse-node" style={{ animationDelay: '1.2s' }} />
              </svg>
            </div>
            <h3 className="font-serif text-2xl font-semibold text-midnight">Spatial Kinematics</h3>
            <p className="mt-2 text-sm leading-relaxed text-midnight/75">
              Continuous hand motion, posture, and non-manual cues unfold across frames rather than frozen single-letter poses.
            </p>
          </article>

          {/* Act 2: UNDERSTANDING */}
          <article
            onMouseEnter={() => onSetStage && onSetStage(1)}
            className="act-card group rounded-[2rem] bg-white border border-subtleborder p-6 sm:p-7 shadow-editorial transition-all duration-300 hover:-translate-y-1.5 hover:border-slatecobalt cursor-pointer"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-semibold uppercase tracking-widest text-slatecobalt">Act 02 · Understanding</span>
              <span className="h-2.5 w-2.5 rounded-full bg-slatecobalt" />
            </div>
            {/* Looping SVG: Temporal Landmark Mesh Convergence */}
            <div className="my-6 h-36 rounded-2xl bg-ivory border border-subtleborder flex items-center justify-center p-4">
              <svg viewBox="0 0 200 110" className="w-full h-full" fill="none" aria-hidden="true">
                <polygon points="40,55 95,22 155,55 95,88" stroke="#C97861" strokeWidth="2" className="svg-flow-line" />
                <line x1="40" y1="55" x2="155" y2="55" stroke="#6D263D" strokeWidth="1.5" strokeDasharray="4 4" />
                <line x1="95" y1="22" x2="95" y2="88" stroke="#D6AD70" strokeWidth="1.5" strokeDasharray="4 4" />
                <circle cx="95" cy="55" r="9" fill="#C97861" fillOpacity="0.22" stroke="#C97861" strokeWidth="2" className="svg-pulse-node" />
                <circle cx="95" cy="55" r="3.5" fill="#6D263D" />
              </svg>
            </div>
            <h3 className="font-serif text-2xl font-semibold text-midnight">Temporal Alignment</h3>
            <p className="mt-2 text-sm leading-relaxed text-midnight/75">
              Landmark trajectories are analysed across time—abstaining honestly with a repeat prompt if confidence is uncertain.
            </p>
          </article>

          {/* Act 3: TEXT */}
          <article
            onMouseEnter={() => onSetStage && onSetStage(2)}
            className="act-card group rounded-[2rem] bg-white border border-subtleborder p-6 sm:p-7 shadow-editorial transition-all duration-300 hover:-translate-y-1.5 hover:border-amber cursor-pointer"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-semibold uppercase tracking-widest text-midnight">Act 03 · Text</span>
              <span className="h-2.5 w-2.5 rounded-full bg-amber" />
            </div>
            {/* Looping SVG: Structured Sentence & Bilingual Script */}
            <div className="my-6 h-36 rounded-2xl bg-ivory border border-subtleborder flex flex-col justify-center gap-2.5 px-5">
              <div className="flex items-center gap-2">
                <span className="h-2 w-14 rounded-full bg-midnight" />
                <span className="h-2 w-20 rounded-full bg-coral" />
                <span className="h-2 w-8 rounded-full bg-amber" />
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-24 rounded-full bg-slatecobalt" />
                <span className="h-2 w-12 rounded-full bg-midnight/40" />
              </div>
              <div className="mt-1 inline-flex items-center justify-between rounded-lg bg-white border border-subtleborder px-2.5 py-1.5">
                <span className="font-mono text-[10px] font-semibold text-midnight">EN / हिन्दी · Editable</span>
                <span className="h-2 w-2 rounded-full bg-amber animate-ping" />
              </div>
            </div>
            <h3 className="font-serif text-2xl font-semibold text-midnight">Editable Sentence</h3>
            <p className="mt-2 text-sm leading-relaxed text-midnight/75">
              The signer inspects, refines, and selects English or Hindi text before authorizing any audio output.
            </p>
          </article>

          {/* Act 4: SPEECH */}
          <article
            onMouseEnter={() => onSetStage && onSetStage(3)}
            className="act-card group rounded-[2rem] bg-white border border-subtleborder p-6 sm:p-7 shadow-editorial transition-all duration-300 hover:-translate-y-1.5 hover:border-coral cursor-pointer"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-semibold uppercase tracking-widest text-coral">Act 04 · Speech</span>
              <span className="h-2.5 w-2.5 rounded-full bg-coral" />
            </div>
            {/* Looping SVG: Confirmed Harmonic Acoustic Waveform */}
            <div className="my-6 h-36 rounded-2xl bg-ivory border border-subtleborder flex items-center justify-center p-4">
              <svg viewBox="0 0 180 90" className="w-full h-full" fill="none" aria-hidden="true">
                <rect x="24" y="25" width="6" height="40" rx="3" fill="#6D263D" className="svg-wave-bar" style={{ animationDelay: '0.0s' }} />
                <rect x="38" y="15" width="6" height="60" rx="3" fill="#C97861" className="svg-wave-bar" style={{ animationDelay: '0.15s' }} />
                <rect x="52" y="22" width="6" height="46" rx="3" fill="#D6AD70" className="svg-wave-bar" style={{ animationDelay: '0.3s' }} />
                <rect x="66" y="10" width="6" height="70" rx="3" fill="#3B292B" className="svg-wave-bar" style={{ animationDelay: '0.45s' }} />
                <rect x="80" y="18" width="6" height="54" rx="3" fill="#C97861" className="svg-wave-bar" style={{ animationDelay: '0.6s' }} />
                <rect x="94" y="12" width="6" height="66" rx="3" fill="#6D263D" className="svg-wave-bar" style={{ animationDelay: '0.75s' }} />
                <rect x="108" y="24" width="6" height="42" rx="3" fill="#D6AD70" className="svg-wave-bar" style={{ animationDelay: '0.9s' }} />
                <rect x="122" y="16" width="6" height="58" rx="3" fill="#6D263D" className="svg-wave-bar" style={{ animationDelay: '1.05s' }} />
                <rect x="136" y="28" width="6" height="34" rx="3" fill="#C97861" className="svg-wave-bar" style={{ animationDelay: '1.2s' }} />
                <rect x="150" y="20" width="6" height="50" rx="3" fill="#3B292B" className="svg-wave-bar" style={{ animationDelay: '1.35s' }} />
              </svg>
            </div>
            <h3 className="font-serif text-2xl font-semibold text-midnight">Confirmed Voice</h3>
            <p className="mt-2 text-sm leading-relaxed text-midnight/75">
              Clear spoken output plays in English or Hindi only after the user explicitly confirms the final sentence.
            </p>
          </article>
        </div>
      </div>
    </section>
  );
}

