import React from 'react';
import { ArrowRight } from 'lucide-react';
import KineticHeading from './common/KineticHeading';
import { STAGE_CAPTIONS } from '../utils/constants';

export default function Hero({
  stageIndex,
  onSetStage,
  onPulse,
  onOpenWalkthrough,
}) {
  const STAGES = [
    { label: 'Gesture', index: 0 },
    { label: 'Intent', index: 1 },
    { label: 'Text', index: 2 },
    { label: 'Speech', index: 3 },
  ];

  return (
    <section id="hero" className="relative min-h-screen pt-28 pb-16 lg:pt-36 lg:pb-24 flex items-center">
      <div className="max-w-7xl mx-auto px-4 sm:px-8 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-8 items-center">
          {/* Left Asymmetrical Editorial Column */}
          <div className="lg:col-span-7">
            <div className="inline-flex items-center gap-2.5 rounded-full bg-white/90 backdrop-blur-md border border-subtleborder px-4 py-1.5 shadow-sm">
              <span className="h-2 w-2 rounded-full bg-coral" />
              <span className="font-mono text-[11px] font-medium uppercase tracking-[0.16em] text-midnight">
                Indian Sign Language → Text &amp; Speech
              </span>
            </div>

            <KineticHeading
              as="h1"
              lines={['Beyond words.', 'Beyond barriers.']}
              italics={['words.', 'barriers.']}
              italicClass="italic font-normal text-coral"
              className="mt-6 font-serif font-semibold text-5xl sm:text-7xl xl:text-[5.5rem] leading-[0.96] tracking-[-0.02em] text-midnight"
            />

            <p className="mt-6 max-w-xl text-base sm:text-lg leading-relaxed text-midnight/85 bg-white/85 backdrop-blur-sm rounded-2xl p-4 border border-subtleborder">
              SignBridge AI bridges Indian Sign Language and spoken communication—translating supported signing sequences into editable English and Hindi text, and speaking aloud only after your explicit confirmation.
            </p>

            {/* Hero CTAs */}
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <button
                type="button"
                onClick={() => onOpenWalkthrough('Everyday Conversations')}
                className="group inline-flex items-center gap-3 rounded-full bg-coral px-7 py-4 text-xs sm:text-sm font-semibold uppercase tracking-[0.12em] text-ivory shadow-editorial transition-all duration-300 hover:bg-midnight hover:-translate-y-0.5"
              >
                <span>Explore Prototype Flow</span>
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-amber text-midnight transition-transform duration-300 group-hover:translate-x-1">
                  <ArrowRight className="w-4 h-4" />
                </span>
              </button>
              <a
                href="#visual-journey"
                className="inline-flex items-center gap-2 rounded-full bg-white/90 backdrop-blur-md border border-subtleborder px-6 py-4 text-xs sm:text-sm font-semibold uppercase tracking-[0.12em] text-midnight transition-all hover:bg-white hover:border-coral/50"
              >
                <span>Witness the 4-Act Bridge</span>
              </a>
            </div>

            {/* Layered Architectural Metadata Badges */}
            <div className="mt-10 grid grid-cols-3 gap-3 max-w-lg">
              <div className="rounded-2xl bg-white/90 backdrop-blur-md border border-subtleborder p-3.5">
                <span className="font-mono text-[10px] uppercase tracking-wider text-coral font-semibold block">Temporal AI</span>
                <span className="font-serif text-lg sm:text-xl font-semibold text-midnight">Sequence-Level</span>
              </div>
              <div className="rounded-2xl bg-white/90 backdrop-blur-md border border-subtleborder p-3.5">
                <span className="font-mono text-[10px] uppercase tracking-wider text-slatecobalt font-semibold block">Bilingual</span>
                <span className="font-serif text-lg sm:text-xl font-semibold text-midnight">English &amp; हिन्दी</span>
              </div>
              <div className="rounded-2xl bg-white/90 backdrop-blur-md border border-subtleborder p-3.5">
                <span className="font-mono text-[10px] uppercase tracking-wider text-coral font-semibold block">Human-First</span>
                <span className="font-serif text-lg sm:text-xl font-semibold text-midnight">Edit &amp; Confirm</span>
              </div>
            </div>
          </div>

          {/* Right Interactive 3D Sculpture Stage Controller */}
          <div className="lg:col-span-5 flex flex-col justify-end lg:min-h-[500px]">
            <div className="mt-auto rounded-[2rem] bg-white/92 backdrop-blur-md border border-subtleborder p-5 sm:p-6 shadow-editorialLg">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-coral animate-ping" />
                  <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-midnight">
                    Interactive 3D Silk Sculpture
                  </span>
                </div>
                <button
                  type="button"
                  onClick={onPulse}
                  className="rounded-full bg-amber/30 hover:bg-amber border border-subtleborder px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-midnight transition-colors"
                >
                  Emit Signal Pulse
                </button>
              </div>

              <p
                className="mt-2.5 text-xs sm:text-sm leading-relaxed text-midnight/80"
                dangerouslySetInnerHTML={{ __html: STAGE_CAPTIONS[stageIndex] || STAGE_CAPTIONS[0] }}
              />

              {/* Live 4-Stage Morph Switcher Pill */}
              <div
                role="tablist"
                aria-label="3D Sculpture Transformation Stages"
                className="mt-4 pt-4 border-t border-subtleborder grid grid-cols-4 gap-1.5"
              >
                {STAGES.map((stage) => {
                  const active = stageIndex === stage.index;
                  return (
                    <button
                      key={stage.index}
                      type="button"
                      role="tab"
                      aria-selected={active}
                      onClick={() => onSetStage(stage.index)}
                      className={`stage-pill rounded-xl py-2 px-2 font-mono text-[11px] font-semibold uppercase tracking-wider transition-all ${
                        active
                          ? 'bg-coral text-ivory'
                          : 'bg-ivory text-midnight hover:bg-amber/25'
                      }`}
                    >
                      {stage.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

