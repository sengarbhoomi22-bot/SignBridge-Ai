import React, { useState } from 'react';
import {
  Compass,
  Camera,
  Hand,
  Cpu,
  Edit3,
  Languages,
  Volume2,
} from 'lucide-react';
import KineticHeading from './common/KineticHeading';
import { WORKFLOW_DATA } from '../utils/constants';

const ICON_MAP = {
  Compass,
  Camera,
  Hand,
  Cpu,
  Edit3,
  Languages,
  Volume2,
};

export default function Workflow({ onSetStage }) {
  const [activeWfIndex, setActiveWfIndex] = useState(0);

  const handleSelectStep = (idx) => {
    setActiveWfIndex(idx);
    if (onSetStage && WORKFLOW_DATA[idx]) {
      onSetStage(WORKFLOW_DATA[idx].stageMap);
    }
  };

  const handlePrev = () => {
    const nextIdx = activeWfIndex === 0 ? WORKFLOW_DATA.length - 1 : activeWfIndex - 1;
    handleSelectStep(nextIdx);
  };

  const handleNext = () => {
    const nextIdx = (activeWfIndex + 1) % WORKFLOW_DATA.length;
    handleSelectStep(nextIdx);
  };

  const currentStep = WORKFLOW_DATA[activeWfIndex];
  const StepIcon = ICON_MAP[currentStep.icon] || Compass;

  return (
    <section id="how-it-works" className="relative py-24 lg:py-32 bg-coral text-ivory border-y border-amber/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-8">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
          <div className="max-w-2xl">
            <span className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-amber">
              Seven-Step Translation Workflow
            </span>
            <KineticHeading
              as="h2"
              lines={['From natural signing', 'to confirmed speech.']}
              italics={['signing', 'confirmed']}
              italicClass="italic font-normal text-amber"
              className="mt-3 font-serif font-semibold text-4xl sm:text-6xl leading-[1.02] text-ivory"
            />
          </div>
          <p className="max-w-sm text-xs sm:text-sm leading-relaxed text-ivory/90 bg-midnight/35 border border-ivory/15 rounded-2xl p-4">
            Select any stage below to inspect the user journey and synchronize the 3D silk sculpture behind the canvas.
          </p>
        </div>

        <div className="mt-14 grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
          {/* Left 7-Step Interactive Rail */}
          <div role="tablist" aria-label="Seven Workflow Steps" className="lg:col-span-6 flex flex-col gap-2.5">
            {WORKFLOW_DATA.map((step, i) => {
              const selected = i === activeWfIndex;
              return (
                <button
                  key={step.num}
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  onClick={() => handleSelectStep(i)}
                  className={`text-left rounded-2xl px-5 py-3.5 transition-all duration-300 border flex items-center justify-between gap-4 ${
                    selected
                      ? 'bg-white text-midnight border-subtleborder shadow-editorial translate-x-1.5'
                      : 'bg-midnight/30 text-ivory border-ivory/15 hover:bg-midnight/45'
                  }`}
                >
                  <div className="flex items-center gap-3.5">
                    <span
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full font-mono text-xs font-semibold ${
                        selected ? 'bg-coral text-ivory' : 'bg-ivory/15 text-ivory'
                      }`}
                    >
                      {step.num}
                    </span>
                    <span className="font-semibold text-sm sm:text-base">{step.title}</span>
                  </div>
                  <span
                    className={`font-mono text-xs ${
                      selected ? 'text-coral font-bold' : 'text-ivory/60'
                    }`}
                  >
                    {selected ? 'ACTIVE' : 'Inspect →'}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Right Pure White Stage Canvas */}
          <div className="lg:col-span-6 flex">
            <div className="w-full rounded-[2.5rem] bg-white text-midnight p-8 sm:p-11 shadow-editorialLg border border-subtleborder flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between gap-4">
                  <span className="inline-flex items-center gap-2 rounded-full bg-amber/30 border border-subtleborder px-4 py-1.5 font-mono text-xs font-semibold uppercase tracking-wider text-midnight">
                    {currentStep.badge}
                  </span>
                  <span className="font-serif italic text-5xl sm:text-6xl text-midnight/25">
                    {currentStep.num}
                  </span>
                </div>

                <h3 className="mt-7 font-serif font-semibold text-3xl sm:text-4xl leading-tight text-midnight">
                  {currentStep.title}
                </h3>
                <p className="mt-4 text-base sm:text-lg leading-relaxed text-midnight/80">
                  {currentStep.desc}
                </p>

                {/* Animated Stage Diagram Box */}
                <div className="mt-7 rounded-2xl bg-ivory border border-subtleborder p-4 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-coral text-ivory">
                      <StepIcon className="w-5 h-5" />
                    </span>
                    <div>
                      <span className="font-mono text-[10px] uppercase tracking-wider text-midnight/65 block">
                        Active Safeguard
                      </span>
                      <span className="text-xs sm:text-sm font-semibold text-midnight">
                        {currentStep.safeguard}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-9 pt-6 border-t border-subtleborder">
                <div className="flex items-center justify-between font-mono text-xs text-midnight/75 mb-3">
                  <span>Stage {activeWfIndex + 1} of 7</span>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={handlePrev}
                      className="rounded-full border border-subtleborder px-4 py-1.5 font-semibold text-midnight hover:bg-midnight hover:text-ivory transition-colors"
                    >
                      Prev
                    </button>
                    <button
                      type="button"
                      onClick={handleNext}
                      className="rounded-full bg-coral text-ivory px-4 py-1.5 font-semibold hover:bg-midnight transition-colors"
                    >
                      Next
                    </button>
                  </div>
                </div>
                <div className="h-2 w-full rounded-full bg-ivory overflow-hidden">
                  <div
                    className="h-full rounded-full bg-coral transition-all duration-500"
                    style={{ width: `${((activeWfIndex + 1) / WORKFLOW_DATA.length) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

