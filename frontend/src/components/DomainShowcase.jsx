import React, { useState } from 'react';
import {
  GraduationCap,
  HeartPulse,
  Landmark,
  Briefcase,
  TrainFront,
  Building2,
  MessageCircleHeart,
  ArrowUpRight,
} from 'lucide-react';
import KineticHeading from './common/KineticHeading';
import { DOMAIN_WORLDS } from '../utils/constants';

const DOMAIN_ICONS = {
  GraduationCap,
  HeartPulse,
  Landmark,
  Briefcase,
  TrainFront,
  Building2,
  MessageCircleHeart,
};

export default function DomainShowcase({ onOpenWalkthrough, onPulse }) {
  const [activeDomainIndex, setActiveDomainIndex] = useState(0);

  const handleSelectDomain = (idx) => {
    setActiveDomainIndex(idx);
    if (onPulse) onPulse();
  };

  const currentDomain = DOMAIN_WORLDS[activeDomainIndex];
  const DomainIcon = DOMAIN_ICONS[currentDomain.icon] || GraduationCap;

  return (
    <section id="domains" className="relative py-24 lg:py-32 bg-ivory/92 backdrop-blur-md border-b border-subtleborder editorial-grain">
      <div className="max-w-7xl mx-auto px-4 sm:px-8">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
          <div className="max-w-2xl">
            <span className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-coral">
              Seven Communication Environments
            </span>
            <KineticHeading
              as="h2"
              lines={['Seven worlds.', 'One shared engine.']}
              italics={['worlds.', 'shared']}
              italicClass="italic font-normal text-coral"
              className="mt-3 font-serif font-semibold text-4xl sm:text-6xl leading-[1.02] text-midnight"
            />
          </div>
          <p className="max-w-md text-xs sm:text-sm leading-relaxed text-midnight/85 bg-white border border-subtleborder rounded-2xl p-4 shadow-sm">
            <strong>Honest Scope Distinction:</strong> These seven domains structure vocabulary labels and interface flows. Actual model recognition covers only validated phrases backed by trained datasets.
          </p>
        </div>

        {/* Asymmetric Architectural Domain Explorer */}
        <div className="mt-14 grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
          {/* Left: 7 Organic Sculptural Domain Nodes */}
          <div className="lg:col-span-5 flex flex-col gap-3">
            {DOMAIN_WORLDS.map((item, i) => {
              const active = i === activeDomainIndex;
              return (
                <button
                  key={item.name}
                  type="button"
                  onClick={() => handleSelectDomain(i)}
                  className={`group text-left rounded-[1.5rem] px-5 py-4 border transition-all duration-300 flex items-center justify-between gap-4 ${
                    active
                      ? 'bg-coral text-ivory border-coral shadow-editorial translate-x-2'
                      : 'bg-white text-midnight border-subtleborder hover:border-coral/40'
                  }`}
                >
                  <div className="flex items-center gap-3.5">
                    <span
                      className="h-3 w-3 rounded-full shrink-0"
                      style={{ backgroundColor: active ? '#D6AD70' : item.accentHex }}
                    />
                    <span className="font-serif font-semibold text-xl sm:text-2xl">{item.name}</span>
                  </div>
                  <span
                    className={`font-mono text-[11px] uppercase tracking-wider ${
                      active ? 'text-amber' : 'text-midnight/45'
                    }`}
                  >
                    0{i + 1}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Right: Dimensional Domain Showcase Stage */}
          <div className="lg:col-span-7 flex">
            <div className="w-full rounded-[2.5rem] bg-white border border-subtleborder p-8 sm:p-11 shadow-editorialLg flex flex-col justify-between transition-all duration-500">
              <div>
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <span
                    className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 font-mono text-xs font-semibold uppercase tracking-wider text-ivory ${currentDomain.badgeBg}`}
                  >
                    Domain 0{activeDomainIndex + 1} / 07
                  </span>
                  <span className="font-mono text-xs uppercase tracking-widest text-midnight/65">
                    {currentDomain.tag}
                  </span>
                </div>

                <div className="mt-8 grid grid-cols-1 sm:grid-cols-12 gap-6 items-center">
                  <div className="sm:col-span-8">
                    <h3 className="font-serif font-semibold text-3xl sm:text-5xl text-midnight">
                      {currentDomain.name}
                    </h3>
                    <p className="mt-4 text-base sm:text-lg leading-relaxed text-midnight/80">
                      {currentDomain.desc}
                    </p>
                  </div>
                  <div className="sm:col-span-4 flex justify-center">
                    <div
                      className={`h-32 w-32 rounded-[2rem] border flex items-center justify-center transition-all duration-500 ${currentDomain.emblemBg}`}
                    >
                      <DomainIcon className="w-14 h-14" />
                    </div>
                  </div>
                </div>

                {/* Sample Context Phrase Box */}
                <div className="mt-8 rounded-2xl bg-ivory border border-subtleborder p-5">
                  <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-midnight/60 block">
                    Illustrative Context Example (UI Scope)
                  </span>
                  <p className="mt-1.5 font-serif italic text-xl sm:text-2xl text-midnight">
                    {currentDomain.example}
                  </p>
                </div>
              </div>

              <div className="mt-8 pt-6 border-t border-subtleborder flex flex-wrap items-center justify-between gap-4">
                <span className="text-xs text-midnight/70">
                  Shared recognition engine · Vocabulary labels adapt to context
                </span>
                <button
                  type="button"
                  onClick={() => onOpenWalkthrough && onOpenWalkthrough(currentDomain.name)}
                  className="inline-flex items-center gap-2 rounded-full bg-coral px-5 py-2.5 text-xs font-semibold uppercase tracking-wider text-ivory hover:bg-midnight transition-colors"
                >
                  <span>Preview This Context</span>
                  <ArrowUpRight className="w-4 h-4 text-amber" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

