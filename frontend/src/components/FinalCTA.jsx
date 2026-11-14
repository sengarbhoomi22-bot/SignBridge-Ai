import React from 'react';
import { ArrowRight } from 'lucide-react';
import KineticHeading from './common/KineticHeading';

export default function FinalCTA({ onOpenWalkthrough }) {
  return (
    <section className="py-20 lg:py-28 px-4 sm:px-8 bg-ivory/95">
      <div className="max-w-7xl mx-auto">
        <div className="rounded-[2.75rem] bg-midnight text-ivory px-7 py-16 sm:px-14 sm:py-24 text-center shadow-editorialLg relative overflow-hidden">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -top-24 -left-24 h-72 w-72 rounded-full bg-coral/35 blur-3xl"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -bottom-24 -right-24 h-72 w-72 rounded-full bg-amber/20 blur-3xl"
          />

          <div className="relative z-10 max-w-2xl mx-auto">
            <span className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-amber">
              Experience the Platform Flow
            </span>
            <KineticHeading
              as="h2"
              lines={["Let's make understanding", 'effortless.']}
              italics={['understanding', 'effortless.']}
              italicClass="italic font-normal text-amber"
              className="mt-4 font-serif font-semibold text-4xl sm:text-6xl leading-[1.02] text-ivory"
            />
            <p className="mt-5 text-base sm:text-lg text-ivory/85">
              Step inside the interactive SignBridge AI walkthrough preview to explore domain contexts, camera framing guidance, sentence review, and English/Hindi speech synthesis.
            </p>
            <div className="mt-9">
              <button
                type="button"
                onClick={() => onOpenWalkthrough && onOpenWalkthrough('Everyday Conversations')}
                className="group inline-flex items-center gap-3 rounded-full bg-coral px-8 py-4 text-xs sm:text-sm font-semibold uppercase tracking-[0.14em] text-ivory shadow-editorial transition-all duration-300 hover:bg-amber hover:text-midnight hover:-translate-y-0.5"
              >
                <span>Enter Prototype Walkthrough</span>
                <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

