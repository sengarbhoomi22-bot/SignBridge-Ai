import React from 'react';
import { Layers, UserCheck } from 'lucide-react';
import KineticHeading from './common/KineticHeading';

export default function Story() {
  return (
    <section id="story" className="relative py-24 lg:py-36 bg-midnight text-ivory overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-center">
          <div className="lg:col-span-6">
            <span className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-amber">
              Our Story &amp; Purpose
            </span>
            <KineticHeading
              as="h2"
              lines={['Different ways of expressing.', 'One way of understanding.']}
              italics={['expressing.', 'understanding.']}
              italicClass="italic font-normal text-amber"
              className="mt-4 font-serif font-semibold text-4xl sm:text-6xl leading-[1.02] text-ivory"
            />
            <p className="mt-6 text-base sm:text-lg leading-relaxed text-ivory/85">
              Most basic sign-language demos recognize isolated finger-spelled letters in a single frozen frame. Real conversations are continuous—woven from hand movement, body posture, facial grammar, and context.
            </p>
            <p className="mt-4 text-base sm:text-lg leading-relaxed text-ivory/80">
              SignBridge AI was conceived to bridge everyday interactions between Indian Sign Language users and non-signing listeners—while keeping human review at the center so technology assists rather than speaks over the signer.
            </p>
          </div>

          {/* Layered Editorial Contrast Composition */}
          <div className="lg:col-span-6 space-y-6">
            <div className="rounded-[2.25rem] bg-white text-midnight p-7 sm:p-9 shadow-editorialLg border border-subtleborder">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold uppercase tracking-widest text-coral">
                  01 · Beyond Isolated Alphabets
                </span>
                <Layers className="w-5 h-5 text-coral" />
              </div>
              <h3 className="mt-4 font-serif text-2xl sm:text-3xl font-semibold">
                Sequence-Aware Interpretation
              </h3>
              <p className="mt-2.5 text-sm sm:text-base leading-relaxed text-midnight/80">
                Designed around multi-frame temporal sequences for supported vocabulary rather than single-snapshot gesture gimmicks.
              </p>
            </div>

            <div className="rounded-[2.25rem] bg-coral/40 backdrop-blur-md text-ivory p-7 sm:p-9 border border-amber/30">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold uppercase tracking-widest text-amber">
                  02 · Signer Dignity &amp; Agency
                </span>
                <UserCheck className="w-5 h-5 text-amber" />
              </div>
              <h3 className="mt-4 font-serif text-2xl sm:text-3xl font-semibold">
                Mandatory Review Before Speech
              </h3>
              <p className="mt-2.5 text-sm sm:text-base leading-relaxed text-ivory/85">
                Every translated sentence can be edited in English or Hindi before speech playback—ensuring the signer owns every spoken word.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

