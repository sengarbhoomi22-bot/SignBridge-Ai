import React from 'react';
import {
  ShieldCheck,
  VideoOff,
  Edit3,
  RotateCcw,
  AlertCircle,
} from 'lucide-react';
import KineticHeading from './common/KineticHeading';

export default function TrustPrivacy() {
  return (
    <section id="trust" className="relative py-24 lg:py-32 bg-slatecobalt/14 backdrop-blur-md border-t border-subtleborder">
      <div className="max-w-7xl mx-auto px-4 sm:px-8">
        {/* Human Connection Editorial Banner */}
        <div className="rounded-[2.5rem] bg-slatecobalt text-ivory p-8 sm:p-12 lg:p-14 shadow-editorialLg grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          <div className="lg:col-span-7">
            <span className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-ivory/90">
              Human-Centred Accessibility
            </span>
            <KineticHeading
              as="h2"
              lines={['Technology should', 'bring people closer.']}
              italics={['closer.']}
              italicClass="italic font-normal text-coral"
              className="mt-3 font-serif font-semibold text-4xl sm:text-5xl lg:text-6xl leading-[1.02] text-ivory"
            />
            <p className="mt-5 text-base sm:text-lg leading-relaxed text-ivory/95 max-w-2xl">
              Designed for students, patients, travellers, families, and public-service staff across age groups—combining legible typography, calm motion, and complete user control over every spoken output.
            </p>
          </div>
          <div className="lg:col-span-5 flex justify-center">
            <div className="rounded-[2rem] bg-white text-midnight p-7 text-center max-w-xs w-full shadow-editorial border border-subtleborder">
              <span className="font-mono text-[11px] font-semibold uppercase tracking-widest text-coral">
                Bilingual Bridge
              </span>
              <div className="my-3 font-serif italic text-4xl sm:text-5xl font-semibold text-midnight">
                ISL ↔
              </div>
              <div className="font-serif text-2xl font-semibold text-slatecobalt">
                English · हिन्दी
              </div>
              <p className="mt-2 text-xs text-midnight/70">
                Confirmed by the signer prior to playback
              </p>
            </div>
          </div>
        </div>

        {/* Privacy, Safety & Trust Pillars */}
        <div className="mt-16">
          <div className="max-w-2xl">
            <span className="font-mono text-xs font-semibold uppercase tracking-[0.18em] text-midnight/70">
              Privacy, Safety &amp; Consent
            </span>
            <h3 className="mt-2 font-serif font-semibold text-3xl sm:text-5xl text-midnight">
              Built on transparency and care.
            </h3>
          </div>

          <div className="mt-10 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="rounded-[2rem] bg-white border border-subtleborder p-6 shadow-editorial">
              <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slatecobalt text-ivory">
                <ShieldCheck className="w-5 h-5" />
              </span>
              <h4 className="mt-5 font-serif text-2xl font-semibold text-midnight">On-Demand Camera</h4>
              <p className="mt-2 text-sm leading-relaxed text-midnight/80">
                Camera access is requested only when you actively start capture—never silently in the background.
              </p>
            </div>

            <div className="rounded-[2rem] bg-white border border-subtleborder p-6 shadow-editorial">
              <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-coral text-ivory">
                <VideoOff className="w-5 h-5" />
              </span>
              <h4 className="mt-5 font-serif text-2xl font-semibold text-midnight">No Default Storage</h4>
              <p className="mt-2 text-sm leading-relaxed text-midnight/80">
                Raw camera video is processed temporarily for inference and discarded by default rather than stored.
              </p>
            </div>

            <div className="rounded-[2rem] bg-white border border-subtleborder p-6 shadow-editorial">
              <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-amber text-midnight">
                <Edit3 className="w-5 h-5" />
              </span>
              <h4 className="mt-5 font-serif text-2xl font-semibold text-midnight">Edit Before Speech</h4>
              <p className="mt-2 text-sm leading-relaxed text-midnight/80">
                Users review and edit translated sentences in English or Hindi before confirming speech playback.
              </p>
            </div>

            <div className="rounded-[2rem] bg-white border border-subtleborder p-6 shadow-editorial">
              <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-midnight text-ivory">
                <RotateCcw className="w-5 h-5" />
              </span>
              <h4 className="mt-5 font-serif text-2xl font-semibold text-midnight">Honest Abstention</h4>
              <p className="mt-2 text-sm leading-relaxed text-midnight/80">
                Uncertain framing or low confidence prompts the user to repeat rather than guessing a sentence.
              </p>
            </div>
          </div>

          {/* Assistive Communication Aid Disclaimer */}
          <div className="mt-8 rounded-[2rem] bg-white border border-subtleborder p-6 sm:p-7 flex flex-col sm:flex-row items-start sm:items-center gap-4 shadow-sm">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-coral text-ivory">
              <AlertCircle className="w-6 h-6" />
            </span>
            <p className="text-sm sm:text-base leading-relaxed text-midnight">
              <strong>Assistive Communication Aid Notice:</strong> SignBridge AI is a supportive communication aid and is <strong>not</strong> a replacement for qualified human sign language interpreters in medical, legal, financial, or emergency situations.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

