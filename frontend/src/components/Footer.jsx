import React from 'react';

export default function Footer() {
  return (
    <footer className="relative z-10 bg-ivory border-t border-subtleborder py-14">
      <div className="max-w-7xl mx-auto px-4 sm:px-8">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-10 pb-10 border-b border-subtleborder">
          <div className="md:col-span-5">
            <a href="#hero" className="inline-flex items-center gap-3">
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-midnight text-ivory font-serif font-bold">
                S
              </span>
              <span className="font-serif font-bold text-xl text-midnight">
                SignBridge <span className="italic font-normal text-coral">AI</span>
              </span>
            </a>
            <p className="mt-4 text-sm leading-relaxed text-midnight/75 max-w-sm">
              An AI-powered Indian Sign Language (ISL) to English and Hindi text and speech platform designed around consent, human review, and inclusive visual storytelling.
            </p>
          </div>

          <div className="md:col-span-3">
            <h3 className="font-mono text-xs font-semibold uppercase tracking-[0.16em] text-midnight/60">
              Navigation
            </h3>
            <ul className="mt-4 space-y-2.5 text-sm font-medium text-midnight/85">
              <li><a href="#hero" className="hover:text-coral transition-colors">Home</a></li>
              <li><a href="#visual-journey" className="hover:text-slatecobalt transition-colors">4-Act Visual Journey</a></li>
              <li><a href="#story" className="hover:text-coral transition-colors">Our Story</a></li>
              <li><a href="#how-it-works" className="hover:text-amber transition-colors">How It Works</a></li>
              <li><a href="#domains" className="hover:text-coral transition-colors">Seven Domains</a></li>
              <li><a href="#trust" className="hover:text-midnight transition-colors">Trust &amp; Privacy</a></li>
            </ul>
          </div>

          <div className="md:col-span-4">
            <h3 className="font-mono text-xs font-semibold uppercase tracking-[0.16em] text-midnight/60">
              Accessibility &amp; Scope
            </h3>
            <p className="mt-4 text-xs sm:text-sm leading-relaxed text-midnight/75">
              Supports keyboard navigation, visible focus states, reduced-motion preferences, and WebGL fallbacks. Assistive prototype only—not a substitute for qualified human interpreters.
            </p>
          </div>
        </div>

        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-midnight/60">
          <p>© 2026 SignBridge AI · Building a bridge between signing and understanding.</p>
          <p className="font-mono">Warm Ivory &amp; Rich Burgundy Editorial Edition</p>
        </div>
      </div>
    </footer>
  );
}
