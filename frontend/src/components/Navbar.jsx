import React, { useState } from 'react';
import { ArrowUpRight, Menu, X } from 'lucide-react';

export default function Navbar({ onOpenWalkthrough }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => setMobileMenuOpen((prev) => !prev);
  const closeMobileMenu = () => setMobileMenuOpen(false);

  const handleLaunchClick = (domain = 'Everyday Conversations') => {
    closeMobileMenu();
    if (onOpenWalkthrough) onOpenWalkthrough(domain);
  };

  return (
    <header className="fixed top-0 inset-x-0 z-40 py-4 px-4 sm:px-8 transition-all duration-300">
      <div className="max-w-7xl mx-auto">
        <nav
          aria-label="Primary Navigation"
          className="flex items-center justify-between rounded-full bg-ivory/88 backdrop-blur-md border border-subtleborder px-5 sm:px-7 py-3 shadow-pill"
        >
          {/* Brand Emblem & Wordmark */}
          <a href="#hero" className="flex items-center gap-3 group">
            <span className="relative flex h-9 w-9 items-center justify-center rounded-full bg-midnight text-ivory shadow-sm transition-transform duration-500 group-hover:rotate-12">
              <svg viewBox="0 0 28 28" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 18 C9 8, 13 20, 17 10 C19 6, 22 12, 24 9" stroke="#D6AD70" strokeLinecap="round" />
                <circle cx="5" cy="18" r="2" fill="#6D263D" stroke="none" />
                <circle cx="24" cy="9" r="2" fill="#FAF5EF" stroke="none" />
              </svg>
            </span>
            <span className="font-serif font-bold text-xl sm:text-2xl tracking-tight text-midnight">
              SignBridge <span className="italic font-normal text-coral">AI</span>
            </span>
          </a>

          {/* Desktop Anchor Links */}
          <ul className="hidden lg:flex items-center gap-7 text-xs font-semibold uppercase tracking-[0.14em] text-midnight/80">
            <li><a href="#story" className="hover:text-coral transition-colors py-1">Our Story</a></li>
            <li><a href="#visual-journey" className="hover:text-slatecobalt transition-colors py-1">Visual Journey</a></li>
            <li><a href="#how-it-works" className="hover:text-amber transition-colors py-1">How It Works</a></li>
            <li><a href="#domains" className="hover:text-coral transition-colors py-1">Seven Domains</a></li>
            <li><a href="#trust" className="hover:text-midnight transition-colors py-1">Trust &amp; Privacy</a></li>
          </ul>

          {/* Launch Prototype CTA */}
          <div className="hidden sm:flex items-center gap-3">
            <button
              type="button"
              onClick={() => handleLaunchClick('Everyday Conversations')}
              className="inline-flex items-center gap-2 rounded-full bg-midnight px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.12em] text-ivory transition-all duration-300 hover:bg-coral hover:-translate-y-0.5 shadow-sm"
            >
              <span>Launch Prototype Preview</span>
              <ArrowUpRight className="w-4 h-4 text-amber" />
            </button>
          </div>

          {/* Mobile Menu Toggle */}
          <button
            type="button"
            onClick={toggleMobileMenu}
            aria-expanded={mobileMenuOpen}
            aria-controls="mobile-nav-panel"
            aria-label="Open Navigation Menu"
            className="lg:hidden inline-flex items-center justify-center rounded-full border border-subtleborder bg-ivory p-2.5 text-midnight focus:outline-none"
          >
            {mobileMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </nav>

        {/* Mobile Dropdown Drawer */}
        {mobileMenuOpen && (
          <div
            id="mobile-nav-panel"
            className="lg:hidden mt-2 rounded-3xl bg-ivory/96 backdrop-blur-md border border-subtleborder p-6 shadow-editorialLg"
          >
            <ul className="flex flex-col gap-3.5 text-sm font-semibold uppercase tracking-[0.12em] text-midnight">
              <li>
                <a href="#story" onClick={closeMobileMenu} className="block py-1 hover:text-coral">
                  Our Story
                </a>
              </li>
              <li>
                <a href="#visual-journey" onClick={closeMobileMenu} className="block py-1 hover:text-slatecobalt">
                  Visual Journey
                </a>
              </li>
              <li>
                <a href="#how-it-works" onClick={closeMobileMenu} className="block py-1 hover:text-amber">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#domains" onClick={closeMobileMenu} className="block py-1 hover:text-coral">
                  Seven Domains
                </a>
              </li>
              <li>
                <a href="#trust" onClick={closeMobileMenu} className="block py-1 hover:text-midnight">
                  Trust &amp; Privacy
                </a>
              </li>
            </ul>
            <button
              type="button"
              onClick={() => handleLaunchClick('Everyday Conversations')}
              className="mt-5 w-full inline-flex items-center justify-center gap-2 rounded-full bg-midnight px-5 py-3 text-xs font-semibold uppercase tracking-[0.12em] text-ivory hover:bg-coral transition-colors"
            >
              <span>Launch Prototype Preview</span>
              <ArrowUpRight className="w-4 h-4 text-amber" />
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
