import React, { useEffect, useRef, useState } from 'react';

export default function KineticHeading({
  lines,
  italics = [],
  className = '',
  italicClass = 'italic font-normal text-coral',
  as: Component = 'h2',
}) {
  const headingRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const el = headingRef.current;
    if (!el) return;

    // Check prefers-reduced-motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.14 }
    );

    observer.observe(el);

    return () => {
      if (el) observer.unobserve(el);
    };
  }, []);

  const fullText = lines.join(' ');
  let letterIdx = 0;

  return (
    <Component
      ref={headingRef}
      className={`kinetic-heading ${isVisible ? 'k-visible' : ''} ${className}`}
      aria-label={fullText}
    >
      {lines.map((line, lineIndex) => (
        <span key={lineIndex} className="k-line" aria-hidden="true">
          {line.split(' ').map((word, wordIndex) => {
            const isItalic = italics.includes(word);
            return (
              <span
                key={wordIndex}
                className={`k-word ${isItalic ? italicClass : ''}`}
              >
                {word.split('').map((ch, charIndex) => {
                  const delay = (letterIdx++ * 0.022).toFixed(3);
                  return (
                    <span
                      key={charIndex}
                      className="k-char"
                      style={{ transitionDelay: `${delay}s` }}
                    >
                      {ch}
                    </span>
                  );
                })}
              </span>
            );
          })}
        </span>
      ))}
    </Component>
  );
}

