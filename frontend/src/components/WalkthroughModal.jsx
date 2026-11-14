import React, { useState, useEffect, useRef } from 'react';
import { X, Camera, Volume2 } from 'lucide-react';
import { CONTEXT_SAMPLES } from '../utils/constants';

export default function WalkthroughModal({
  isOpen,
  onClose,
  initialDomain = 'Everyday Conversations',
}) {
  const [domain, setDomain] = useState(initialDomain);
  const [selectedLang, setSelectedLang] = useState('en-IN');
  const [sentence, setSentence] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStatus, setCameraStatus] = useState(
    'Camera is off. Permission is requested only when you click Start.'
  );
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechStatus, setSpeechStatus] = useState(
    'Speech plays only after clicking Confirm & Speak.'
  );

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const modalContentRef = useRef(null);

  // Sync initialDomain when modal opens
  useEffect(() => {
    if (isOpen) {
      const targetDomain = CONTEXT_SAMPLES[initialDomain]
        ? initialDomain
        : 'Everyday Conversations';
      setDomain(targetDomain);
      setSentence(CONTEXT_SAMPLES[targetDomain][selectedLang] || '');
      setSpeechStatus('Speech plays only after clicking Confirm & Speak.');
    }
  }, [isOpen, initialDomain]);

  // Update sample sentence when domain or language changes, unless user typed custom text
  const handleDomainChange = (e) => {
    const newDomain = e.target.value;
    setDomain(newDomain);
    if (CONTEXT_SAMPLES[newDomain]) {
      setSentence(CONTEXT_SAMPLES[newDomain][selectedLang]);
    }
  };

  const handleLangSelect = (langCode) => {
    setSelectedLang(langCode);
    if (CONTEXT_SAMPLES[domain]) {
      setSentence(CONTEXT_SAMPLES[domain][langCode]);
    }
  };

  // Safe camera cleanup helper
  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {
          console.warn('Error stopping track:', e);
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setCameraStatus('Camera stopped. No raw video was saved.');
  };

  // Toggle camera
  const handleToggleCamera = async () => {
    if (cameraActive) {
      stopCameraStream();
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraStatus('Camera API is not supported in this browser environment.');
      return;
    }

    try {
      setCameraStatus('Requesting on-demand camera permission...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
      setCameraStatus('Camera active for framing check. Not stored.');
    } catch (err) {
      console.warn('Camera access denied or unavailable:', err);
      setCameraActive(false);
      setCameraStatus('Camera permission denied or camera unavailable.');
    }
  };

  // Stop camera stream & speech synthesis on modal close or unmount
  const handleModalClose = () => {
    stopCameraStream();
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
    onClose();
  };

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        handleModalClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopCameraStream();
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  // Speech synthesis handling
  const handleSpeak = () => {
    const textToSpeak = sentence.trim();
    if (!textToSpeak) {
      setSpeechStatus('Please review or enter a sentence before playback.');
      return;
    }

    if (!('speechSynthesis' in window)) {
      setSpeechStatus('Browser SpeechSynthesis is unavailable in this environment.');
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.lang = selectedLang;
    utterance.rate = 0.96;

    // Voice matching preference
    const voices = window.speechSynthesis.getVoices();
    const targetLangPrefix = selectedLang.toLowerCase().split('-')[0];
    const matchedVoice = voices.find((v) =>
      v.lang.toLowerCase().replace('_', '-').startsWith(targetLangPrefix)
    );
    if (matchedVoice) {
      utterance.voice = matchedVoice;
    }

    utterance.onstart = () => {
      setIsSpeaking(true);
      setSpeechStatus(
        `Speaking confirmed ${selectedLang === 'hi-IN' ? 'Hindi' : 'English'} output...`
      );
    };

    utterance.onend = () => {
      setIsSpeaking(false);
      setSpeechStatus('Playback complete. Edit and replay anytime.');
    };

    utterance.onerror = (e) => {
      setIsSpeaking(false);
      console.warn('Speech synthesis event:', e);
      setSpeechStatus('Speech synthesis finished or voice unavailable.');
    };

    window.speechSynthesis.speak(utterance);
  };

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="wt-modal-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) handleModalClose();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-midnight/65 backdrop-blur-sm"
    >
      <div
        ref={modalContentRef}
        className="relative w-full max-w-2xl rounded-[2.25rem] bg-ivory border border-subtleborder p-6 sm:p-9 shadow-editorialLg max-h-[92vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 pb-5 border-b border-subtleborder">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full bg-amber/35 border border-subtleborder px-3.5 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-midnight">
              UI Walkthrough Preview (Not Live AI Inference)
            </span>
            <h2 id="wt-modal-title" className="mt-3 font-serif font-semibold text-2xl sm:text-3xl text-midnight">
              SignBridge AI Interactive Flow
            </h2>
          </div>
          <button
            type="button"
            onClick={handleModalClose}
            aria-label="Close Prototype Walkthrough"
            className="rounded-full border border-subtleborder bg-white p-2.5 text-midnight hover:bg-coral hover:text-ivory transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Honest Scope Notice */}
        <p className="mt-4 text-xs sm:text-sm leading-relaxed text-midnight/85 bg-white border border-subtleborder rounded-2xl p-4">
          <strong>Honest Scope Notice:</strong> No live ML model backend is connected in this landing page preview, so automated sign prediction is not simulated. You can test domain selection, on-demand camera permission framing, sentence editing, and real browser speech synthesis in English or Hindi after confirmation.
        </p>

        <div className="mt-6 space-y-5">
          {/* 1. Domain Context */}
          <div>
            <label
              htmlFor="wt-domain-select"
              className="block font-mono text-xs font-semibold uppercase tracking-wider text-midnight/75 mb-2"
            >
              1. Communication Context
            </label>
            <select
              id="wt-domain-select"
              value={domain}
              onChange={handleDomainChange}
              className="w-full rounded-2xl bg-white border border-subtleborder px-4 py-3 text-sm font-semibold text-midnight focus:outline-none focus:border-coral"
            >
              <option value="Everyday Conversations">Everyday Conversations</option>
              <option value="Education">Education</option>
              <option value="Healthcare">Healthcare</option>
              <option value="Banking">Banking</option>
              <option value="Workplaces">Workplaces</option>
              <option value="Travel">Travel</option>
              <option value="Government & Public Services">Government &amp; Public Services</option>
            </select>
          </div>

          {/* 2. On-Demand Camera Framing Preview */}
          <div className="rounded-2xl bg-white border border-subtleborder p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <span className="font-mono text-xs font-semibold uppercase tracking-wider text-midnight">
                  2. Camera Permission &amp; Framing Guide
                </span>
                <p className="text-xs text-midnight/70 mt-0.5">{cameraStatus}</p>
              </div>
              <button
                type="button"
                onClick={handleToggleCamera}
                className="inline-flex items-center gap-2 rounded-full bg-slatecobalt text-ivory px-4 py-2 text-xs font-semibold uppercase tracking-wider hover:bg-coral transition-colors"
              >
                <Camera className="w-3.5 h-3.5" />
                <span>{cameraActive ? 'Stop Camera' : 'Start Camera Preview'}</span>
              </button>
            </div>

            {cameraActive && (
              <div className="mt-4 relative rounded-2xl overflow-hidden bg-midnight aspect-video flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-4 rounded-xl border border-dashed border-amber flex items-end justify-center pb-2.5"
                >
                  <span className="rounded-full bg-midnight/85 px-3 py-1 font-mono text-[10px] text-amber">
                    Keep hands &amp; upper posture inside framing guide
                  </span>
                </div>
                <span className="absolute top-3 left-3 inline-flex items-center gap-1.5 rounded-full bg-coral px-3 py-1 font-mono text-[10px] font-semibold text-ivory">
                  <span className="h-2 w-2 rounded-full bg-amber animate-ping" />
                  LIVE PREVIEW · NOT STORED
                </span>
              </div>
            )}
          </div>

          {/* 3. Review & Edit Sentence + Confirmed Speech */}
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <label
                htmlFor="wt-sentence"
                className="font-mono text-xs font-semibold uppercase tracking-wider text-midnight/75"
              >
                3. Review &amp; Edit Sentence Before Speech
              </label>
              <div className="inline-flex rounded-full bg-white border border-subtleborder p-0.5 text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => handleLangSelect('en-IN')}
                  className={`rounded-full px-3.5 py-1 transition-all ${
                    selectedLang === 'en-IN'
                      ? 'bg-coral text-ivory'
                      : 'text-midnight hover:text-coral'
                  }`}
                >
                  English
                </button>
                <button
                  type="button"
                  onClick={() => handleLangSelect('hi-IN')}
                  className={`rounded-full px-3.5 py-1 transition-all ${
                    selectedLang === 'hi-IN'
                      ? 'bg-coral text-ivory'
                      : 'text-midnight hover:text-coral'
                  }`}
                >
                  हिन्दी (Hindi)
                </button>
              </div>
            </div>

            <textarea
              id="wt-sentence"
              rows={2}
              value={sentence}
              onChange={(e) => setSentence(e.target.value)}
              className="w-full rounded-2xl bg-white border border-subtleborder p-3.5 text-sm sm:text-base text-midnight leading-relaxed focus:outline-none focus:border-coral"
            />

            {/* Visual Acoustic Bar Feedback during Speech */}
            {isSpeaking && (
              <div className="mt-3 rounded-xl bg-white border border-subtleborder px-4 py-2.5 flex items-center justify-between">
                <span className="font-mono text-[11px] font-semibold text-coral">
                  Synthesizing Confirmed Speech...
                </span>
                <div className="flex items-center gap-1 h-4">
                  <span className="w-1 h-full bg-coral rounded-full svg-wave-bar" />
                  <span
                    className="w-1 h-full bg-amber rounded-full svg-wave-bar"
                    style={{ animationDelay: '0.2s' }}
                  />
                  <span
                    className="w-1 h-full bg-slatecobalt rounded-full svg-wave-bar"
                    style={{ animationDelay: '0.4s' }}
                  />
                  <span
                    className="w-1 h-full bg-midnight rounded-full svg-wave-bar"
                    style={{ animationDelay: '0.6s' }}
                  />
                </div>
              </div>
            )}

            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <span className="text-xs text-midnight/75">{speechStatus}</span>
              <button
                type="button"
                onClick={handleSpeak}
                className="inline-flex items-center gap-2 rounded-full bg-coral px-5 py-2.5 text-xs sm:text-sm font-semibold uppercase tracking-wider text-ivory shadow-sm hover:bg-midnight transition-colors"
              >
                <Volume2 className="w-4 h-4" />
                <span>Confirm &amp; Speak</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
