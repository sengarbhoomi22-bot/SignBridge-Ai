import React, { useState, useCallback } from 'react';
import ThreeDVisual from './components/ThreeDVisual';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import VisualJourney from './components/VisualJourney';
import Story from './components/Story';
import Workflow from './components/Workflow';
import DomainShowcase from './components/DomainShowcase';
import TrustPrivacy from './components/TrustPrivacy';
import FinalCTA from './components/FinalCTA';
import Footer from './components/Footer';
import WalkthroughModal from './components/WalkthroughModal';

export default function App() {
  const [stageIndex, setStageIndex] = useState(0);
  const [pulseTrigger, setPulseTrigger] = useState(0);
  const [walkthroughOpen, setWalkthroughOpen] = useState(false);
  const [walkthroughDomain, setWalkthroughDomain] = useState('Everyday Conversations');

  const handleSetStage = useCallback((idx) => {
    setStageIndex(idx);
    setPulseTrigger((prev) => prev + 1);
  }, []);

  const handlePulse = useCallback(() => {
    setPulseTrigger((prev) => prev + 1);
  }, []);

  const handleOpenWalkthrough = useCallback((domain = 'Everyday Conversations') => {
    setWalkthroughDomain(domain);
    setWalkthroughOpen(true);
  }, []);

  const handleCloseWalkthrough = useCallback(() => {
    setWalkthroughOpen(false);
  }, []);

  return (
    <div className="relative min-h-screen bg-ivory text-midnight antialiased selection:bg-coral selection:text-ivory">
      {/* Persistent Full-Viewport 3D Background */}
      <ThreeDVisual stageIndex={stageIndex} pulseTrigger={pulseTrigger} />

      {/* Floating Pill Navigation */}
      <Navbar onOpenWalkthrough={handleOpenWalkthrough} />

      {/* Main Narrative Content */}
      <main id="narrative-root" className="relative z-10">
        <Hero
          stageIndex={stageIndex}
          onSetStage={handleSetStage}
          onPulse={handlePulse}
          onOpenWalkthrough={handleOpenWalkthrough}
        />
        <VisualJourney onSetStage={handleSetStage} />
        <Story />
        <Workflow onSetStage={handleSetStage} />
        <DomainShowcase
          onOpenWalkthrough={handleOpenWalkthrough}
          onPulse={handlePulse}
        />
        <TrustPrivacy />
        <FinalCTA onOpenWalkthrough={handleOpenWalkthrough} />
      </main>

      {/* Editorial Footer */}
      <Footer />

      {/* Interactive Prototype Walkthrough Modal */}
      <WalkthroughModal
        isOpen={walkthroughOpen}
        onClose={handleCloseWalkthrough}
        initialDomain={walkthroughDomain}
      />
    </div>
  );
}

