import React, { useState } from 'react';
import { View } from './types';
import Header from './components/Header';
import Hero from './components/Hero';
import SpeechToISL from './components/SpeechToISL';
import ISLToSpeech from './components/ISLToSpeech';
import AICoach from './components/AICoach';
import Footer from './components/Footer';

const App: React.FC = () => {
  const [view, setView] = useState<View>(View.HOME);

  const renderView = () => {
    switch (view) {
      case View.SPEECH_TO_ISL:
        return <SpeechToISL />;
      case View.ISL_TO_SPEECH:
        return <ISLToSpeech />;
      case View.AI_COACH:
        return <AICoach />;
      case View.HOME:
      default:
        return <Hero setView={setView} />;
    }
  };
  
  const backgroundClass = view === View.HOME
    ? 'bg-gradient-to-br from-blue-50 via-green-50 to-orange-50'
    : 'bg-background';

  return (
    <div className={`min-h-screen ${backgroundClass} text-text-primary flex flex-col font-sans`}>
      <Header view={view} setView={setView} />
      <main className="flex-grow container mx-auto p-4 md:p-8">
        <div key={view} className="animate-fadeIn">
          {renderView()}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default App;