import React, { useState, useEffect } from 'react';

const SpeechToISL: React.FC = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [animations, setAnimations] = useState<string[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState('en-IN');

  useEffect(() => {
    // Mock processing transcript to generate "animations"
    if (transcript) {
      const words = transcript.split(' ').filter(word => word.length > 0);
      setAnimations(words);
    } else {
      setAnimations([]);
    }
  }, [transcript]);

  const handleListen = () => {
    setIsListening(!isListening);
    if (!isListening) {
      // In a real app, you'd start speech recognition here.
      // We'll simulate it with a timeout.
      setTranscript('Simulating speech recognition...');
      setTimeout(() => {
        const mockTranscripts: { [key: string]: string } = {
          'en-IN': 'Hello how are you doing today',
          'hi-IN': 'नमस्ते आप आज कैसे हैं',
          'kn-IN': 'ನಮಸ್ಕಾರ ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಿ',
        };
        setTranscript(mockTranscripts[selectedLanguage]);
        setIsListening(false);
      }, 3000);
    } else {
      // Stop listening
      setTranscript('');
    }
  };
  
  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">Speech to ISL Translator</h2>
      
      <div className="flex flex-col md:flex-row items-center justify-center gap-4 mb-6">
        <div>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className="w-full md:w-auto bg-white border border-gray-300 text-gray-700 py-2 px-4 pr-8 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="en-IN">English</option>
            <option value="hi-IN">हिन्दी (Hindi)</option>
            <option value="kn-IN">ಕನ್ನಡ (Kannada)</option>
          </select>
        </div>

        <button
          onClick={handleListen}
          className={`flex items-center justify-center gap-2 px-6 py-2 font-semibold rounded-md text-white transition-colors ${
            isListening ? 'bg-red-500 hover:bg-red-600' : 'bg-primary hover:opacity-90'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          <span>{isListening ? 'Stop Listening' : 'Start Listening'}</span>
        </button>
      </div>

      <div className="w-full bg-gray-50 p-4 rounded-md min-h-[4rem] text-center border border-gray-200 mb-6">
        <p className="text-text-secondary italic">{transcript || "Your transcribed text will appear here..."}</p>
      </div>
      
      <div className="bg-gray-100 p-4 rounded-md min-h-[16rem]">
        <h3 className="text-md font-semibold mb-4 text-text-primary">ISL Animation Output:</h3>
        {animations.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {animations.map((word, index) => (
              <div key={index} className="flex flex-col items-center bg-white p-2 rounded-md shadow-sm text-center">
                <div className="w-full aspect-square bg-gray-200 rounded animate-pulse">
                    {/* REMOVED THE IMG TAG HERE */}
                    {/* You can add your actual image source here when ready */}
                    {/* <img 
                        src={`https://picsum.photos/seed/${word.replace(/\s/g, '')}/200`} 
                        alt={`Animation for ${word}`} 
                        className="w-full h-full object-cover rounded" 
                    /> */}
                    {/* This div will act as the empty placeholder */}
                    <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm">
                      Image Placeholder
                    </div>
                </div>
                <p className="mt-2 font-medium text-xs text-text-primary h-8 flex items-center">{word}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 pt-16">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
            </svg>
            <p>Animations will be displayed here.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SpeechToISL;