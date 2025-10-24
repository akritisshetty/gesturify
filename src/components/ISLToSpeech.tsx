import React, { useState, useRef, useEffect } from 'react';

const ISLToSpeech: React.FC = () => {
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [recognizedText, setRecognizedText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionIntervalRef = useRef<number | null>(null);

  const mockRecognitions = [
      "Hello", "Thank you", "How are you?", "Help", "Water", "Food", "Yes", "No", "My name is..."
  ];

  const startCamera = async () => {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setError(null);
      } else {
        setError("Your browser does not support camera access.");
      }
    } catch (err) {
      console.error("Error accessing camera: ", err);
      setError("Camera access was denied. Please allow camera permissions in your browser settings.");
    }
  };
  
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const handleRecognitionToggle = () => {
    if (isRecognizing) {
      // Stop recognition
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
        recognitionIntervalRef.current = null;
      }
      stopCamera();
      setRecognizedText('');
    } else {
      // Start recognition
      startCamera();
      recognitionIntervalRef.current = window.setInterval(() => {
        // In a real app, you would send frames to your ML model here.
        const randomSign = mockRecognitions[Math.floor(Math.random() * mockRecognitions.length)];
        setRecognizedText(prev => `${prev} ${randomSign}`);
      }, 2000);
    }
    setIsRecognizing(!isRecognizing);
  };

  useEffect(() => {
    return () => {
      // Cleanup on component unmount
      stopCamera();
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
      }
    };
  }, []);

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">ISL to Speech Translator</h2>
      
      <div className="relative w-full aspect-video bg-gray-800 rounded-md overflow-hidden mb-4">
        <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover transform -scale-x-100"></video>
        {!isRecognizing && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60">
             <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-white/50 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.55a2.5 2.5 0 010 4.1L15 18M3 8a2 2 0 012-2h4a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
             </svg>
            <p className="text-white font-medium">Camera is off</p>
          </div>
        )}
      </div>

      {error && <p className="text-center text-red-500 mb-4">{error}</p>}

      <div className="flex items-center justify-center mb-4">
        <button
          onClick={handleRecognitionToggle}
          className={`flex items-center justify-center gap-2 px-6 py-2 font-semibold rounded-md text-white transition-colors ${
            isRecognizing ? 'bg-red-500 hover:bg-red-600' : 'bg-primary hover:opacity-90'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V7a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span>{isRecognizing ? 'Stop Recognition' : 'Start Recognition'}</span>
        </button>
      </div>

      <div className="w-full bg-gray-50 p-4 rounded-md min-h-[6rem] flex items-center justify-center text-center border border-gray-200">
        {recognizedText ? (
            <p className="text-text-secondary">{recognizedText}</p>
        ) : (
            <div className="text-gray-400">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mx-auto text-gray-300 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p>Recognized text will appear here...</p>
            </div>
        )}
      </div>
    </div>
  );
};

export default ISLToSpeech;