import React from 'react';
import { View } from '../types';

interface HeroProps {
  setView: (view: View) => void;
}

const MicrophoneIcon: React.FC = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
  </svg>
);

const VideoCameraIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9A2.25 2.25 0 0 0 13.5 5.25h-9A2.25 2.25 0 0 0 2.25 7.5v9A2.25 2.25 0 0 0 4.5 18.75Z" />
    </svg>
);

const GraduationCapIcon: React.FC = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-primary mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.627 48.627 0 0 1 12 20.904a48.627 48.627 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.57 50.57 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.902 59.902 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0 1 12 13.489a50.702 50.702 0 0 1 7.74-3.342M6.75 15a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm0 0v-3.675A55.378 55.378 0 0 1 12 8.443m-7.007 11.55A5.981 5.981 0 0 0 6.75 15.75v-1.5" />
    </svg>
);

const Feature: React.FC<{ icon?: React.ReactNode; title: string; description: string; action: () => void }> = ({ icon, title, description, action }) => (
  <div className="rounded-xl shadow-md hover:shadow-xl hover:-translate-y-1 transition-all duration-300 bg-gradient-to-br from-primary via-secondary to-accent p-0.5">
    <div className="bg-white p-6 rounded-lg h-full flex flex-col items-center text-center">
      {icon}
      <h3 className="text-xl font-semibold mb-2 text-text-primary">{title}</h3>
      <p className="text-text-secondary mb-4 flex-grow">{description}</p>
      <button onClick={action} className="text-primary font-semibold hover:underline mt-auto">
        Try Now &rarr;
      </button>
    </div>
  </div>
);


const Hero: React.FC<HeroProps> = ({ setView }) => {
  return (
    <div className="text-center">
      <div className="rounded-xl shadow-xl p-1 bg-gradient-to-br from-primary via-secondary to-accent mb-8">
        <div className="bg-white rounded-lg p-8 md:p-12">
          <h1 className="text-4xl md:text-5xl font-extrabold text-text-primary mb-4">
            Breaking Communication Barriers
          </h1>
          <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-8">
            Connecting the Deaf and Hearing Communities through AI-powered Indian Sign Language tools.
          </p>
          <button
            onClick={() => setView(View.AI_COACH)}
            className="bg-accent hover:opacity-90 text-white font-bold py-3 px-8 rounded-lg transition-opacity text-lg"
          >
            Start Learning ISL
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        <Feature
          icon={<MicrophoneIcon />}
          title="Speech to ISL"
          description="Convert spoken words into real-time ISL gesture animations."
          action={() => setView(View.SPEECH_TO_ISL)}
        />
        <Feature
          icon={<VideoCameraIcon />}
          title="ISL to Speech"
          description="Use your webcam to translate ISL gestures into spoken language."
          action={() => setView(View.ISL_TO_SPEECH)}
        />
        <Feature
          icon={<GraduationCapIcon />}
          title="AI-Powered ISL Coach"
          description="Learn ISL through guided lessons and interactive feedback from our AI."
          action={() => setView(View.AI_COACH)}
        />
      </div>
    </div>
  );
};

export default Hero;