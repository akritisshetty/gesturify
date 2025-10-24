import React, { useState } from 'react';
import Logo from './Logo';
import FeedbackModal from './FeedbackModal';

const Footer: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <>
      <footer className="bg-text-primary text-white mt-8">
        <div className="container mx-auto py-6 px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center text-center">
          <div className="flex justify-center items-center gap-2 mb-4 sm:mb-0">
              <Logo className="h-10 w-10" />
              <span className="font-bold text-lg">Gesturify</span>
          </div>
          <button
            onClick={() => setIsModalOpen(true)}
            className="bg-primary hover:opacity-90 text-white font-semibold py-2 px-4 rounded-md transition-opacity"
          >
            Provide Feedback
          </button>
        </div>
      </footer>
      <FeedbackModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </>
  );
};

export default Footer;