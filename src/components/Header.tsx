import React from 'react';
import { View } from '../types';
import Logo from './Logo';

interface HeaderProps {
  view: View;
  setView: (view: View) => void;
}

const NavLink: React.FC<{ onClick: () => void; children: React.ReactNode; isActive: boolean }> = ({ onClick, children, isActive }) => (
  <button
    onClick={onClick}
    className={`font-medium transition-colors relative py-2 ${isActive ? 'text-primary' : 'text-text-secondary hover:text-primary'}`}
    aria-current={isActive ? 'page' : undefined}
  >
    {children}
    {isActive && <span className="absolute bottom-0 left-0 w-full h-0.5 bg-primary rounded-full"></span>}
  </button>
);

const Header: React.FC<HeaderProps> = ({ view, setView }) => {
  return (
    <header className="bg-white/80 backdrop-blur-sm shadow-sm sticky top-0 z-50">
      <nav className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div 
            className="flex-shrink-0 cursor-pointer flex items-center gap-2"
            onClick={() => setView(View.HOME)}
          >
            <Logo className="h-10 w-auto" />
            <span className="text-xl font-bold text-text-primary">Gesturify</span>
          </div>
          <div className="hidden md:flex items-center space-x-6">
            <NavLink onClick={() => setView(View.SPEECH_TO_ISL)} isActive={view === View.SPEECH_TO_ISL}>Speech to ISL</NavLink>
            <NavLink onClick={() => setView(View.ISL_TO_SPEECH)} isActive={view === View.ISL_TO_SPEECH}>ISL to Speech</NavLink>
            <NavLink onClick={() => setView(View.AI_COACH)} isActive={view === View.AI_COACH}>AI Coach</NavLink>
          </div>
          <div className="flex md:hidden">
            {/* Mobile menu button can be added here */}
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;