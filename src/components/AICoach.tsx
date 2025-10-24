import React, { useState, useCallback } from 'react';
import { getLessonContent } from '../services/mockData';
import type { Lesson, Sign } from '../types';

const lessonTopics = [
  { title: 'Greetings', icon: '👋' },
  { title: 'Family', icon: '👨‍👩‍👧‍👦' },
  { title: 'Common Questions', icon: '❓' },
  { title: 'Days of the Week', icon: '🗓️' },
  { title: 'Food & Drink', icon: '🍎' },
  { title: 'Feelings', icon: '😊' },
];

const LoadingSpinner: React.FC = () => (
    <div className="flex justify-center items-center p-8">
      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
    </div>
);

const SignDetail: React.FC<{ sign: Sign; style?: React.CSSProperties }> = ({ sign, style }) => (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 mb-4 flex flex-col sm:flex-row items-start gap-4 animate-fadeIn" style={style}>
        <div className="w-full sm:w-48 h-48 flex-shrink-0 bg-gray-200 rounded-md flex items-center justify-center overflow-hidden">
            {sign.imageUrl ? (
                <img src={sign.imageUrl} alt={`Indian Sign Language for ${sign.name}`} className="w-full h-full object-cover" />
            ) : (
                <div className="w-full h-full bg-gray-200 animate-pulse"></div>
            )}
        </div>
        <div className="flex-grow">
            <h4 className="font-bold text-primary text-xl">{sign.name}</h4>
            <p className="text-text-secondary mt-2 whitespace-pre-wrap">{sign.description}</p>
        </div>
    </div>
);

const AICoach: React.FC = () => {
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLesson = useCallback(async (topic: string) => {
    if (selectedTopic === topic && lesson) {
        setSelectedTopic(null); // Allow collapsing the view
        setLesson(null);
        return;
    }
    setSelectedTopic(topic);
    setIsLoading(true);
    setError(null);
    setLesson(null);
    try {
      const lessonData = await getLessonContent(topic);
      setLesson(lessonData);
    } catch (err) {
      console.error(err);
      setError("Failed to load lesson. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [selectedTopic, lesson]);

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-text-primary">AI-Powered ISL Coach</h2>
        <p className="text-text-secondary mt-2">Select a topic to start your interactive learning journey.</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {lessonTopics.map((topic) => (
          <button
            key={topic.title}
            onClick={() => fetchLesson(topic.title)}
            className={`p-4 rounded-lg shadow-md hover:shadow-lg transform hover:-translate-y-1 transition-all duration-200 text-center border-2 ${
                selectedTopic === topic.title
                    ? 'border-primary ring-2 ring-primary/50 bg-primary/10'
                    : 'bg-white border-transparent hover:border-gray-300'
            }`}
          >
            <div className="text-3xl mb-2">{topic.icon}</div>
            <h3 className="font-semibold text-md text-text-primary">{topic.title}</h3>
          </button>
        ))}
      </div>

      {selectedTopic && (
        <div className="mt-8 p-6 bg-white rounded-lg shadow-md border border-gray-200">
          <h3 className="text-2xl font-bold text-text-primary mb-4 text-center">Lesson: {selectedTopic}</h3>
          {isLoading && <LoadingSpinner />}
          {error && <p className="text-center text-red-500">{error}</p>}
          {lesson && (
              <div>
                  {lesson.signs.map((sign, index) => (
                      <SignDetail 
                        key={index} 
                        sign={sign} 
                        style={{ animationDelay: `${index * 150}ms`, opacity: 0 }}
                      />
                  ))}
              </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AICoach;