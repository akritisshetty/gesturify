import type { Lesson } from '../types';

const mockLessons: { [key: string]: Lesson } = {
  'Greetings': {
    lessonTitle: 'Greetings',
    signs: [
      { name: 'Hello / Namaste', description: 'Hold your right hand up to your forehead with your palm facing out, then move it outwards and slightly downwards in a saluting motion.'},
      { name: 'Good Morning', description: 'Sign for "Good" (touch chin with flat hand, move forward) followed by "Morning" (non-dominant arm horizontal, dominant hand moves up from under it like the sun rising).' },
      { name: 'Thank You', description: 'With a flat hand, touch your chin and then move your hand forward and slightly upwards towards the person you are thanking.' },
      { name: 'Sorry', description: 'Make a fist with your right hand and rub it in a circular motion over your heart.' },
    ]
  },
  'Family': {
    lessonTitle: 'Family',
    signs: [
      { name: 'Family', description: 'Form "F" handshapes with both hands. Bring them together so the circles formed by your index fingers and thumbs touch, then circle them outwards until your pinkies touch.'},
      { name: 'Mother', description: 'With an open right hand (5-handshape), tap your thumb to the side of your chin two times.' },
      { name: 'Father', description: 'With an open right hand (5-handshape), tap your thumb to the side of your forehead two times.' },
    ]
  },
  'Common Questions': {
    lessonTitle: 'Common Questions',
    signs: [
        { name: 'Who?', description: 'Place your thumb on your chin and wiggle your index finger. Your facial expression should be questioning.' },
        { name: 'What?', description: 'Hold both hands open, palms up, and shake them side-to-side a few times. Furrow your eyebrows.' },
        { name: 'Where?', description: 'Shake your index finger back and forth. Furrow your eyebrows.' },
        { name: 'Why?', description: 'Touch your forehead with the middle finger of one hand, then move the hand away, changing it to the "Y" handshape.' },
    ]
  },
  'Days of the Week': {
    lessonTitle: 'Days of the Week',
    signs: [
        { name: 'Day', description: 'Hold your non-dominant arm horizontally in front of you. Rest your dominant elbow on the back of your non-dominant hand, and move your dominant arm downwards in an arc.' },
        { name: 'Monday', description: 'Form an "M" handshape and move it in a small circle.'},
        { name: 'Tuesday', description: 'Form a "T" handshape and move it in a small circle.' },
    ]
  },
  'Food & Drink': {
    lessonTitle: 'Food & Drink',
    signs: [
        { name: 'Food', description: 'With a flattened "O" handshape, tap your fingertips to your mouth a couple of times.'},
        { name: 'Water', description: 'Make a "W" handshape and tap your index finger to the side of your mouth.' },
        { name: 'Eat', description: 'Same as "Food", tap your fingertips to your mouth a couple of times.'},
    ]
  },
  'Feelings': {
    lessonTitle: 'Feelings',
    signs: [
        { name: 'Happy', description: 'Brush upwards on your chest a few times with a flat hand. Smile while signing.' },
        { name: 'Sad', description: 'Hold both open hands in front of your face, palms in, and move them downwards. Your facial expression should be sad.' },
        { name: 'Angry', description: 'Form claw shapes with both hands and move them up to the sides of your head as if you are enraged. Your facial expression should be angry.'  },
    ]
  }
};

const defaultLesson = (topic: string): Lesson => ({
    lessonTitle: `Mock Lesson: ${topic}`,
    signs: [
        { name: "Sign A (Mock)", description: "This is mock data because a specific lesson for this topic was not found."},
        { name: "Sign B (Mock)", description: "This demonstrates the fallback functionality." },
    ]
});

export const getLessonContent = (topic: string): Promise<Lesson> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(mockLessons[topic] || defaultLesson(topic));
    }, 300 + Math.random() * 400); // Simulate network delay
  });
};
