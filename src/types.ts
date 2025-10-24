
export enum View {
  HOME = 'HOME',
  SPEECH_TO_ISL = 'SPEECH_TO_ISL',
  ISL_TO_SPEECH = 'ISL_TO_SPEECH',
  AI_COACH = 'AI_COACH',
}

export interface Sign {
  name: string;
  description: string;
  imageUrl?: string;
}

export interface Lesson {
  lessonTitle: string;
  signs: Sign[];
}