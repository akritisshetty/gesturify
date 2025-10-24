// This file tells TypeScript how to handle image imports.
// It declares that any import ending in .jpg, .jpeg, .png, or .svg
// is a module that exports a string (the image's path).

declare module '*.jpg' {
  const value: string;
  export default value;
}

declare module '*.jpeg' {
  const value: string;
  export default value;
}

declare module '*.png' {
  const value: string;
  export default value;
}

declare module '*.svg' {
  const value: string;
  export default value;
}
