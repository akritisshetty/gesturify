// src/components/declaration.d.ts

// Existing declarations for other image types (if any)
declare module '*.png' {
  const value: string;
  export default value;
}

// THIS IS THE CRUCIAL DECLARATION FOR YOUR CURRENT ERROR
declare module '*.jpg' {
  const value: string;
  export default value;
}

// Add other image types if you use them, e.g.:
declare module '*.jpeg' {
  const value: string;
  export default value;
}
declare module '*.gif' {
  const value: string;
  export default value;
}
declare module '*.svg' {
  import React = require('react');
  export const ReactComponent: React.FC<React.SVGProps<SVGSVGElement>>;
  const src: string;
  export default src;
}