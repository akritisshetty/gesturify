import React from 'react';

// The application will now look for your logo file in this path.
// Please create an 'assets' folder in 'src' and place your 'logo.jpg' file there.
import logoImage from '../assets/logo.jpg';

// The component has been updated to render an HTML <img> tag.
// It now accepts image-specific attributes for better control.
const Logo: React.FC<React.ImgHTMLAttributes<HTMLImageElement>> = (props) => (
  <img 
    src={logoImage} 
    alt="Gesturify Logo" 
    {...props} 
  />
);

export default Logo;
