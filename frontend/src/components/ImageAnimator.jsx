import React, { useState, useEffect, useRef } from 'react';

const ImageAnimator = ({ 
  images = [], 
  isPlaying = false, 
  frameRate = 500, // milliseconds per frame (0.5 seconds)
  onAnimationComplete, // Callback when animation completes
  className = "",
  altText = "Animated frame"
}) => {
  const [currentFrame, setCurrentFrame] = useState(0);
  const animationRef = useRef(null);

  // Clean up function
  const cleanup = () => {
    if (animationRef.current) {
      clearTimeout(animationRef.current);
      animationRef.current = null;
    }
  };

  // Main animation function - runs once through all frames
  const animate = (frameIndex = 0) => {
    if (frameIndex >= images.length) {
      // Animation complete - notify parent and stop
      setCurrentFrame(0); // Reset to first frame
      if (onAnimationComplete) {
        onAnimationComplete();
      }
      return;
    }

    setCurrentFrame(frameIndex);
    
    // Schedule next frame
    animationRef.current = setTimeout(() => {
      animate(frameIndex + 1);
    }, frameRate);
  };

  // Start animation when isPlaying becomes true
  useEffect(() => {
    if (isPlaying) {
      cleanup(); // Clean any existing timers
      
      if (images.length <= 1) {
        setCurrentFrame(0);
        // Still notify completion even for single/no images
        setTimeout(() => {
          if (onAnimationComplete) onAnimationComplete();
        }, 100);
        return;
      }

      // Start animation from first frame
      animate(0);
    } else {
      // Stop animation and reset to first frame
      cleanup();
      setCurrentFrame(0);
    }

    return cleanup; // Cleanup on unmount or dependency change
  }, [isPlaying]);

  // Reset when images change
  useEffect(() => {
    cleanup();
    setCurrentFrame(0);
  }, [images]);

  // Cleanup on unmount
  useEffect(() => {
    return cleanup;
  }, []);

  // Get current image to display
  const getCurrentImage = () => {
    if (images.length === 0) return null;
    const safeIndex = Math.min(currentFrame, images.length - 1);
    return images[safeIndex];
  };

  const currentImage = getCurrentImage();

  if (!currentImage) {
    return (
      <div className={`aspect-square bg-gray-800 rounded flex items-center justify-center ${className}`}>
        <div className="text-gray-400 text-xs">No Image</div>
      </div>
    );
  }

  return (
    <div className={`aspect-square bg-black rounded overflow-hidden relative ${className}`}>
      <img
        src={currentImage.url}
        alt={`${altText} - Frame ${currentFrame + 1}`}
        className="w-full h-full object-cover"
        onError={(e) => {
          console.error('Image load error:', currentImage.filename);
          e.target.style.display = 'none';
        }}
      />
      
      {/* Frame indicator - always show when we have images */}
      {images.length > 1 && (
        <div className="absolute bottom-1 right-1 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
          {`${currentFrame + 1}/${images.length}`}
        </div>
      )}
      
      {/* Animation status indicator - only show when playing */}
      {isPlaying && images.length > 1 && (
        <div className="absolute top-1 left-1">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
        </div>
      )}
    </div>
  );
};

export default ImageAnimator;