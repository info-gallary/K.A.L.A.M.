// components/Resizer.jsx
import React from 'react';

const Resizer = ({ 
  direction = 'vertical', // 'vertical' or 'horizontal'
  onMouseDown,
  themeClasses,
  isDragging = false
}) => {
  const isVertical = direction === 'vertical';
  
  return (
    <div
      className={`
        ${isVertical ? 'w-1.5 cursor-col-resize' : 'h-1.5 cursor-row-resize'}
        ${themeClasses.border} 
        ${isVertical ? 'border-r' : 'border-b'}
        group
        relative
        transition-colors
        ${isDragging ? themeClasses.accent : ''}
      `}
      onMouseDown={onMouseDown}
      style={{
        backgroundColor: isDragging ? themeClasses.accent : undefined
      }}
    >
      {/* Hover area - wider than visual line */}
      <div 
        className={`
          absolute inset-0
          ${isVertical ? 'w-3 -ml-1' : 'h-3 -mt-1'}
          ${themeClasses.hover}
          group-hover:bg-opacity-50
          transition-colors
        `}
      />
      
      {/* Visual indicator on hover */}
      <div 
        className={`
          absolute 
          ${isVertical 
            ? 'left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 w-0.5 h-8' 
            : 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-0.5 w-8'
          }
          opacity-0 group-hover:opacity-100
          transition-opacity
          rounded-full
        `}
        style={{ backgroundColor: themeClasses.accent }}
      />
      
      {/* Drag dots indicator */}
      <div 
        className={`
          absolute 
          ${isVertical 
            ? 'left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 flex flex-col space-y-0.5' 
            : 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex space-x-0.5'
          }
          opacity-0 group-hover:opacity-60
          transition-opacity
        `}
      >
        {[...Array(3)].map((_, i) => (
          <div
            key={i}
            className="w-0.5 h-0.5 rounded-full"
            style={{ backgroundColor: themeClasses.accent }}
          />
        ))}
      </div>
    </div>
  );
};

export default Resizer;