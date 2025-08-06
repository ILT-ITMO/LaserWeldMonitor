import React, { useEffect, useRef } from 'react';

const BaseImageCanvas = ({ imageUrl }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.src = imageUrl;

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height); // clear any previous
      ctx.drawImage(img, 0, 0);
    };

    img.onerror = () => {
      console.error('Failed to load base image:', imageUrl);
    };
  }, [imageUrl]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'absolute', top: 0, left: 0, zIndex: 0, maxWidth: '100%', height: 'auto' }}
    />
  );
};

export default BaseImageCanvas;

