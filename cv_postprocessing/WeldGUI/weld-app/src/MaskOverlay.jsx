import { useEffect, useRef } from 'react';

export default function MaskOverlay({ canvasRef, imageDimensions, maskContours }) {
  const overlayRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !maskContours || maskContours.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!visible) return;
    // Draw each contour
    maskContours.forEach(contour => {
      if (contour.length < 2) return;

      ctx.beginPath();
      ctx.moveTo(contour[0][0], contour[0][1]);

      for (let i = 1; i < contour.length; i++) {
        ctx.lineTo(contour[i][0], contour[i][1]);
      }

      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 255, 0, 0.3)'; // Semi-transparent green
      ctx.fill();
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }, [maskContours, imageDimensions, visible]);

  return null; // This component doesn't render anything visible
}