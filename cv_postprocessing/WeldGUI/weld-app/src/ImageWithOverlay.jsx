import React, { useEffect, useRef } from 'react';

const ImageWithOverlay = ({ imageUrl, linesData }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl || !linesData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Load the image
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      // Set canvas dimensions to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the image
      ctx.drawImage(img, 0, 0);

      // Draw the lines
      drawLines(ctx, linesData);
    };
  }, [imageUrl, linesData]);

  const drawLines = (ctx, linesData) => {
    // Configure line style
    ctx.lineWidth = 7;

    // Draw contour lines (blue)
    if (linesData.contour_lines) {
      ctx.strokeStyle = 'blue';
      linesData.contour_lines.forEach(polyline => {
        drawPolyline(ctx, polyline);
      });
    }

    // Draw plate lines (red)
    if (linesData.plate_lines) {
      ctx.strokeStyle = 'red';
      linesData.plate_lines.forEach(line => {
        drawLine(ctx, line[0], line[1]);
      });
    }

    // Draw deviation lines (green)
    if (linesData.deviation_lines) {
      ctx.strokeStyle = 'green';
      linesData.deviation_lines.forEach(line => {
        drawLine(ctx, line[0], line[1]);
      });
    }

    // Draw main sides (yellow)
    if (linesData.main_sides) {
      ctx.strokeStyle = 'yellow';
      linesData.main_sides.forEach(side => {
        drawLine(ctx, side[0], side[1]);
      });
    }
  };

  const drawLine = (ctx, point1, point2) => {
    ctx.beginPath();
    ctx.moveTo(point1[0], point1[1]);
    ctx.lineTo(point2[0], point2[1]);
    ctx.stroke();
  };

  const drawPolyline = (ctx, points) => {
    if (points.length === 0) return;

    ctx.beginPath();
    ctx.moveTo(points[0][0][0], points[0][0][1]);

    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0][0], points[i][0][1]);
    }

    ctx.stroke();
  };

  return (
    <div className="image-container">
      <canvas
        ref={canvasRef}
        style={{ maxWidth: '100%', height: 'auto' }}
      />
    </div>
  );
};

export default ImageWithOverlay;