import React, { useEffect, useRef } from 'react';

const ImageWithOverlay = ({ imageUrl, linesData, scaleParams }) => {
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
      if (linesData.misalignment) {
        drawText(
          ctx,
          `Misalignment: ${linesData.misalignment[0].toFixed(2)} ${scaleParams.u}`,
          20,  // x position
          40,  // y position
          'red',
          50
        );
      }

      // Draw the lines
      drawLines(ctx, linesData);
    };
  }, [imageUrl, linesData]);
const calculateLength = (point1, point2) => {
  const dx = point2[0] - point1[0];
  const dy = point2[1] - point1[1];
  const pixelLength = Math.sqrt(dx * dx + dy * dy);
// console.log(imageUrl);
  return (pixelLength * scaleParams.le).toFixed(2); // Convert to real units
};

const drawText = (ctx, text, x, y, color = 'black', fontSize = 20) => {
  ctx.font = `${fontSize}px Arial`;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
};
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

    const midX = (point1[0] + point2[0]) / 2;
    const midY = (point1[1] + point2[1]) / 2;

      // Draw length text
    const length = calculateLength(point1, point2);
    drawText(ctx, `${length} ${scaleParams.u}`, midX, midY, 'white', 40);
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