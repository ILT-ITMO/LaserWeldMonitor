import React from 'react';

export default function ToggleControls({
  showMask,
  setShowMask,
  showMeasurements,
  setShowMeasurements,
  showScaleLine,
  setShowScaleLine
}) {
  return (
    <div className="toggle-controls">
      <button
        onClick={() => setShowMask(!showMask)}
        className={showMask ? 'active' : ''}
      >
        {showMask ? 'Hide Mask' : 'Show Mask'}
      </button>
      <button
        onClick={() => setShowMeasurements(!showMeasurements)}
        className={showMeasurements ? 'active' : ''}
      >
        {showMeasurements ? 'Hide Measurements' : 'Show Measurements'}
      </button>
      <button
        onClick={() => setShowScaleLine(!showScaleLine)}
        className={showScaleLine ? 'active' : ''}
      >
        {showScaleLine ? 'Hide Scale Line' : 'Show Scale Line'}
      </button>
    </div>
  );
}