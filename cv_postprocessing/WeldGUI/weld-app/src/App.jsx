import { useCallback, useState, useRef, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import "./App.css"

// export default function FilePicker() {
//   return <div className="container">
//     <input type='file' className='picker' />
//   </div>
// }

export default function MyDropzone() {
  const [dataURL, setDataURL] = useState(null)
  const [uploadedURL, setUploadedURL] = useState(null)
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const onDrop = useCallback(acceptedFiles => {
    acceptedFiles.forEach(file => {
      const reader = new FileReader()
      reader.onabort = () => console.log("file reading was aborted")
      reader.onerror = () => console.log("file reading has failed")
      reader.onload = () => {
        const binaryStr = reader.result
        setDataURL(binaryStr)
      }
      reader.readAsDataURL(file)
    })
  }, [])
useEffect(() => {
    if (imageRef.current && dataURL) {
      const img = new Image()
      img.onload = () => {
        setImageDimensions({
          width: img.width,
          height: img.height
        })
      }
      img.src = dataURL
    }
  }, [dataURL])

  useEffect(() => {
    if (uploadedURL && canvasRef.current && imageDimensions.width > 0) {
      drawReferenceLine()
    }
  }, [uploadedURL, imageDimensions])

  const drawReferenceLine = () => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scaleParams = uploadedURL[0].scale_params
    const [point1, point2] = scaleParams.reference_line

    // Draw the reference line
    ctx.beginPath()
    ctx.moveTo(point1[0], point1[1])
    ctx.lineTo(point2[0], point2[1])
    ctx.strokeStyle = '#00FFFF'
    ctx.lineWidth = 5
    ctx.stroke()

    // Calculate the length in real units
    const dx = point2[0] - point1[0]
    const dy = point2[1] - point1[1]
    const pixelLength = Math.sqrt(dx * dx + dy * dy)
    const realLength = pixelLength * scaleParams.le

    // Draw the scale text
    const midX = (point1[0] + point2[0]) / 2
    const midY = (point1[1] + point2[1]) / 2
    const text = `${realLength.toFixed(2)} ${scaleParams.unit}`;
    const textWidth = ctx.measureText(text).width;
    const textY = midY - 10
    ctx.font = '60px Arial'
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';

//     ctx.fillRect(midX - textWidth / 2 - 5, textY - 60, textWidth + 10, 30);

      // White text for readability
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';


    ctx.fillText(text, midX, textY)

    // Draw small perpendicular lines at the ends
    const lineLength = 10
    const angle = Math.atan2(dy, dx)

    // Perpendicular angle
    const perpAngle = angle + Math.PI / 2

    // Draw end markers
    drawEndMarker(ctx, point1[0], point1[1], perpAngle, lineLength)
    drawEndMarker(ctx, point2[0], point2[1], perpAngle, lineLength)
  }

 const drawEndMarker = (ctx, x, y, angle, length) => {
     // Draw the surrounding circle
     ctx.beginPath();
     ctx.arc(x, y, length * 1.2, 0, Math.PI * 2);
     ctx.strokeStyle = '#00FFFF';
     ctx.lineWidth = 3;
     ctx.stroke();

     // Draw an "X" inside the circle (two crossed lines at 45° and -45°)
     const xLength = length * 0.8; // Slightly smaller than the circle radius

     // First diagonal line (top-left to bottom-right)
     ctx.beginPath();
     ctx.moveTo(x - xLength * Math.cos(angle + Math.PI / 4), y - xLength * Math.sin(angle + Math.PI / 4));
     ctx.lineTo(x + xLength * Math.cos(angle + Math.PI / 4), y + xLength * Math.sin(angle + Math.PI / 4));
     ctx.strokeStyle = '#00FFFF';
     ctx.lineWidth = 3; // Slightly thinner than the circle
     ctx.stroke();

     // Second diagonal line (top-right to bottom-left)
     ctx.beginPath();
     ctx.moveTo(x - xLength * Math.cos(angle - Math.PI / 4), y - xLength * Math.sin(angle - Math.PI / 4));
     ctx.lineTo(x + xLength * Math.cos(angle - Math.PI / 4), y + xLength * Math.sin(angle - Math.PI / 4));
     ctx.strokeStyle = '#00FFFF';
     ctx.lineWidth = 3;
     ctx.stroke();
 };

  const {
    getRootProps,
    acceptedFiles,
    getInputProps,
    isDragActive,

  } = useDropzone({ onDrop,  multiple:true })

//   const selectedFile = acceptedFiles[0]
const uploadImage = async () => {
  const urls = [];
  for (let file of acceptedFiles) {
    const formData = new FormData();
    formData.append("image", file);

    // Send to `/api/get_scale_params` instead of `/api/predict`
    const res = await fetch(`http://localhost:5000/api/get_scale_params`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    urls.push(data); // Store the entire response (not just URL)

    // Log the scale parameters to console
    console.log("Scale Params:", data.scale_params);
  }
  setUploadedURL(urls); // Now stores the full response (including scale_params)
};

  return (
    <div className="container">
      <div className="zone">
      {dataURL ? (
                <div className="selected">
                  <div style={{ position: 'relative' }}>
                    <img
                      ref={imageRef}
                      src={dataURL}
                      style={{ maxWidth: '100%', height: 'auto' }}
                      alt="Uploaded"
                    />
                    {uploadedURL && imageDimensions.width > 0 && (
                      <canvas
                        ref={canvasRef}
                        width={imageDimensions.width}
                        height={imageDimensions.height}
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%'
                        }}
                      />
                    )}
                  </div>
            <div className="actions">
              {uploadedURL ? (
                <span className="uploaded-txt">Uploaded!</span>
              ) : (
                <button
                  onClick={uploadImage}
                  className="upload-btn"
                >
                  Upload
                </button>
              )}
              <button
                onClick={() => setDataURL(null)}
                className="cancel-btn"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="drop-zone" {...getRootProps()}>
            <input {...getInputProps()} />
            {isDragActive ? (
              <div className="drop-files">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  height="50"
                  width="50"
                  fill="currentColor"
                >
                  <path d="M1 14.5C1 12.1716 2.22429 10.1291 4.06426 8.9812C4.56469 5.044 7.92686 2 12 2C16.0731 2 19.4353 5.044 19.9357 8.9812C21.7757 10.1291 23 12.1716 23 14.5C23 17.9216 20.3562 20.7257 17 20.9811L7 21C3.64378 20.7257 1 17.9216 1 14.5ZM16.8483 18.9868C19.1817 18.8093 21 16.8561 21 14.5C21 12.927 20.1884 11.4962 18.8771 10.6781L18.0714 10.1754L17.9517 9.23338C17.5735 6.25803 15.0288 4 12 4C8.97116 4 6.42647 6.25803 6.0483 9.23338L5.92856 10.1754L5.12288 10.6781C3.81156 11.4962 3 12.927 3 14.5C3 16.8561 4.81833 18.8093 7.1517 18.9868L7.325 19H16.675L16.8483 18.9868ZM13 13V17H11V13H8L12 8L16 13H13Z"></path>
                </svg>
              </div>
            ) : (
              <div className="drag-files">
                Drop your files here or click to browse
              </div>
            )}
          </div>
        )}
      </div>
      {uploadedURL && (
             <div className="scale-params-display">
               <h3>Scale Parameters:</h3>
               <pre>{JSON.stringify(uploadedURL[0], null, 2)}</pre>
             </div>
           )}
    </div>
  )
}