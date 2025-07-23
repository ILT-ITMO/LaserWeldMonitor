import { useCallback, useState, useRef, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import "./App.css"

export default function MyDropzone() {
  const [dataURL, setDataURL] = useState(null)
  const [uploadedURL, setUploadedURL] = useState(null)
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })
  const [isEditing, setIsEditing] = useState(false)
  const [showLengthPopup, setShowLengthPopup] = useState(false)
  const [lengthValue, setLengthValue] = useState("")
  const [unitValue, setUnitValue] = useState("")
  const [referencePoints, setReferencePoints] = useState([{x: 0, y: 0}, {x: 0, y: 0}])
  const [isDragging, setIsDragging] = useState(null)
  const [pixelToUnitRatio, setPixelToUnitRatio] = useState(1)

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
      // Initialize reference points from the server response
      const scaleParams = uploadedURL[0].scale_params
      const newPoints = [
        { x: scaleParams.reference_line[0][0], y: scaleParams.reference_line[0][1] },
        { x: scaleParams.reference_line[1][0], y: scaleParams.reference_line[1][1] }
      ]
      setReferencePoints(newPoints)
      setPixelToUnitRatio(scaleParams.le)

      // Calculate initial length
      const dx = newPoints[1].x - newPoints[0].x
      const dy = newPoints[1].y - newPoints[0].y
      const pixelLength = Math.sqrt(dx * dx + dy * dy)
      const realLength = pixelLength * scaleParams.le

      setLengthValue(realLength.toFixed(2))
      setUnitValue(scaleParams.unit)

      drawReferenceLine()
    }
  }, [uploadedURL, imageDimensions])

  useEffect(() => {
    if (canvasRef.current && imageDimensions.width > 0 && referencePoints[0].x !== 0) {
      drawReferenceLine()
    }
  }, [referencePoints, isEditing])

  const drawReferenceLine = () => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const point1 = referencePoints[0]
    const point2 = referencePoints[1]

    // Draw the reference line
    ctx.beginPath()
    ctx.moveTo(point1.x, point1.y)
    ctx.lineTo(point2.x, point2.y)
    ctx.strokeStyle = isEditing ? '#FFA500' : '#00FFFF' // Orange when editing, cyan otherwise
    ctx.lineWidth = 5
    ctx.stroke()

    // Calculate the length in real units
    const dx = point2.x - point1.x
    const dy = point2.y - point1.y
    const pixelLength = Math.sqrt(dx * dx + dy * dy)
    const realLength = pixelLength * pixelToUnitRatio

    // Draw the scale text
    const midX = (point1.x + point2.x) / 2
    const midY = (point1.y + point2.y) / 2
    const text = `${realLength.toFixed(2)} ${unitValue}`
    const textWidth = ctx.measureText(text).width
    const textY = midY - 10
    ctx.font = '60px Arial'
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
    ctx.fillStyle = '#FFFFFF'
    ctx.textAlign = 'center'
    ctx.fillText(text, midX, textY)

    // Draw end markers
    const angle = Math.atan2(dy, dx)
    const perpAngle = angle + Math.PI / 2
    const lineLength = 10

    drawEndMarker(ctx, point1.x, point1.y, perpAngle, lineLength)
    drawEndMarker(ctx, point2.x, point2.y, perpAngle, lineLength)
  }

  const drawEndMarker = (ctx, x, y, angle, length) => {
    ctx.beginPath()
    ctx.arc(x, y, length * 1.2, 0, Math.PI * 2)
    ctx.strokeStyle = isEditing ? '#FFA500' : '#00FFFF'
    ctx.lineWidth = 3
    ctx.stroke()

    const xLength = length * 0.8
    ctx.beginPath()
    ctx.moveTo(x - xLength * Math.cos(angle + Math.PI / 4), y - xLength * Math.sin(angle + Math.PI / 4))
    ctx.lineTo(x + xLength * Math.cos(angle + Math.PI / 4), y + xLength * Math.sin(angle + Math.PI / 4))
    ctx.strokeStyle = isEditing ? '#FFA500' : '#00FFFF'
    ctx.lineWidth = 3
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(x - xLength * Math.cos(angle - Math.PI / 4), y - xLength * Math.sin(angle - Math.PI / 4))
    ctx.lineTo(x + xLength * Math.cos(angle - Math.PI / 4), y + xLength * Math.sin(angle - Math.PI / 4))
    ctx.strokeStyle = isEditing ? '#FFA500' : '#00FFFF'
    ctx.lineWidth = 3
    ctx.stroke()
  }

  const handleCanvasMouseDown = (e) => {
    if (!isEditing) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    // Check if mouse is near any of the reference points
    const threshold = 20
    for (let i = 0; i < referencePoints.length; i++) {
      const point = referencePoints[i]
      const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2))
      if (distance < threshold) {
        setIsDragging(i)
        return
      }
    }
  }

  const handleCanvasMouseMove = (e) => {
    if (!isEditing || isDragging === null) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    const newPoints = [...referencePoints]
    newPoints[isDragging] = { x, y }
    setReferencePoints(newPoints)
  }

  const handleCanvasMouseUp = () => {
    setIsDragging(null)
  }

  const handleAcceptLine = () => {
    // Calculate the current length to show in the popup
    const point1 = referencePoints[0]
    const point2 = referencePoints[1]
    const dx = point2.x - point1.x
    const dy = point2.y - point1.y
    const pixelLength = Math.sqrt(dx * dx + dy * dy)
    const realLength = pixelLength * pixelToUnitRatio

    setLengthValue(realLength.toFixed(2))
    setShowLengthPopup(true)
  }

 const handleSaveLength = async () => {
   try {
     // Calculate the new pixel-to-unit ratio (le)
     const point1 = referencePoints[0];
     const point2 = referencePoints[1];
     const dx = point2.x - point1.x;
     const dy = point2.y - point1.y;
     const pixelLength = Math.sqrt(dx * dx + dy * dy);
     const newLe = parseFloat(lengthValue) / pixelLength;

     // Prepare the data to send to the backend
     const requestData = {
       filename: uploadedURL[0].filename, // Assuming the filename is in the response
       scale_params: {
         le: newLe,
         unit: unitValue,
         reference_line: [
           [referencePoints[0].x, referencePoints[0].y],
           [referencePoints[1].x, referencePoints[1].y]
         ]
       }
     };

     // Send the request to your backend
     const response = await fetch('http://localhost:5000/api/process_image', {
       method: 'POST',
       headers: {
         'Content-Type': 'application/json',
       },
       body: JSON.stringify(requestData)
     });

     if (!response.ok) {
       throw new Error('Failed to update scale parameters');
     }

     const result = await response.json();
     console.log('Scale parameters updated:', result);

     // Close the popup and exit edit mode
     setShowLengthPopup(false);
     setIsEditing(false);

     // Optionally update the local state with the new parameters
     setUploadedURL(prev => {
       const newData = [...prev];
       newData[0] = {
         ...newData[0],
         scale_params: {
           le: newLe,
           unit: unitValue,
           reference_line: [
             [referencePoints[0].x, referencePoints[0].y],
             [referencePoints[1].x, referencePoints[1].y]
           ]
         }
       };
       return newData;
     });

   } catch (error) {
     console.error('Error updating scale parameters:', error);
     // You might want to show an error message to the user here
   }
 };

  const { getRootProps, acceptedFiles, getInputProps, isDragActive } = useDropzone({ onDrop, multiple: true })

  const uploadImage = async () => {
    const urls = []
    for (let file of acceptedFiles) {
      const formData = new FormData()
      formData.append("image", file)

      const res = await fetch(`http://localhost:5000/api/get_scale_params`, {
        method: "POST",
        body: formData,
      })
      const data = await res.json()
      urls.push(data)
    }
    setUploadedURL(urls)
  }

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
                    height: '100%',
                    cursor: isEditing ? 'pointer' : 'default'
                  }}
                  onMouseDown={handleCanvasMouseDown}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseUp={handleCanvasMouseUp}
                  onMouseLeave={handleCanvasMouseUp}
                />
              )}
            </div>
            <div className="actions">
              {uploadedURL ? (
                <>
                  {!isEditing ? (
                    <>
                      <button
                        onClick={() => setIsEditing(true)}
                        className="edit-btn"
                      >
                        Edit Line
                      </button>
                      <button
                        onClick={handleAcceptLine}
                        className="accept-btn"
                      >
                        Accept
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => setIsEditing(false)}
                        className="cancel-btn"
                      >
                        Cancel Edit
                      </button>
                      <button
                        onClick={handleAcceptLine}
                        className="accept-btn"
                      >
                        Save
                      </button>
                    </>
                  )}
                </>
              ) : (
                <button
                  onClick={uploadImage}
                  className="upload-btn"
                >
                  Upload
                </button>
              )}
              <button
                onClick={() => {
                  setDataURL(null)
                  setUploadedURL(null)
                  setIsEditing(false)
                }}
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

      {showLengthPopup && (
        <div className="popup-overlay">
          <div className="length-popup">
            <h3>Confirm Reference Length</h3>
            <div className="input-group">
              <input
                type="number"
                value={lengthValue}
                onChange={(e) => setLengthValue(e.target.value)}
              />
              <select
                value={unitValue}
                onChange={(e) => setUnitValue(e.target.value)}
              >
                <option value="mm">mm</option>
                <option value="cm">cm</option>
                <option value="m">m</option>
                <option value="in">in</option>
              </select>
            </div>
            <div className="popup-buttons">
              <button onClick={() => setShowLengthPopup(false)}>Cancel</button>
              <button onClick={handleSaveLength} className="primary">OK</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}