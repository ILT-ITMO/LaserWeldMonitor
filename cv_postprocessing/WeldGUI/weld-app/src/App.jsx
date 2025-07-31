import { useCallback, useState, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";
import JSZip from "jszip";
import ImageWithOverlay from "./ImageWithOverlay";

export default function MyDropzone() {
  const [dataURL, setDataURL] = useState(null);
  const [uploadedURL, setUploadedURL] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [isEditing, setIsEditing] = useState(false);
  const [isProcessed, setIsProcessed] = useState(false);
  const [showLengthPopup, setShowLengthPopup] = useState(false);
  const [lengthValue, setLengthValue] = useState("");
  const [unitValue, setUnitValue] = useState("");
  const [referencePoints, setReferencePoints] = useState([
    { x: 0, y: 0 },
    { x: 0, y: 0 },
  ]);
  const [isDragging, setIsDragging] = useState(null);
  const [pixelToUnitRatio, setPixelToUnitRatio] = useState(1);
  const [allFiles, setAllFiles] = useState([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [showProcessChoice, setShowProcessChoice] = useState(false);
  const [processingMode, setProcessingMode] = useState(null);
  const [bulkResults, setBulkResults] = useState([]);
  const [currentBulkIndex, setCurrentBulkIndex] = useState(0);
  const [analysisResults, setAnalysisResults] = useState(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    if (acceptedFiles.length === 1) {
      const reader = new FileReader();
      reader.onload = () => {
        setDataURL(reader.result);
      };
      reader.readAsDataURL(acceptedFiles[0]);
    } else {
      setAllFiles(acceptedFiles);
      setShowProcessChoice(true);
    }
  }, []);

  const handleProcessChoice = (choice) => {
    setProcessingMode(choice);
    setShowProcessChoice(false);

    if (choice === "one-by-one") {
      setCurrentFileIndex(0);
      const reader = new FileReader();
      reader.onload = () => {
        setDataURL(reader.result);
      };
      reader.readAsDataURL(allFiles[0]);
    } else {
      setCurrentFileIndex(0);
      const reader = new FileReader();
      reader.onload = () => {
        setDataURL(reader.result);
      };
      reader.readAsDataURL(allFiles[0]);
    }
  };

  const goToNextFile = () => {
    const nextIndex = currentFileIndex + 1;
    if (nextIndex < allFiles.length) {
      setCurrentFileIndex(nextIndex);
      const reader = new FileReader();
      reader.onload = () => {
        setDataURL(reader.result);
      };
      reader.readAsDataURL(allFiles[nextIndex]);

      setUploadedURL(null);
      setIsEditing(false);
      setIsProcessed(false);
      setShowLengthPopup(false);
      setReferencePoints([
        { x: 0, y: 0 },
        { x: 0, y: 0 },
      ]);
    } else {
      setAllFiles([]);
      setCurrentFileIndex(0);
    }
  };

  useEffect(() => {
    if (imageRef.current && dataURL) {
      const img = new Image();
      img.onload = () => {
        setImageDimensions({
          width: img.width,
          height: img.height,
        });
      };
      img.src = dataURL;
    }
  }, [dataURL]);

  useEffect(() => {
    if (uploadedURL && canvasRef.current && imageDimensions.width > 0) {
      const scaleParams = uploadedURL[0].scale_params;
      const newPoints = [
        { x: scaleParams.reference_line[0][0], y: scaleParams.reference_line[0][1] },
        { x: scaleParams.reference_line[1][0], y: scaleParams.reference_line[1][1] },
      ];
      setReferencePoints(newPoints);
      setPixelToUnitRatio(scaleParams.le);

      const dx = newPoints[1].x - newPoints[0].x;
      const dy = newPoints[1].y - newPoints[0].y;
      const pixelLength = Math.sqrt(dx * dx + dy * dy);
      const realLength = pixelLength * scaleParams.le;

      setLengthValue(realLength.toFixed(2));
      setUnitValue(scaleParams.unit);

      drawReferenceLine();
    }
  }, [uploadedURL, imageDimensions]);

  useEffect(() => {
    if (canvasRef.current && imageDimensions.width > 0 && referencePoints[0].x !== 0) {
      drawReferenceLine();
    }
  }, [referencePoints, isEditing]);

  const drawReferenceLine = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const point1 = referencePoints[0];
    const point2 = referencePoints[1];

    ctx.beginPath();
    ctx.moveTo(point1.x, point1.y);
    ctx.lineTo(point2.x, point2.y);
    ctx.strokeStyle = isEditing ? "#FFA500" : "#00FFFF";
    ctx.lineWidth = 5;
    ctx.stroke();

    const dx = point2.x - point1.x;
    const dy = point2.y - point1.y;
    const pixelLength = Math.sqrt(dx * dx + dy * dy);
    const realLength = pixelLength * pixelToUnitRatio;

    const midX = (point1.x + point2.x) / 2;
    const midY = (point1.y + point2.y) / 2;
    const text = `${realLength.toFixed(2)} ${unitValue}`;
    ctx.font = "60px Arial";
    ctx.fillStyle = "#FFFFFF";
    ctx.textAlign = "center";
    ctx.fillText(text, midX, midY - 10);

    const angle = Math.atan2(dy, dx);
    const perpAngle = angle + Math.PI / 2;
    const lineLength = 10;

    drawEndMarker(ctx, point1.x, point1.y, perpAngle, lineLength);
    drawEndMarker(ctx, point2.x, point2.y, perpAngle, lineLength);
  };

  const drawEndMarker = (ctx, x, y, angle, length) => {
    ctx.beginPath();
    ctx.arc(x, y, length * 1.2, 0, Math.PI * 2);
    ctx.strokeStyle = isEditing ? "#FFA500" : "#00FFFF";
    ctx.lineWidth = 3;
    ctx.stroke();

    const xLength = length * 0.8;
    ctx.beginPath();
    ctx.moveTo(
      x - xLength * Math.cos(angle + Math.PI / 4),
      y - xLength * Math.sin(angle + Math.PI / 4)
    );
    ctx.lineTo(
      x + xLength * Math.cos(angle + Math.PI / 4),
      y + xLength * Math.sin(angle + Math.PI / 4)
    );
    ctx.strokeStyle = isEditing ? "#FFA500" : "#00FFFF";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(
      x - xLength * Math.cos(angle - Math.PI / 4),
      y - xLength * Math.sin(angle - Math.PI / 4)
    );
    ctx.lineTo(
      x + xLength * Math.cos(angle - Math.PI / 4),
      y + xLength * Math.sin(angle - Math.PI / 4)
    );
    ctx.strokeStyle = isEditing ? "#FFA500" : "#00FFFF";
    ctx.lineWidth = 3;
    ctx.stroke();
  };

  const handleCanvasMouseDown = (e) => {
    if (!isEditing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const threshold = 20;
    for (let i = 0; i < referencePoints.length; i++) {
      const point = referencePoints[i];
      const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
      if (distance < threshold) {
        setIsDragging(i);
        return;
      }
    }
  };

  const handleCanvasMouseMove = (e) => {
    if (!isEditing || isDragging === null) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const newPoints = [...referencePoints];
    newPoints[isDragging] = { x, y };
    setReferencePoints(newPoints);
  };

  const handleCanvasMouseUp = () => {
    setIsDragging(null);
  };

  const handleAcceptLine = () => {
    const point1 = referencePoints[0];
    const point2 = referencePoints[1];
    const dx = point2.x - point1.x;
    const dy = point2.y - point1.y;
    const pixelLength = Math.sqrt(dx * dx + dy * dy);
    const realLength = pixelLength * pixelToUnitRatio;

    setLengthValue(realLength.toFixed(2));
    setShowLengthPopup(true);
  };

  const extractImagesFromZip = useCallback(async (zipBlob) => {
    try {
      const zip = await JSZip.loadAsync(zipBlob);
      const imageUrls = [];
      const filePromises = [];

      zip.forEach((relativePath, file) => {
        if (!file.dir) {
          const lowerPath = relativePath.toLowerCase();
          if (
            lowerPath.endsWith(".jpg") ||
            lowerPath.endsWith(".jpeg") ||
            lowerPath.endsWith(".png")
          ) {
            filePromises.push(
              file.async("blob").then((blob) => {
                const url = URL.createObjectURL(blob);
                imageUrls.push({
                  url,
                  name: relativePath.split("/").pop(),
                });
              })
            );
          }
        }
      });

      await Promise.all(filePromises);
      return imageUrls;
    } catch (error) {
      console.error("Error extracting ZIP:", error);
      throw error;
    }
  }, []);

  const handleSaveLength = async () => {
    try {
      const point1 = referencePoints[0];
      const point2 = referencePoints[1];
      const dx = point2.x - point1.x;
      const dy = point2.y - point1.y;
      const pixelLength = Math.sqrt(dx * dx + dy * dy);
      const newLe = parseFloat(lengthValue) / pixelLength;

      const requestData = {
        filename: uploadedURL[0].filename,
        scale_params: {
          le: newLe,
          unit: unitValue,
          reference_line: [
            [referencePoints[0].x, referencePoints[0].y],
            [referencePoints[1].x, referencePoints[1].y],
          ],
        },
      };

      if (processingMode === "bulk") {
        const formData = new FormData();
        allFiles.forEach((file) => formData.append("images", file));

        formData.append(
          "scale_params",
          JSON.stringify({
            le: newLe,
            unit: unitValue,
            reference_line: [
              [referencePoints[0].x, referencePoints[0].y],
              [referencePoints[1].x, referencePoints[1].y],
            ],
          })
        );

        const response = await fetch("http://localhost:5000/api/process_bulk_images", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const responseContentType = response.headers.get("content-type");
          console.log("Response content type:", responseContentType);

          if (responseContentType.includes("application/zip")) {
            const zipBlob = await response.blob();
            const urls = await extractImagesFromZip(zipBlob);
            setBulkResults(urls);
          } else {
            const data = await response.json();
            setBulkResults(data.processedImages);
            setAnalysisResults(data); // Store the analysis results
          }
          setShowLengthPopup(false);
          setIsEditing(false);
          setIsProcessed(true);
          setCurrentBulkIndex(0);
        }

        setUploadedURL((prev) => {
          const newData = [...prev];
          newData[0] = {
            ...newData[0],
            scale_params: {
              le: newLe,
              unit: unitValue,
              reference_line: [
                [referencePoints[0].x, referencePoints[0].y],
                [referencePoints[1].x, referencePoints[1].y],
              ],
            },
          };
          return newData;
        });
      } else {
        const response = await fetch("http://localhost:5000/api/process_image", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestData),
        });

        if (!response.ok) {
          throw new Error("Failed to update scale parameters");
        }

        const contentType = response.headers.get("content-type");
        console.log("Content-Type:", contentType);

        if (contentType && contentType.includes("application/json")) {
          const data = await response.json();
          setAnalysisResults(data.analysis_results); // Store the analysis results
          setShowLengthPopup(false);
          setIsEditing(false);
          setIsProcessed(true);
        } else if (contentType && contentType.includes("image")) {
          const blob = await response.blob();
          const imageUrl = URL.createObjectURL(blob);
//           setDataURL(imageUrl);
          setShowLengthPopup(false);
          setIsEditing(false);
          setIsProcessed(true);
        } else {
          const text = await response.text();
          console.warn("Unexpected response:", text);
          throw new Error("Server did not return expected response");
        }

        if (processingMode === "one-by-one") {
          setTimeout(goToNextFile, 1000);
        }
      }
    } catch (error) {
      console.error("Error updating scale parameters:", error);
    }
  };

  const uploadImage = async () => {
    const urls = [];
    for (let file of acceptedFiles) {
      const formData = new FormData();
      const fileToUpload =
        allFiles.length > 0 ? allFiles[currentFileIndex] : acceptedFiles[0];
      formData.append("image", fileToUpload);
      const res = await fetch(`http://localhost:5000/api/get_scale_params`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      urls.push(data);
    }
    setUploadedURL(urls);
  };

  const { getRootProps, acceptedFiles, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true,
  });

  return (
    <div className="container">
      <div className="zone">
        {allFiles.length > 1 && processingMode === "one-by-one" && (
          <div className="file-counter">
            File {currentFileIndex + 1} of {allFiles.length}
          </div>
        )}

        {dataURL ? (
          <div className="selected">
            {isProcessed && analysisResults ? (
              <ImageWithOverlay
                imageUrl={dataURL}
                linesData={analysisResults.images[0].linesData}
              />
            ) : (
              <div style={{ position: "relative" }}>
                <img
                  ref={imageRef}
                  src={dataURL}
                  style={{ maxWidth: "100%", height: "auto" }}
                  alt="Uploaded"
                />
                {uploadedURL && imageDimensions.width > 0 && (
                  <canvas
                    ref={canvasRef}
                    width={imageDimensions.width}
                    height={imageDimensions.height}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      cursor: isEditing ? "pointer" : "default",
                    }}
                    onMouseDown={handleCanvasMouseDown}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseUp={handleCanvasMouseUp}
                    onMouseLeave={handleCanvasMouseUp}
                  />
                )}
              </div>
            )}

            {!isProcessed && (
              <div className="actions">
                {uploadedURL ? (
                  <>
                    {!isEditing ? (
                      <>
                        <button onClick={() => setIsEditing(true)} className="edit-btn">
                          Edit Line
                        </button>
                        <button onClick={handleAcceptLine} className="accept-btn">
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
                        <button onClick={handleAcceptLine} className="accept-btn">
                          Save
                        </button>
                      </>
                    )}
                  </>
                ) : (
                  <button onClick={uploadImage} className="upload-btn">
                    Upload
                  </button>
                )}
                <button
                  onClick={() => {
                    setDataURL(null);
                    setUploadedURL(null);
                    setIsEditing(false);
                    setIsProcessed(false);
                    setAnalysisResults(null);
                  }}
                  className="cancel-btn"
                >
                  Cancel
                </button>
              </div>
            )}
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
              <div className="drag-files">Drop your files here or click to browse</div>
            )}
          </div>
        )}
      </div>

      {showProcessChoice && (
        <div className="popup-overlay">
          <div className="process-choice-popup">
            <h3>Multiple Files Detected</h3>
            <p>How would you like to process these {allFiles.length} files?</p>
            <div className="choice-buttons">
              <button onClick={() => handleProcessChoice("one-by-one")}>
                Process One by One
              </button>
              <button onClick={() => handleProcessChoice("bulk")}>
                Process in Bulk
              </button>
            </div>
          </div>
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
              <button onClick={handleSaveLength} className="primary">
                OK
              </button>
            </div>
          </div>
        </div>
      )}

      {processingMode === "bulk" && bulkResults.length > 0 && (
        <div className="bulk-navigation">
          <button
            onClick={() => setCurrentBulkIndex((prev) => Math.max(0, prev - 1))}
            disabled={currentBulkIndex === 0}
          >
            Previous
          </button>
          <span>
            Image {currentBulkIndex + 1} of {bulkResults.length}
          </span>
          <button
            onClick={() =>
              setCurrentBulkIndex((prev) =>
                Math.min(bulkResults.length - 1, prev + 1)
              )
            }
            disabled={currentBulkIndex === bulkResults.length - 1}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}