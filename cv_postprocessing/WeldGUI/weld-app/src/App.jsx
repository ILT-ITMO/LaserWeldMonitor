import { useCallback, useState, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import "./App.css";
import JSZip from "jszip";
import ImageWithOverlay from "./ImageWithOverlay";
import ToggleControls from "./ToggleControls"
export default function MyDropzone() {
  const [dataURL, setDataURL] = useState(null);
  const [uploadedURL, setUploadedURL] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [isEditing, setIsEditing] = useState(false);
  const [isProcessed, setIsProcessed] = useState(false);
  const [showLengthPopup, setShowLengthPopup] = useState(false);
  const [lengthValue, setLengthValue] = useState("");
  const [unitValue, setUnitValue] = useState("");
  const [blobUrl, setBlobUrl] = useState(null);
  const [uploadedPaths, setUploadedPaths] = useState([]);
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
  const [csvData, setCsvData] = useState(null);
  const [showMask, setShowMask] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showScaleLine, setShowScaleLine] = useState(true);
  const maskCanvasRef = useRef(null);

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
    const img = new Image();
    img.src = dataURL;

  }, [dataURL]);

useEffect(() => {
  if (showMask && maskCanvasRef.current && analysisResults) {
    const canvas = maskCanvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw mask contours if they exist
    if (analysisResults.images[0]?.linesData?.mask_contours?.length > 0) {
      ctx.fillStyle = "rgba(0, 255, 0, 0.3)"; // Semi-transparent green

      analysisResults.images[0].linesData.mask_contours.forEach(contour => {
        if (contour.length > 0) {
          ctx.beginPath();
          // Move to first point
          ctx.moveTo(contour[0][0], contour[0][1]);

          // Draw lines to subsequent points
          for (let i = 1; i < contour.length; i++) {
            ctx.lineTo(contour[i][0], contour[i][1]);
          }

          // Close the path and fill
          ctx.closePath();
          ctx.fill();
        }
      });
    }
  }
}, [showMask, analysisResults]);
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
        console.log(uploadedURL);
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
  }, [referencePoints, isEditing, showScaleLine, isProcessed, imageDimensions]);
// useEffect(() => {
//   if (canvasRef.current && imageDimensions.width > 0 && referencePoints[0].x !== 0) {
//     drawReferenceLine();
//   }
// }, [showScaleLine, referencePoints, isEditing, isProcessed]);
  const drawReferenceLine = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const point1 = referencePoints[0];
    const point2 = referencePoints[1];
//     if (isProcessed && !showScaleLine) return;
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
     const baseURL = "https://aveen007-laserweld.hf.space";
     const apiEndpoint = "/call/predict_4";

     try {
      const pathObjects = uploadedPaths.map(path => ({
               path: path
             }));

             console.log("Transformed paths:", pathObjects);

             // Step 2: Call predict_4 with all paths and scale params
             const callBody = {
               "data": [
                 pathObjects, // Now this is properly formatted as [{path: "...}, {path: "..."}]
                 {
                   "le": newLe,
                   "unit": unitValue,
                   "reference_line": [
                     [referencePoints[0].x, referencePoints[0].y],
                     [referencePoints[1].x, referencePoints[1].y],
                   ]
                 }
               ]
             };

             console.log("Sending bulk processing request:", JSON.stringify(callBody, null, 2))

       const callResponse = await fetch(`${baseURL}${apiEndpoint}`, {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(callBody),
       });

       if (!callResponse.ok) {
         throw new Error(`Bulk processing failed with status ${callResponse.status}`);
       }

       const callData = await callResponse.json();
       const eventId = callData.event_id;
       console.log("Bulk processing started, Event ID:", eventId);

       // Step 3: Poll for results
       const pollInterval = 1000;
       let resultData = null;

       await new Promise(resolve => setTimeout(resolve, pollInterval));

       const pollUrl = `${baseURL}${apiEndpoint}/${eventId}`;
       const pollResponse = await fetch(pollUrl);

       if (!pollResponse.ok) {
         throw new Error(`Polling failed with status ${pollResponse.status}`);
       }

       let pollData;
       const responseText = await pollResponse.text();

       try {
         const dataMatch = responseText.match(/data: (.*)/s);
         if (dataMatch && dataMatch[1]) {
           const jsonString = dataMatch[1].trim();
           pollData = JSON.parse(jsonString);
         } else {
           if (responseText.includes('status')) {
             pollData = JSON.parse(responseText);
           } else {
             pollData = { data: null, status: { status: 'waiting', stage: 'unknown' } };
           }
         }
       } catch (jsonError) {
         console.error("Failed to parse JSON response during polling:", jsonError, "Raw text:", responseText);
         throw new Error("Invalid or unexpected server response.");
       }

       console.log("Received bulk poll data:", pollData);

       // Extract the analysis results from the response
       if (pollData && pollData.length > 0) {
         resultData = pollData[0]; // Get the first result
       } else if (pollData.data) {
         resultData = pollData.data[0];
       } else {
         throw new Error("No valid result data received");
       }

       // Process the results
       if (resultData && resultData.analysis_results) {
         console.log("Bulk processing complete:", resultData);

         // Create results with original URLs for display
         const resultsWithUrls = resultData.analysis_results.images.map((result, index) => ({
           ...result,
           originalUrl: URL.createObjectURL(allFiles[index])
         }));

         setBulkResults(resultsWithUrls);
         setCurrentBulkIndex(0);

         // Set analysis results with the transformed structure
         const transformedResults = {
           images: resultsWithUrls,
           summary: resultData.analysis_results.summary || null,
           csv_data: resultData.analysis_results.csv_data || null
         };

         setAnalysisResults(transformedResults);

         // Update CSV data if available
         if (resultData.analysis_results.csv_data) {
           setCsvData(resultData.analysis_results.csv_data);
         }

         // Show first image
         setDataURL(resultsWithUrls[0].originalUrl);

         setShowLengthPopup(false);
         setIsEditing(false);
         setIsProcessed(true);

         console.log("Bulk analysis results set successfully");
       } else {
         console.warn("No valid analysis results in bulk response");
       }

     } catch (error) {
       console.error("Error in bulk processing:", error);
     }
   } else {
      // REPLACE THIS PART with Gradio predict_3 call
      const baseURL = "https://aveen007-laserweld.hf.space";
      const apiEndpoint = "/call/predict_3";

      // --- STEP 1: POST /upload (Upload File) ---
      const fileToUpload = allFiles[currentFileIndex] || acceptedFiles[0];
      let uploadedPath = null;

      const uploadFormData = new FormData();
      uploadFormData.append("files", fileToUpload, fileToUpload.name);

      const uploadResponse = await fetch(`${baseURL}/upload`, {
        method: "POST",
        body: uploadFormData,
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed with status ${uploadResponse.status}`);
      }

      const uploadData = await uploadResponse.json();
      uploadedPath = uploadData[0];
      console.log("Uploaded Path:", uploadedPath);

      // --- STEP 2: POST /call/predict_3 (Start Processing) ---
      const callBody = {
        "data": [requestData] // This contains the updated scale params
      };

      console.log("Sending updated scale params to predict_3:", callBody);

      const callResponse = await fetch(`${baseURL}${apiEndpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(callBody),
      });

      if (!callResponse.ok) {
        throw new Error(`Call start failed with status ${callResponse.status}`);
      }

      const callData = await callResponse.json();
      const eventId = callData.event_id;
      console.log("Processing started with updated scale params, Event ID:", eventId);

      // --- STEP 3: GET /call/predict_3/<event_id> (Poll for Result) ---
      const pollInterval = 1000;
      let resultData = null;

      try {
        // Log the event ID for context
        console.log("Starting polling for Event ID:", eventId);

        await new Promise(resolve => setTimeout(resolve, pollInterval));

        const pollUrl = `${baseURL}${apiEndpoint}/${eventId}`;
        const pollResponse = await fetch(pollUrl);

        if (!pollResponse.ok) {
          const errorText = await pollResponse.text();
          console.error(`HTTP Polling Error: Status ${pollResponse.status}. Server response:`, errorText);
          throw new Error(`Polling failed with status ${pollResponse.status}`);
        }

        let pollData;
        let responseText;
        try {
          responseText = await pollResponse.text();
          const dataMatch = responseText.match(/data: (.*)/s);

          if (dataMatch && dataMatch[1]) {
            const jsonString = dataMatch[1].trim();
            pollData = JSON.parse(jsonString);
          } else {
            if (responseText.includes('status')) {
              pollData = JSON.parse(responseText);
            } else {
              pollData = { data: null, status: { status: 'waiting', stage: 'unknown' } };
            }
          }
        } catch (jsonError) {
          console.error("Failed to parse JSON response during polling:", jsonError, "Raw text:", responseText);
          throw new Error("Invalid or unexpected server response.");
        }

        console.log("Received poll data:", pollData);

        if (pollData && pollData.length > 0) {
          // Gradio returned the direct result array (the structure you showed)
          resultData = pollData;
        } else if (pollData.data) {
          // Gradio returned a wrapped result (fallback for older or status results)
          resultData = pollData.data;
        } else if (pollData.error) {
          // Gradio sent a specific error message
          throw new Error(`Gradio processing error: ${pollData.error}`);
        } else {
          // If we received a status update but no final data on this single poll, fail quickly.
          throw new Error("Processing result was not immediately available.");
        }

        if (!resultData) {
          throw new Error("Gradio processing timed out.");
        }

        console.log("Processing complete, final result data:", resultData);

        // Transform the response to match the old format
        console.log("- Is array?", Array.isArray(resultData));
        let analysisResultsData = resultData;

        // Transform the new format to old format
        if (Array.isArray(resultData) && resultData.length > 0) {
          const firstResult = resultData[0];

          if (firstResult.analysis_results) {
            // Create the exact same structure as the old format
            analysisResultsData = {
              images: firstResult.analysis_results.images || [],
              summary: firstResult.analysis_results.summary || null,
              csv_data: firstResult.analysis_results.csv_data || null
            };
            console.log("analysisResults:", analysisResultsData);

            // Update csvData separately
            if (firstResult.analysis_results.csv_data) {
              setCsvData(firstResult.analysis_results.csv_data);
            }

            console.log("Transformed to old format:", analysisResultsData);
          }
        }

        // Set the analysis results with the transformed data
        if (analysisResultsData) {
          setAnalysisResults(analysisResultsData);
          setUploadedURL(resultData);
          setShowLengthPopup(false);
          setIsEditing(false);
          setIsProcessed(true);
          console.log("Analysis results set successfully");
        } else {
          console.warn("No valid analysis results data");
        }

      } catch (error) {
        console.error("Gradio polling failed:", error);
      }

      // Move to next file if in one-by-one mode
      if (processingMode === "one-by-one") {
        setTimeout(goToNextFile, 1000);
      }
    }
  } catch (error) {
    console.error("Error updating scale parameters:", error);
  }
};
const downloadCsv = (data, filename) => {
  if (!data || data.length === 0) return;

  // Define the exact column order you want - now with split key columns
  const columnOrder = [
    'key_number',  // New column for the numeric part
    'key_letter',  // New column for the letter part
    'b_upper',
    't',
    'A',
    'hg',
    'he',
    'hp',
    'hs',
    'hm',
    'hi',
    'b_downer'
  ];

  // Define number formatting precision
  const numberPrecision = {
    'A': 4,
    'b_downer': 2,
    'b_upper': 2,
    'he': 4,
    'hg': 4,
    'hi': 2,
    'hm': 4,
    'hp': 4,
    'hs': 4,
    't': 4
  };

  // Create CSV content
  const csvRows = [
    columnOrder.join(','), // Header row
    ...data.map(row => {
      // Split the key into number and letter parts
      const key = row.key || '';
      const keyMatch = key.match(/(\d+)([a-zA-Z]*)/) || ['', '', ''];
      const keyNumber = keyMatch[1] || '';
      const keyLetter = keyMatch[2] || '';

      return columnOrder.map(fieldName => {
        // Handle the split key columns
        if (fieldName === 'key_number') return keyNumber;
        if (fieldName === 'key_letter') return keyLetter;

        const value = row[fieldName];

        // Format numbers
        if (numberPrecision[fieldName] !== undefined && value !== null) {
          return Number(value).toFixed(numberPrecision[fieldName]);
        }

        // Handle strings
        if (typeof value === 'string') {
          return value.includes(',') ? `"${value}"` : value;
        }

        return value;
      }).join(',');
    })
  ];

  const csvContent = csvRows.join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};
const uploadImage = async () => {
  const baseURL = "https://aveen007-laserweld.hf.space";
  const apiEndpoint = "/call/predict_1";

  // Get all files to upload
  const filesToUpload = allFiles.length > 0 ? allFiles : acceptedFiles;

  if (!filesToUpload || filesToUpload.length === 0) {
    console.error("Please select files first.");
    return;
  }

  const urls = [];

  // Process each file individually
  for (let file of filesToUpload) {
    let uploadedPath = null;
    let resultData = null;

    try {
      // --- STEP 1: POST /upload (Upload Single File) ---
      const uploadFormData = new FormData();
      uploadFormData.append("files", file, file.name);

      const uploadResponse = await fetch(`${baseURL}/upload`, {
        method: "POST",
        body: uploadFormData,
      });

      if (!uploadResponse.ok) {
        console.error(`Upload failed for ${file.name}`);
        continue; // Skip to next file if upload fails
      }

      const uploadData = await uploadResponse.json();
      uploadedPath = uploadData[0]; // Get the first (and only) path
      console.log(`Uploaded Path for ${file.name}:`, uploadedPath);
      uploadedPaths.push(uploadedPath);
      // --- STEP 2: POST /call/predict_1 (Start Processing for single file) ---
      const callBody = {
        "data": [{ "path": uploadedPath }]
      };

      console.log(`Sending ${file.name} for scale detection:`, callBody);

      const callResponse = await fetch(`${baseURL}${apiEndpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(callBody),
      });

      if (!callResponse.ok) {
        console.error(`Scale detection failed for ${file.name}`);
        continue; // Skip to next file if API call fails
      }

      const callData = await callResponse.json();
      const eventId = callData.event_id;
      console.log(`Scale detection started for ${file.name}, Event ID:`, eventId);

      // --- STEP 3: GET /call/predict_1/<event_id> (Poll for Result) ---
      const pollInterval = 1000;

      await new Promise(resolve => setTimeout(resolve, pollInterval));

      const pollUrl = `${baseURL}${apiEndpoint}/${eventId}`;
      const pollResponse = await fetch(pollUrl);

      if (!pollResponse.ok) {
        console.error(`Polling failed for ${file.name}`);
        continue;
      }

      let pollData;
      const responseText = await pollResponse.text();

      try {
        const dataMatch = responseText.match(/data: (.*)/s);
        if (dataMatch && dataMatch[1]) {
          const jsonString = dataMatch[1].trim();
          pollData = JSON.parse(jsonString);
        } else {
          if (responseText.includes('status')) {
            pollData = JSON.parse(responseText);
          } else {
            pollData = { data: null, status: { status: 'waiting', stage: 'unknown' } };
          }
        }
      } catch (jsonError) {
        console.error(`Failed to parse JSON response for ${file.name}`);
        continue;
      }

      console.log(`Received poll data for ${file.name}:`, pollData);

      if (pollData && pollData.length > 0) {
        resultData = pollData[0]; // Get the first result
      } else if (pollData.data) {
        resultData = pollData.data[0];
      } else {
        console.error(`No valid result data for ${file.name}`);
        continue;
      }

      if (resultData) {
        console.log(`Scale detection complete for ${file.name}:`, resultData);
        urls.push(resultData);
      }

    } catch (error) {
      console.error(`Error processing ${file.name}:`, error);
      // Continue with next file even if one fails
    }
  }

  // Set all collected results
  if (urls.length > 0) {
    setUploadedURL(urls);
    setUploadedPaths(uploadedPaths);
    console.log(`Successfully processed ${urls.length} files:`, urls);

    // If we have multiple files, show the first one
    if (urls.length > 1) {
      console.log("Multiple files processed, showing first result");
    }
  } else {
    console.error("No files were successfully processed");
  }
};
// useEffect(() => {
//   if (processingMode === "bulk" && bulkResults.length > 0 && bulkResults[currentBulkIndex]) {
//     const currentImage = bulkResults[currentBulkIndex];
//     const imagePath = `/welding/examples/images/${currentImage.id}.jpg`;
//     setDataURL(imagePath);
//     setAnalysisResults({
//                               images: [currentImage]
//
//                             });
//   }
// }, [currentBulkIndex, bulkResults, processingMode]);
  const { getRootProps, acceptedFiles, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true,
  });
const logNavigation = (direction, newIndex) => {
  console.log(`Navigating ${direction} from index ${currentBulkIndex} to ${newIndex}`);
  console.log('Current bulk results:', bulkResults);
  console.log('Current analysis results:', analysisResults);
  console.log('New image data:', bulkResults[newIndex]);
};

  return (

    <div className="container">
        {isProcessed && analysisResults && (
                        <ToggleControls
                          showMask={showMask}
                          setShowMask={setShowMask}
                          showMeasurements={showOverlay}
                          setShowMeasurements={setShowOverlay}
                          showScaleLine={showScaleLine}
                          setShowScaleLine={setShowScaleLine}
                        />
                      )}
      <div className="zone">
        {allFiles.length > 1 && processingMode === "one-by-one" && (
          <div className="file-counter">
            File {currentFileIndex + 1} of {allFiles.length}
          </div>
        )}

        {dataURL ? (
          <div className="selected">

{showMask && analysisResults?.images[0]?.linesData?.mask_contours && (
  <canvas
    ref={maskCanvasRef}
    width={imageDimensions.width}
    height={imageDimensions.height}
    style={{
      position: "absolute",

      width: "100%",
      height: "auto",
      pointerEvents: "none",
      zIndex: 2
    }}
  />
)}

            {isProcessed && analysisResults ? (

           <>
   <img
      src={dataURL}
      alt="debug"
      style={{
        maxWidth: "100%",
        height: "auto",
        border: "5px solid red", // helps see it
        backgroundColor: "yellow",
      }}
      onLoad={() => console.log("Image loaded:", dataURL)}
      onError={(e) => console.error("Image failed to load", dataURL, e)}
      style={{ display: showOverlay ? 'none' : 'block', maxWidth: '100%' }}
    />
               {showOverlay && (
                 <ImageWithOverlay
                   imageUrl={dataURL}
                   linesData={analysisResults.images[0].linesData}
                   scaleParams={analysisResults.images[0].scaleParams}
                    style={{
                       position: 'absolute',
                       top: 0,
                       left: 0,
                       width: '100%',
                       height: '100%'
                     }}
                 />
               )}

               {showScaleLine && (
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
                     pointerEvents: "none", // prevent interfering with interactions
                     zIndex: 2
                   }}
                 />
               )}
             </>

            ) : (
              <div style={{ position: "relative" }}>
                <img
                  ref={imageRef}
                  src={dataURL}
                  style={{ maxWidth: "100%", height: "auto" }}
                  alt="Uploaded"
                />
                {uploadedURL && imageDimensions.width > 0 &&(
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
      onClick={() => {
        const prevIndex = currentBulkIndex - 1;
          logNavigation('back', prevIndex);
        setCurrentBulkIndex(prevIndex);
        setDataURL(bulkResults[prevIndex].originalUrl);
         setAnalysisResults({
            images: [bulkResults[prevIndex]]
          });
      }}
      disabled={currentBulkIndex === 0}
    >
      Previous
    </button>

    <span>
      Image {currentBulkIndex + 1} of {bulkResults.length}
    </span>

    <button
      onClick={() => {

        const nextIndex = currentBulkIndex + 1;
         logNavigation('forward', nextIndex);
        setCurrentBulkIndex(nextIndex);
        setDataURL(bulkResults[nextIndex].originalUrl);
          setAnalysisResults({
            images: [bulkResults[nextIndex]]
          });
      }}
      disabled={currentBulkIndex === bulkResults.length - 1}
    >
      Next
    </button>
  </div>
)}
 {csvData?.properties && (
   <div className="results-section">
     <h3>Analysis Results</h3>

     {/* Download Button */}
     <button
       onClick={() => downloadCsv(csvData.properties, 'properties.csv')}
       className="download-btn"
     >
       Download CSV
     </button>

     {/* Results Table */}
     <div className="results-table-container">
       <table className="results-table">
         <thead>
           <tr>
             {[
               'key_number',  // Numeric part
               'key_letter',  // Letter part
               'b_upper',
               't',
               'A',
               'hg',
               'he',
               'hp',
               'hs',
               'hm',
               'hi',
               'b_downer'
             ].map(header => (
               <th key={header}>{header}</th>
             ))}
           </tr>
         </thead>
       <tbody>
           {csvData.properties.map((row, index) => {
             // Split the key here for display
             const key = row.key || '';
             const keyMatch = key.match(/(\d+)([a-zA-Z]*)/) || ['', '', ''];
             const keyNumber = keyMatch[1] || '';
             const keyLetter = keyMatch[2] || '';

             return (
               <tr key={index}>
                 {/* Key number and letter cells */}
                 <td>{keyNumber}</td>
                 <td>{keyLetter}</td>

                 {/* Rest of the cells */}
                 {[
                   'b_upper',
                   't',
                   'A',
                   'hg',
                   'he',
                   'hp',
                   'hs',
                   'hm',
                   'hi',
                   'b_downer'
                 ].map(column => {
                   const value = row[column];
                   const formattedValue = typeof value === 'number'
                     ? value.toFixed(column === 'A' ? 4 : 2)
                     : value;
                   return <td key={`${index}-${column}`}>{formattedValue}</td>;
                 })}
               </tr>
             );
           })}
         </tbody>
       </table>
     </div>
   </div>
 )}
    </div>
  );

}