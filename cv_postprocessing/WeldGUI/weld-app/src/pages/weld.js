import React, { useState, useCallback } from "react";
import { Plus, X, ZoomIn, ZoomOut } from "lucide-react";
import {userDropzone} from "react-dropzone"

export default function WeldGeo() {
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [zoom, setZoom] = useState(1);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0] || e.dataTransfer?.files?.[0];
    if (file && file.type.match("image.*")) {
      setPreview(URL.createObjectURL(file));
    }
  };
const{
    getRootProps,
    acceptedFiles,
    getInputProps,
    isDragActive,
}=userDropzone({onDrop})
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleImageChange(e);
  }, []);

  return (

  <div className="drop-zone"
    <div className="flex flex-col items-center justify-start min-h-screen p-4 bg-gray-50">
      <h1 className="text-3xl font-bold mb-6">Weld Geometry Upload</h1>

      {/* Upload Box */}
      <div className="w-full flex justify-center">
        <div
          className={`relative w-[10px] h-46 border-2 rounded-lg overflow-hidden cursor-pointer flex items-center justify-center transition-colors duration-200 ${
            isDragging ? "border-blue-500 bg-blue-50" : "border-dashed border-gray-200 bg-white"
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById("upload")?.click()}
          style={{ boxShadow: "0 0 0 1px #d1d5db" }}
        >
          {preview ? (
            <>
              <div className="absolute inset-0 flex items-center justify-center">
                <img
                  src={preview}
                  alt="Uploaded Weld"
                  className="object-scale-down max-h-full max-w-full"
                  style={{ transform: `scale(${zoom})` }}
                />
              </div>

              {/* Remove Button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setPreview(null);
                  setZoom(1);
                }}
                className="absolute top-2 right-2 bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-red-600 z-10"
              >
                <X className="w-4 h-4" />
              </button>

              {/* Zoom Controls */}
              <div className="absolute bottom-2 right-2 flex gap-2 items-center bg-white bg-opacity-80 px-2 py-1 rounded">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setZoom((z) => Math.max(0.5, z - 0.1));
                  }}
                >
                  <ZoomOut className="w-4 h-4 text-gray-700" />
                </button>
                <span className="text-sm text-gray-700">{(zoom * 100).toFixed(0)}%</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setZoom((z) => Math.min(3, z + 0.1));
                  }}
                >
                  <ZoomIn className="w-4 h-4 text-gray-700" />
                </button>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center text-gray-500">
              <Plus className="w-12 h-12 mb-2" />
              <p className="text-center text-sm">Click or drag an image here</p>
              <input
                type="file"
                accept="image/*"
                id="upload"
                className="hidden"
                onChange={handleImageChange}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
