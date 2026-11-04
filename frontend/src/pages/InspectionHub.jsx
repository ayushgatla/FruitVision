import React, { useState } from 'react';
import { inspectSingleImage, inspectMultiFruit } from '../services/api';

export default function InspectionHub({ onInspectionDone }) {
  const [activeSubTab, setActiveSubTab] = useState('single'); // 'single' | 'multi'

  // Single states
  const [singleFile, setSingleFile] = useState(null);
  const [singlePreview, setSinglePreview] = useState(null);
  const [singleResult, setSingleResult] = useState(null);
  const [singleLoading, setSingleLoading] = useState(false);

  // Multi states
  const [multiFile, setMultiFile] = useState(null);
  const [multiPreview, setMultiPreview] = useState(null);
  const [multiResult, setMultiResult] = useState(null);
  const [multiLoading, setMultiLoading] = useState(false);

  // Error message
  const [error, setError] = useState(null);

  // Handle single file change
  const handleSingleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSingleFile(file);
      setSinglePreview(URL.createObjectURL(file));
      setSingleResult(null);
      setError(null);
    }
  };

  // Run single prediction
  const handleRunSingleInspection = async () => {
    if (!singleFile) return;
    setSingleLoading(true);
    setError(null);
    try {
      const data = await inspectSingleImage(singleFile);
      setSingleResult(data);
      if (onInspectionDone) onInspectionDone();
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Inspection request failed.");
    } finally {
      setSingleLoading(false);
    }
  };

  // Load sample fruit for single inspection
  const handleLoadSingleSample = async (sampleName) => {
    try {
      setSingleLoading(true);
      setError(null);
      const res = await fetch(`/static/samples/${sampleName}`);
      const blob = await res.blob();
      const file = new File([blob], sampleName, { type: 'image/jpeg' });
      setSingleFile(file);
      setSinglePreview(URL.createObjectURL(file));

      const data = await inspectSingleImage(file, `Sample: ${sampleName}`);
      setSingleResult(data);
      if (onInspectionDone) onInspectionDone();
    } catch (err) {
      console.error(err);
      setError("Failed to load sample image. Please upload directly.");
    } finally {
      setSingleLoading(false);
    }
  };

  // Handle multi file change
  const handleMultiFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setMultiFile(file);
      setMultiPreview(URL.createObjectURL(file));
      setMultiResult(null);
      setError(null);
    }
  };

  // Run multi prediction
  const handleRunMultiInspection = async () => {
    if (!multiFile) return;
    setMultiLoading(true);
    setError(null);
    try {
      const data = await inspectMultiFruit(multiFile);
      setMultiResult(data);
      if (onInspectionDone) onInspectionDone();
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Multi-fruit inspection failed.");
    } finally {
      setMultiLoading(false);
    }
  };

  // Load sample multi fruit
  const handleLoadMultiSample = async () => {
    try {
      setMultiLoading(true);
      setError(null);
      const res = await fetch('/static/samples/sample_multi_fruits.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'sample_multi_fruits.jpg', { type: 'image/jpeg' });
      setMultiFile(file);
      setMultiPreview(URL.createObjectURL(file));

      const data = await inspectMultiFruit(file);
      setMultiResult(data);
      if (onInspectionDone) onInspectionDone();
    } catch (err) {
      console.error(err);
      setError("Failed to load sample multi-fruit image.");
    } finally {
      setMultiLoading(false);
    }
  };

  // Download annotated image
  const handleDownloadImage = (url, prefix = "Inspection") => {
    if (!url) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = `FruitVision_${prefix}_${Date.now()}.jpg`;
    a.click();
  };

  return (
    <div className="space-y-6 font-mono">
      
      {/* Header & Sub-Tabs */}
      <div className="pb-3 border-b border-neutral-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-base font-bold text-white uppercase tracking-wider">
            Optical Inspection Hub
          </h1>
          <p className="text-xs text-neutral-500">
            High-precision EfficientNetB0 classification for single fruits &amp; conveyor batches
          </p>
        </div>

        {/* Sub-Tabs */}
        <div className="flex items-center space-x-1 border border-neutral-800 p-0.5">
          <button
            onClick={() => setActiveSubTab('single')}
            className={`px-3 py-1 text-xs cursor-pointer ${
              activeSubTab === 'single' ? 'bg-white text-black font-bold' : 'text-neutral-400 hover:text-white'
            }`}
          >
            [SINGLE FRUIT]
          </button>
          <button
            onClick={() => setActiveSubTab('multi')}
            className={`px-3 py-1 text-xs cursor-pointer ${
              activeSubTab === 'multi' ? 'bg-white text-black font-bold' : 'text-neutral-400 hover:text-white'
            }`}
          >
            [MULTI-FRUIT BATCH]
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-neutral-900 border border-neutral-600 text-xs text-white">
          [ERROR]: {error}
        </div>
      )}

      {/* SUB-TAB 1: SINGLE FRUIT INSPECTION */}
      {activeSubTab === 'single' && (
        <div className="space-y-6">
          
          {/* Quick presets */}
          <div className="bg-black border border-neutral-800 p-3">
            <span className="text-[11px] text-neutral-400 uppercase mr-3">TEST PRESETS:</span>
            <div className="inline-flex flex-wrap gap-1.5 mt-2 sm:mt-0">
              <button
                onClick={() => handleLoadSingleSample('sample_good_apple.jpg')}
                className="px-2 py-0.5 border border-neutral-700 hover:border-white text-[11px] text-neutral-300 hover:text-white cursor-pointer"
              >
                [FRESH APPLE]
              </button>
              <button
                onClick={() => handleLoadSingleSample('sample_good_orange.jpg')}
                className="px-2 py-0.5 border border-neutral-700 hover:border-white text-[11px] text-neutral-300 hover:text-white cursor-pointer"
              >
                [FRESH ORANGE]
              </button>
              <button
                onClick={() => handleLoadSingleSample('sample_damaged_apple.jpg')}
                className="px-2 py-0.5 border border-neutral-700 hover:border-white text-[11px] text-neutral-300 hover:text-white cursor-pointer"
              >
                [DAMAGED APPLE]
              </button>
              <button
                onClick={() => handleLoadSingleSample('sample_rotten_apple.jpg')}
                className="px-2 py-0.5 border border-neutral-700 hover:border-white text-[11px] text-neutral-300 hover:text-white cursor-pointer"
              >
                [ROTTEN APPLE]
              </button>
              <button
                onClick={() => handleLoadSingleSample('sample_rotten_banana.jpg')}
                className="px-2 py-0.5 border border-neutral-700 hover:border-white text-[11px] text-neutral-300 hover:text-white cursor-pointer"
              >
                [ROTTEN BANANA]
              </button>
              <button
                onClick={() => handleLoadSingleSample('sample_not_fruit.jpg')}
                className="px-2 py-0.5 border border-neutral-700 hover:border-white text-[11px] text-neutral-400 hover:text-white cursor-pointer"
              >
                [NOT FRUIT (TEST)]
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Upload Zone */}
            <div className="bg-black border border-neutral-800 p-4 space-y-4">
              <div className="border-b border-neutral-900 pb-2">
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  Target Fruit Image
                </span>
              </div>

              <div className="border-2 border-dashed border-neutral-800 p-6 text-center hover:border-neutral-600 transition">
                <input 
                  type="file" 
                  accept="image/jpeg,image/png,image/jpg" 
                  onChange={handleSingleFileChange}
                  className="hidden" 
                  id="single-file-input"
                />
                <label htmlFor="single-file-input" className="cursor-pointer">
                  {singlePreview ? (
                    <img 
                      src={singlePreview} 
                      alt="Upload Preview" 
                      className="max-h-60 mx-auto object-contain border border-neutral-800"
                    />
                  ) : (
                    <div className="space-y-2 py-8">
                      <div className="text-neutral-400 text-xs">CLICK TO CHOOSE FRUIT IMAGE (JPG/PNG)</div>
                      <div className="text-neutral-600 text-[10px]">Optimal input size: 224x224 RGB</div>
                    </div>
                  )}
                </label>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-[11px] text-neutral-500 truncate max-w-[200px]">
                  {singleFile ? singleFile.name : 'No file selected'}
                </span>
                <button
                  onClick={handleRunSingleInspection}
                  disabled={!singleFile || singleLoading}
                  className="px-4 py-2 bg-white text-black hover:bg-neutral-200 text-xs font-bold transition cursor-pointer disabled:opacity-40"
                >
                  {singleLoading ? '[ANALYZING...]' : '[CLASSIFY QUALITY]'}
                </button>
              </div>
            </div>

            {/* Results Inspection Card */}
            <div className="bg-black border border-neutral-800 p-4 flex flex-col justify-between">
              <div>
                <div className="border-b border-neutral-900 pb-2 flex justify-between items-center mb-4">
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Optical Classification Telemetry
                  </span>
                  {singleResult && (
                    <span className="text-[10px] text-neutral-400">
                      ID: #{singleResult.id}
                    </span>
                  )}
                </div>

                {singleResult ? (
                  <div className="space-y-4">
                    {/* Primary Classification */}
                    <div className="p-3 bg-neutral-950 border border-neutral-800 flex items-center justify-between">
                      <div>
                        <div className="text-[10px] text-neutral-500 uppercase">CLASSIFICATION</div>
                        <div className="text-2xl font-bold text-white mt-1">
                          [{singleResult.predicted_class.toUpperCase()}]
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-neutral-500 uppercase">CONFIDENCE</div>
                        <div className="text-2xl font-bold text-white mt-1">
                          {Math.round(singleResult.confidence * 100)}%
                        </div>
                      </div>
                    </div>

                    {/* Industrial Routing Recommendation */}
                    <div className="p-3 border border-neutral-800 bg-neutral-950 space-y-1">
                      <div className="text-[10px] text-neutral-500 uppercase">SUGGESTED INDUSTRIAL ROUTING</div>
                      <div className="text-xs text-neutral-200 font-bold">
                        {singleResult.suggested_action}
                      </div>
                      <div className="text-[10px] text-neutral-500 mt-1">
                        STATUS: [{singleResult.status}]
                      </div>
                    </div>

                    {/* Class Probabilities Breakdown */}
                    <div className="space-y-2 text-xs">
                      <div className="text-[10px] text-neutral-500 uppercase">
                        Softmax Class Probabilities
                      </div>
                      {Object.entries(singleResult.probabilities || {}).map(([cName, prob]) => (
                        <div key={cName}>
                          <div className="flex justify-between text-[11px] mb-1">
                            <span className="text-neutral-400">{cName.toUpperCase()}</span>
                            <span className="text-white">{Math.round(prob * 100)}%</span>
                          </div>
                          <div className="w-full bg-neutral-900 h-1.5 border border-neutral-800">
                            <div 
                              className="bg-white h-full" 
                              style={{ width: `${Math.round(prob * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Annotated Preview & Download */}
                    {singleResult.annotated_image_url && (
                      <div className="pt-2 border-t border-neutral-900 flex items-center justify-between">
                        <span className="text-[11px] text-neutral-500">ANNOTATED HUD SAVED</span>
                        <button
                          onClick={() => handleDownloadImage(singleResult.annotated_image_url, "Single")}
                          className="text-xs text-white underline cursor-pointer"
                        >
                          [DOWNLOAD ANNOTATED]
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="py-20 text-center text-xs text-neutral-600">
                    Upload an image or pick a test preset to perform optical inspection.
                  </div>
                )}
              </div>

              <div className="mt-4 pt-2 border-t border-neutral-900 text-[10px] text-neutral-500 flex justify-between">
                <span>BACKBONE: EfficientNetB0</span>
                <span>INSPECTION STANDARD: ISO 22000</span>
              </div>
            </div>

          </div>
        </div>
      )}

      {/* SUB-TAB 2: MULTI-FRUIT DETECTION */}
      {activeSubTab === 'multi' && (
        <div className="space-y-6">
          
          <div className="bg-black border border-neutral-800 p-3 flex items-center justify-between">
            <span className="text-[11px] text-neutral-400 uppercase">CONVEYOR BATCH PRESET:</span>
            <button
              onClick={handleLoadMultiSample}
              className="px-2.5 py-1 border border-neutral-700 hover:border-white text-xs text-neutral-300 hover:text-white cursor-pointer"
            >
              [LOAD SAMPLE CONVEYOR BATCH (4 FRUITS)]
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Upload multi fruit */}
            <div className="bg-black border border-neutral-800 p-4 space-y-4">
              <div className="border-b border-neutral-900 pb-2">
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  Batch Inspection Image
                </span>
              </div>

              <div className="border-2 border-dashed border-neutral-800 p-6 text-center hover:border-neutral-600 transition">
                <input 
                  type="file" 
                  accept="image/jpeg,image/png,image/jpg" 
                  onChange={handleMultiFileChange}
                  className="hidden" 
                  id="multi-file-input"
                />
                <label htmlFor="multi-file-input" className="cursor-pointer">
                  {multiPreview ? (
                    <img 
                      src={multiResult?.annotated_image_url || multiPreview} 
                      alt="Batch Preview" 
                      className="max-h-60 mx-auto object-contain border border-neutral-800"
                    />
                  ) : (
                    <div className="space-y-2 py-8">
                      <div className="text-neutral-400 text-xs">UPLOAD IMAGE CONTAINING MULTIPLE FRUITS</div>
                      <div className="text-neutral-600 text-[10px]">Crate, conveyor tray, or fruit basket</div>
                    </div>
                  )}
                </label>
              </div>

              <div className="flex items-center justify-between pt-2">
                <span className="text-[11px] text-neutral-500 truncate max-w-[200px]">
                  {multiFile ? multiFile.name : 'No file selected'}
                </span>
                <button
                  onClick={handleRunMultiInspection}
                  disabled={!multiFile || multiLoading}
                  className="px-4 py-2 bg-white text-black hover:bg-neutral-200 text-xs font-bold transition cursor-pointer disabled:opacity-40"
                >
                  {multiLoading ? '[PROCESSING BATCH...]' : '[INSPECT BATCH]'}
                </button>
              </div>
            </div>

            {/* Multi Results Card */}
            <div className="bg-black border border-neutral-800 p-4 flex flex-col justify-between">
              <div>
                <div className="border-b border-neutral-900 pb-2 flex justify-between items-center mb-4">
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Batch Inspection Summary
                  </span>
                  {multiResult && (
                    <span className="border border-neutral-700 px-1.5 py-0.5 text-[10px] text-white">
                      [{multiResult.overall_status}]
                    </span>
                  )}
                </div>

                {multiResult ? (
                  <div className="space-y-4">
                    {/* Batch KPI Row */}
                    <div className="grid grid-cols-4 gap-2 text-center">
                      <div className="p-2 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-neutral-500">TOTAL</div>
                        <div className="text-lg font-bold text-white">{multiResult.total_fruits}</div>
                      </div>
                      <div className="p-2 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-neutral-500">GOOD</div>
                        <div className="text-lg font-bold text-white">{multiResult.good_count}</div>
                      </div>
                      <div className="p-2 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-neutral-500">DAMAGED</div>
                        <div className="text-lg font-bold text-white">{multiResult.damaged_count}</div>
                      </div>
                      <div className="p-2 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-neutral-500">ROTTEN</div>
                        <div className="text-lg font-bold text-white">{multiResult.rotten_count}</div>
                      </div>
                    </div>

                    {/* Acceptance percentage */}
                    <div className="p-3 border border-neutral-800 bg-neutral-950 flex justify-between items-center">
                      <span className="text-xs text-neutral-400">BATCH QUALITY ACCEPTANCE:</span>
                      <span className="text-base font-bold text-white">{multiResult.acceptance_rate}%</span>
                    </div>

                    {/* Detected Fruit Boxes Table */}
                    <div className="border border-neutral-800 max-h-48 overflow-y-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-neutral-950 text-neutral-500 text-[10px] border-b border-neutral-800">
                          <tr>
                            <th className="p-1.5">#</th>
                            <th className="p-1.5">QUALITY</th>
                            <th className="p-1.5">CONF</th>
                            <th className="p-1.5">BBOX (X, Y)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-900 text-neutral-300 text-[11px]">
                          {multiResult.detections?.map((d, idx) => (
                            <tr key={idx}>
                              <td className="p-1.5 text-neutral-500">{idx + 1}</td>
                              <td className="p-1.5">
                                <span className="border border-neutral-700 px-1 py-0.2 text-[9px] text-white">
                                  {d.label.toUpperCase()}
                                </span>
                              </td>
                              <td className="p-1.5">{Math.round(d.confidence * 100)}%</td>
                              <td className="p-1.5 text-neutral-500">[{d.x1},{d.y1} &rarr; {d.x2},{d.y2}]</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {multiResult.annotated_image_url && (
                      <div className="pt-2 border-t border-neutral-900 flex items-center justify-between">
                        <span className="text-[11px] text-neutral-500">BOUNDING BOX OVERLAY READY</span>
                        <button
                          onClick={() => handleDownloadImage(multiResult.annotated_image_url, "Batch")}
                          className="text-xs text-white underline cursor-pointer"
                        >
                          [DOWNLOAD ANNOTATED BATCH]
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="py-20 text-center text-xs text-neutral-600">
                    Upload a batch image to locate individual fruits and run spatial CNN classification.
                  </div>
                )}
              </div>

              <div className="mt-4 pt-2 border-t border-neutral-900 text-[10px] text-neutral-500 flex justify-between">
                <span>LOCALIZATION: OpenCV Blob / Contour</span>
                <span>CLASSIFIER: EfficientNetB0</span>
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
