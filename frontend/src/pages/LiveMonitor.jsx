import React, { useState, useRef, useEffect } from 'react';
import { inspectLiveFrame } from '../services/api';

export default function LiveMonitor({ onAlertLogged }) {
  const [isActive, setIsActive] = useState(false);
  const [fps, setFps] = useState(0);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [alertNotice, setAlertNotice] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [showHudBoxes, setShowHudBoxes] = useState(true);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastProcessedTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const fpsTimerRef = useRef(null);
  const isProcessingRef = useRef(false);

  // Clear overlay canvas
  const clearOverlayCanvas = () => {
    const canvas = overlayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // Draw bounding boxes on the overlay canvas
  const drawBoundingBoxes = (detections, videoW, videoH) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !videoW || !videoH) return;

    if (canvas.width !== videoW || canvas.height !== videoH) {
      canvas.width = videoW;
      canvas.height = videoH;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!showHudBoxes || !detections || detections.length === 0) return;

    detections.forEach((det, idx) => {
      const { x1, y1, x2, y2, label, confidence } = det;
      const bw = x2 - x1;
      const bh = y2 - y1;

      let strokeColor = '#32cd32'; // Green (Good)
      let tagBg = '#148c14';
      if (label === 'Damaged') {
        strokeColor = '#ffa500'; // Amber (Damaged)
        tagBg = '#b87800';
      } else if (label === 'Rotten') {
        strokeColor = '#ff3232'; // Red (Rotten)
        tagBg = '#b81414';
      } else if (label === 'Not Fruits' || label === 'Not Fruit') {
        strokeColor = '#aaaaaa'; // Gray (Non-Produce)
        tagBg = '#555555';
      }

      // 1. Draw bounding box rectangle
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, bw, bh);

      // 2. Corner bracket accents for industrial precision
      const cl = Math.max(6, Math.min(22, Math.floor(bw / 5), Math.floor(bh / 5)));
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;

      // Top-left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cl);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cl, y1);
      ctx.stroke();

      // Top-right
      ctx.beginPath();
      ctx.moveTo(x2 - cl, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + cl);
      ctx.stroke();

      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - cl);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + cl, y2);
      ctx.stroke();

      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x2 - cl, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - cl);
      ctx.stroke();

      // 3. Labeled tag banner
      const tagText = `[#${idx + 1} ${label.toUpperCase()} ${Math.round(confidence * 100)}%]`;
      ctx.font = 'bold 12px monospace';
      const textWidth = ctx.measureText(tagText).width;
      const tagHeight = 18;

      let tagY = y1 - tagHeight - 2;
      if (tagY < 0) tagY = y1 + 2;

      ctx.fillStyle = tagBg;
      ctx.fillRect(x1, tagY, textWidth + 8, tagHeight);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, tagY, textWidth + 8, tagHeight);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(tagText, x1 + 4, tagY + 13);
    });
  };

  // Start Camera
  const startCamera = async () => {
    setErrorMessage(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { max: 30 } },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      streamRef.current = stream;
      setIsActive(true);

      frameCountRef.current = 0;
      fpsTimerRef.current = setInterval(() => {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
      }, 1000);

    } catch (err) {
      console.error("Camera access error:", err);
      setErrorMessage("Could not access optical camera / webcam. Verify browser permissions.");
    }
  };

  // Stop Camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (fpsTimerRef.current) {
      clearInterval(fpsTimerRef.current);
    }
    clearOverlayCanvas();
    setIsActive(false);
    setFps(0);
    setCurrentPrediction(null);
  };

  // Processing loop
  useEffect(() => {
    if (!isActive) return;

    let mounted = true;

    const processLoop = async (now) => {
      if (!mounted) return;

      if (now - lastProcessedTimeRef.current >= 300 && !isProcessingRef.current) {
        lastProcessedTimeRef.current = now;

        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (video && canvas && video.readyState >= 2) {
          isProcessingRef.current = true;
          const vw = video.videoWidth || 640;
          const vh = video.videoHeight || 480;
          canvas.width = vw;
          canvas.height = vh;

          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, vw, vh);

          const frameBase64 = canvas.toDataURL('image/jpeg', 0.6);
          try {
            const data = await inspectLiveFrame(frameBase64, 'optical_live_cam');
            if (mounted) {
              setCurrentPrediction(data);
              frameCountRef.current += 1;

              // Render real-time bounding boxes directly onto overlay canvas
              drawBoundingBoxes(data.detections || [], vw, vh);

              if (data.alert) {
                setAlertNotice(`[DEFECT FLAGGED]: ${data.predicted_class} fruit detected (${Math.round(data.confidence * 100)}%)`);
                if (onAlertLogged) onAlertLogged();
                setTimeout(() => setAlertNotice(null), 3000);
              }
            }
          } catch (err) {
            // Ignore frame dropout
          } finally {
            isProcessingRef.current = false;
          }
        }
      }

      animationFrameRef.current = requestAnimationFrame(processLoop);
    };

    animationFrameRef.current = requestAnimationFrame(processLoop);

    return () => {
      mounted = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isActive, showHudBoxes]);

  const getClassBadgeStyle = (label) => {
    if (label === 'Good') return 'border-green-600 text-green-400 bg-black';
    if (label === 'Damaged') return 'border-amber-600 text-amber-400 bg-black';
    if (label === 'Not Fruits' || label === 'Not Fruit') return 'border-neutral-600 text-neutral-400 bg-black';
    return 'border-red-600 text-red-400 bg-black';
  };

  return (
    <div className="space-y-6 font-mono">
      
      {/* Header */}
      <div className="pb-3 border-b border-neutral-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-base font-bold text-white uppercase tracking-wider">
            Live Stream Inspection Monitor
          </h1>
          <p className="text-xs text-neutral-500">
            Real-time multi-fruit instance localization, bounding box tracking, and quality sorting from camera feed
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {isActive && (
            <button
              onClick={() => setShowHudBoxes(!showHudBoxes)}
              className={`px-2.5 py-1.5 border text-xs font-bold transition cursor-pointer ${
                showHudBoxes ? 'border-white text-white bg-neutral-900' : 'border-neutral-700 text-neutral-500 bg-black'
              }`}
            >
              [BOX OVERLAYS: {showHudBoxes ? 'ON' : 'OFF'}]
            </button>
          )}

          {isActive ? (
            <button
              onClick={stopCamera}
              className="px-3 py-1.5 bg-neutral-800 text-white border border-neutral-700 hover:bg-neutral-900 text-xs font-bold transition cursor-pointer"
            >
              [STOP CAMERA]
            </button>
          ) : (
            <button
              onClick={startCamera}
              className="px-3 py-1.5 bg-white text-black hover:bg-neutral-200 text-xs font-bold transition cursor-pointer"
            >
              [START CAMERA]
            </button>
          )}
        </div>
      </div>

      {errorMessage && (
        <div className="p-3 bg-neutral-900 border border-neutral-700 text-xs text-white">
          {errorMessage}
        </div>
      )}

      {alertNotice && (
        <div className="p-3 bg-white text-black font-bold text-xs flex justify-between items-center">
          <span>{alertNotice}</span>
          <span className="text-[10px]">[AUTO-LOGGED TO AUDIT WITH BOUNDING BOXES]</span>
        </div>
      )}

      {/* Video & Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Main Video Viewport */}
        <div className="lg:col-span-2 bg-black border border-neutral-800 p-4">
          <div className="flex justify-between items-center mb-3 border-b border-neutral-900 pb-2">
            <span className="text-xs font-bold uppercase text-white tracking-wider">
              Optical Sensor Feed
            </span>
            <div className="flex items-center space-x-2 text-[10px] text-neutral-400">
              <span>FPS: {fps}</span>
              <span>/</span>
              <span>{isActive ? '[FEED ACTIVE]' : '[STANDBY]'}</span>
            </div>
          </div>

          <div className="relative aspect-video bg-neutral-950 border border-neutral-800 flex items-center justify-center overflow-hidden">
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-cover ${isActive ? 'block' : 'hidden'}`}
            />
            {/* Hidden capture canvas */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Real-time bounding box overlay canvas */}
            <canvas 
              ref={overlayCanvasRef} 
              className={`absolute inset-0 w-full h-full pointer-events-none ${isActive ? 'block' : 'hidden'}`} 
            />

            {!isActive && (
              <div className="text-center space-y-2">
                <div className="text-xs text-neutral-500">CAMERA STANDBY</div>
                <button
                  onClick={startCamera}
                  className="px-3 py-1 bg-white text-black text-xs font-bold cursor-pointer"
                >
                  ACTIVATE SENSOR
                </button>
              </div>
            )}

            {/* Overlaid Status HUD Badge */}
            {isActive && currentPrediction && (
              <div className={`absolute top-3 left-3 bg-black/90 p-2 text-xs space-y-1 z-10 border ${
                currentPrediction.predicted_class === 'Good' ? 'border-green-500 text-green-400' :
                currentPrediction.predicted_class === 'Damaged' ? 'border-amber-500 text-amber-400' :
                currentPrediction.predicted_class === 'Rotten' ? 'border-red-500 text-red-400' :
                'border-neutral-700 text-neutral-300'
              }`}>
                <div className="text-[10px] text-neutral-400 uppercase tracking-wider">CLASSIFICATION HUD</div>
                <div className="font-bold flex items-center space-x-2">
                  <span>[{currentPrediction.predicted_class.toUpperCase()}] ({Math.round(currentPrediction.confidence * 100)}%)</span>
                  {currentPrediction.total_fruits > 1 && (
                    <span className="text-[10px] bg-neutral-800 text-yellow-400 px-1 border border-neutral-600">
                      {currentPrediction.total_fruits} FRUITS
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-neutral-400">
                  STATUS: {currentPrediction.status}
                </div>
              </div>
            )}

            {/* Real-time Sensor Guidance Banner */}
            {isActive && currentPrediction && (
              <div className="absolute bottom-2 inset-x-2 flex justify-between items-center px-2.5 py-1 bg-black/85 border border-neutral-800 text-[10px] text-neutral-400 z-10">
                <span className={currentPrediction.total_fruits > 0 ? 'text-white' : 'text-neutral-500'}>
                  {currentPrediction.total_fruits > 0
                    ? `[PRODUCE LOCALIZED: ${currentPrediction.total_fruits} ITEM(S)]`
                    : '[NO PRODUCE DETECTED — PRESENT FRUIT TO SENSOR]'}
                </span>
                <span className="text-neutral-500 font-mono">ISO 22000 OPTICAL HUD</span>
              </div>
            )}
          </div>
        </div>

        {/* Telemetry Panel */}
        <div className="bg-black border border-neutral-800 p-4 flex flex-col justify-between">
          <div>
            <div className="border-b border-neutral-900 pb-2 mb-3 flex justify-between items-center">
              <span className="text-xs font-bold uppercase text-white tracking-wider">
                Live Sensor Telemetry
              </span>
              {currentPrediction && (
                <span className="text-[10px] text-neutral-400">
                  ITEMS: {currentPrediction.total_fruits}
                </span>
              )}
            </div>

            {currentPrediction ? (
              <div className="space-y-4">
                
                {/* Multi-Item Summary if multiple fruits on screen */}
                {currentPrediction.total_fruits > 1 && (
                  <div className="space-y-2">
                    <div className="text-[10px] text-yellow-400 uppercase font-bold">
                      [MULTI-FRUIT SCENE DETECTED]
                    </div>
                    <div className="grid grid-cols-4 gap-1 text-center text-xs">
                      <div className="p-1.5 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-neutral-500">TOTAL</div>
                        <div className="font-bold text-white">{currentPrediction.total_fruits}</div>
                      </div>
                      <div className="p-1.5 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-green-500">GOOD</div>
                        <div className="font-bold text-green-400">{currentPrediction.good_count}</div>
                      </div>
                      <div className="p-1.5 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-amber-500">DAMAGED</div>
                        <div className="font-bold text-amber-400">{currentPrediction.damaged_count}</div>
                      </div>
                      <div className="p-1.5 bg-neutral-950 border border-neutral-800">
                        <div className="text-[9px] text-red-500">ROTTEN</div>
                        <div className="font-bold text-red-400">{currentPrediction.rotten_count}</div>
                      </div>
                    </div>

                    {/* Detected Fruit Boxes Chip List */}
                    <div className="space-y-1 pt-1">
                      <div className="text-[9px] text-neutral-500 uppercase">IDENTIFIED BOUNDING BOXES:</div>
                      <div className="space-y-1 max-h-32 overflow-y-auto">
                        {currentPrediction.detections?.map((d, i) => (
                          <div key={i} className="flex justify-between items-center text-[10px] p-1 bg-neutral-950 border border-neutral-800">
                            <span className="text-neutral-400">#{i + 1} BOX [{d.x1},{d.y1}]</span>
                            <span className={`border px-1 py-0.2 font-bold ${getClassBadgeStyle(d.label)}`}>
                              [{d.label.toUpperCase()}] {Math.round(d.confidence * 100)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Primary Overall Grade */}
                <div className="p-3 bg-neutral-950 border border-neutral-800">
                  <div className="text-[10px] text-neutral-500 uppercase">PRIMARY PREDICTION</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    [{currentPrediction.predicted_class.toUpperCase()}]
                  </div>
                  <div className="text-xs text-neutral-400 mt-1">
                    CONFIDENCE: {Math.round(currentPrediction.confidence * 100)}%
                  </div>
                </div>

                {/* Probabilities Breakdown */}
                <div className="space-y-2 text-xs">
                  <div className="text-[10px] text-neutral-500 uppercase">
                    Softmax Probabilities
                  </div>
                  {Object.entries(currentPrediction.probabilities || {}).map(([cName, prob]) => (
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

                <div className="p-2 border border-neutral-800 text-[11px] text-neutral-300">
                  <div className="text-neutral-500 text-[9px] uppercase">RECOMMENDED ACTION</div>
                  <div className="mt-0.5">{currentPrediction.suggested_action}</div>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-xs text-neutral-600">
                Activate camera to view live stream classification and bounding box localization.
              </div>
            )}
          </div>

          <div className="mt-4 pt-2 border-t border-neutral-900 text-[10px] text-neutral-500 flex justify-between">
            <span>INSPECTION CYCLE: ~300ms</span>
            <span>MODEL: EfficientNetB0</span>
          </div>
        </div>

      </div>

    </div>
  );
}
