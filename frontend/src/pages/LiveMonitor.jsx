import React, { useState, useRef, useEffect } from 'react';
import { inspectLiveFrame } from '../services/api';

export default function LiveMonitor({ onAlertLogged }) {
  const [isActive, setIsActive] = useState(false);
  const [fps, setFps] = useState(0);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [alertNotice, setAlertNotice] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastProcessedTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const fpsTimerRef = useRef(null);
  const isProcessingRef = useRef(false);

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
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const frameBase64 = canvas.toDataURL('image/jpeg', 0.6);
          try {
            const data = await inspectLiveFrame(frameBase64, 'optical_live_cam');
            if (mounted) {
              setCurrentPrediction(data);
              frameCountRef.current += 1;

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
  }, [isActive]);

  return (
    <div className="space-y-6 font-mono">
      
      {/* Header */}
      <div className="pb-3 border-b border-neutral-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-base font-bold text-white uppercase tracking-wider">
            Live Stream Inspection Monitor
          </h1>
          <p className="text-xs text-neutral-500">
            Real-time optical quality inspection from webcam / industrial USB camera
          </p>
        </div>

        <div className="flex items-center space-x-2">
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
          <span className="text-[10px]">[AUTO-LOGGED TO AUDIT]</span>
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
            <canvas ref={canvasRef} className="hidden" />

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

            {/* Overlaid HUD Box */}
            {isActive && currentPrediction && (
              <div className="absolute top-3 left-3 bg-black/80 border border-neutral-700 p-2 text-xs space-y-1">
                <div className="text-[10px] text-neutral-400 uppercase">CLASSIFICATION HUD</div>
                <div className="text-white font-bold">
                  [{currentPrediction.predicted_class.toUpperCase()}] ({Math.round(currentPrediction.confidence * 100)}%)
                </div>
                <div className="text-[10px] text-neutral-400">
                  STATUS: {currentPrediction.status}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Telemetry Panel */}
        <div className="bg-black border border-neutral-800 p-4 flex flex-col justify-between">
          <div>
            <div className="border-b border-neutral-900 pb-2 mb-3">
              <span className="text-xs font-bold uppercase text-white tracking-wider">
                Live Sensor Telemetry
              </span>
            </div>

            {currentPrediction ? (
              <div className="space-y-4">
                <div className="p-3 bg-neutral-950 border border-neutral-800">
                  <div className="text-[10px] text-neutral-500 uppercase">PREDICTED GRADE</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    [{currentPrediction.predicted_class.toUpperCase()}]
                  </div>
                  <div className="text-xs text-neutral-400 mt-1">
                    CONFIDENCE: {Math.round(currentPrediction.confidence * 100)}%
                  </div>
                </div>

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
                Activate camera to view live stream classification.
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

