import React, { useState, useEffect, useRef } from 'react';
import { getSimulationSample, getSimulationStats, resetSimulation } from '../services/api';

export default function ConveyorSimulation({ onSimulationUpdate }) {
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(1); // 1x, 2x, 4x
  const [fruitFeed, setFruitFeed] = useState('random');
  const [conveyorMode, setConveyorMode] = useState('all');
  const [currentFruit, setCurrentFruit] = useState(null);
  const [stats, setStats] = useState({
    total_sorted: 0,
    good_count: 0,
    damaged_count: 0,
    rotten_count: 0,
    acceptance_rate: 100.0
  });

  // Animation states
  const [beltPosition, setBeltPosition] = useState(0);
  const [gateStates, setGateStates] = useState({
    gate1: false, // Good -> Accepted
    gate2: false, // Damaged -> Secondary
    gate3: false  // Rotten -> Rejected
  });
  const [recentBins, setRecentBins] = useState({
    accepted: [],
    secondary: [],
    rejected: []
  });

  const timerRef = useRef(null);

  // Fetch current stats
  const fetchStats = async () => {
    try {
      const data = await getSimulationStats();
      setStats(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  // Simulation step function
  const runSimulationStep = async () => {
    try {
      // 1. Move conveyor
      setBeltPosition((prev) => (prev + 1) % 100);

      // 2. Fetch sample from backend (Runs CNN inference!)
      const result = await getSimulationSample(fruitFeed, conveyorMode);
      setCurrentFruit(result);

      // 3. Trigger gate diversion based on model decision
      const gate = result.sorting_gate;
      setGateStates({
        gate1: gate === 'GATE_1_ACCEPTED',
        gate2: gate === 'GATE_2_SECONDARY',
        gate3: gate === 'GATE_3_REJECTED'
      });

      // 4. Update bins
      setRecentBins((prev) => {
        const itemInfo = {
          id: result.item_id,
          type: result.fruit_type,
          conf: Math.round(result.confidence * 100)
        };

        if (gate === 'GATE_1_ACCEPTED') {
          return { ...prev, accepted: [itemInfo, ...prev.accepted.slice(0, 4)] };
        } else if (gate === 'GATE_2_SECONDARY') {
          return { ...prev, secondary: [itemInfo, ...prev.secondary.slice(0, 4)] };
        } else {
          return { ...prev, rejected: [itemInfo, ...prev.rejected.slice(0, 4)] };
        }
      });

      // 5. Update counters
      if (result.live_counters) {
        const total = result.live_counters.total_sorted;
        const good = result.live_counters.good_count;
        const acc = total > 0 ? roundOne((good / total) * 100) : 100;
        setStats({
          total_sorted: total,
          good_count: good,
          damaged_count: result.live_counters.damaged_count,
          rotten_count: result.live_counters.rotten_count,
          acceptance_rate: acc
        });
      }

      if (onSimulationUpdate) onSimulationUpdate();

      // Close gates after a brief animation interval
      setTimeout(() => {
        setGateStates({ gate1: false, gate2: false, gate3: false });
      }, 900 / speed);

    } catch (err) {
      console.error("Simulation step error:", err);
    }
  };

  const roundOne = (num) => Math.round(num * 10) / 10;

  // Run loop
  useEffect(() => {
    if (isRunning) {
      const intervalMs = Math.max(1200, 2600 / speed);
      timerRef.current = setInterval(runSimulationStep, intervalMs);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, speed, fruitFeed]);

  const handleReset = async () => {
    setIsRunning(false);
    try {
      await resetSimulation();
      setStats({
        total_sorted: 0,
        good_count: 0,
        damaged_count: 0,
        rotten_count: 0,
        acceptance_rate: 100.0
      });
      setCurrentFruit(null);
      setRecentBins({ accepted: [], secondary: [], rejected: [] });
      setGateStates({ gate1: false, gate2: false, gate3: false });
      if (onSimulationUpdate) onSimulationUpdate();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="space-y-6 font-mono">
      
      {/* Page Header & Controls */}
      <div className="pb-3 border-b border-neutral-800 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-base font-bold text-white uppercase tracking-wider">
            Industrial Conveyor Sorting Simulation
          </h1>
          <p className="text-xs text-neutral-500">
            Automated optical inspection unit with 3-way pneumatic sorting gates
          </p>
        </div>

        {/* Control toolbar */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Line Mode Selector */}
          <select 
            value={conveyorMode}
            onChange={(e) => setConveyorMode(e.target.value)}
            className="px-2 py-1 bg-black border border-neutral-700 text-xs text-neutral-300 font-mono focus:outline-none focus:border-white cursor-pointer"
          >
            <option value="all">LINE: [ALL PRODUCE MIX]</option>
            <option value="apple">LINE: [APPLE DEDICATED]</option>
            <option value="banana">LINE: [BANANA DEDICATED]</option>
          </select>

          {/* Feed Selector */}
          <select 
            value={fruitFeed}
            onChange={(e) => setFruitFeed(e.target.value)}
            className="px-2 py-1 bg-black border border-neutral-700 text-xs text-neutral-300 font-mono focus:outline-none focus:border-white cursor-pointer"
          >
            <option value="random">FEED: [RANDOM PRODUCTION MIX]</option>
            <option value="good">FEED: [GOOD FRUITS ONLY]</option>
            <option value="damaged">FEED: [DAMAGED SAMPLES]</option>
            <option value="rotten">FEED: [ROTTEN DEFECTS]</option>
          </select>

          {/* Speed Selector */}
          <div className="flex items-center space-x-1 border border-neutral-800 p-0.5">
            {[1, 2, 4].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2 py-0.5 text-xs cursor-pointer ${
                  speed === s ? 'bg-white text-black font-bold' : 'text-neutral-500 hover:text-white'
                }`}
              >
                {s}X
              </button>
            ))}
          </div>

          {/* Step Button */}
          <button
            onClick={runSimulationStep}
            disabled={isRunning}
            className="px-2.5 py-1 bg-black border border-neutral-700 hover:bg-neutral-900 text-xs text-neutral-300 cursor-pointer disabled:opacity-40"
          >
            [STEP]
          </button>

          {/* Start/Pause Button */}
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`px-3 py-1 text-xs font-bold transition cursor-pointer ${
              isRunning 
                ? 'bg-neutral-800 text-white border border-neutral-600' 
                : 'bg-white text-black hover:bg-neutral-200'
            }`}
          >
            {isRunning ? '[PAUSE]' : '[START SIMULATION]'}
          </button>

          {/* Reset Button */}
          <button
            onClick={handleReset}
            className="px-2.5 py-1 bg-black border border-neutral-800 hover:border-neutral-600 text-xs text-neutral-400 hover:text-white cursor-pointer"
          >
            [RESET]
          </button>
        </div>
      </div>

      {/* Real-time KPI Counters */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="p-3 bg-black border border-neutral-800">
          <div className="text-[10px] text-neutral-400 uppercase">TOTAL SORTED</div>
          <div className="text-xl font-bold text-white mt-1">{stats.total_sorted}</div>
        </div>
        <div className="p-3 bg-black border border-neutral-800">
          <div className="text-[10px] text-neutral-400 uppercase">ACCEPTED (GOOD)</div>
          <div className="text-xl font-bold text-white mt-1">{stats.good_count}</div>
        </div>
        <div className="p-3 bg-black border border-neutral-800">
          <div className="text-[10px] text-neutral-400 uppercase">SECONDARY (DAMAGED)</div>
          <div className="text-xl font-bold text-white mt-1">{stats.damaged_count}</div>
        </div>
        <div className="p-3 bg-black border border-neutral-800">
          <div className="text-[10px] text-neutral-400 uppercase">REJECTED (ROTTEN)</div>
          <div className="text-xl font-bold text-white mt-1">{stats.rotten_count}</div>
        </div>
        <div className="p-3 bg-black border border-neutral-800 col-span-2 md:col-span-1">
          <div className="text-[10px] text-neutral-400 uppercase">ACCEPTANCE RATE</div>
          <div className="text-xl font-bold text-white mt-1">{stats.acceptance_rate}%</div>
        </div>
      </div>

      {/* Main Simulation Visual Rig */}
      <div className="bg-black border border-neutral-800 p-4 space-y-6">
        
        {/* Conveyor Belt Track Container */}
        <div className="relative bg-neutral-950 border border-neutral-800 p-6 overflow-hidden">
          
          {/* Conveyor Track Rail */}
          <div className="relative h-24 bg-neutral-900 border-y-2 border-neutral-700 flex items-center justify-between px-4">
            
            {/* Belt rollers pattern background */}
            <div 
              className="absolute inset-0 opacity-20 pointer-events-none"
              style={{
                backgroundImage: 'repeating-linear-gradient(90deg, #555, #555 2px, transparent 2px, transparent 24px)',
                backgroundPositionX: `${beltPosition * 12}px`
              }}
            />

            {/* Stage 1: INFEED HOPPER */}
            <div className="relative z-10 flex flex-col items-center">
              <div className="w-16 h-12 bg-neutral-800 border border-neutral-600 flex items-center justify-center text-[10px] text-neutral-400 text-center">
                FEED HOPPER
              </div>
              <span className="text-[9px] text-neutral-500 mt-1">[INLET]</span>
            </div>

            {/* Stage 2: OPTICAL INSPECTION CHAMBER */}
            <div className="relative z-10 flex flex-col items-center">
              <div className={`w-28 h-18 bg-black border-2 ${currentFruit ? 'border-white' : 'border-neutral-600'} flex flex-col items-center justify-center relative overflow-hidden`}>
                
                {/* Laser scan animation when fruit is present */}
                {isRunning && (
                  <div className="absolute inset-x-0 h-0.5 bg-red-500 shadow-[0_0_8px_#ff0000] animate-scan pointer-events-none" />
                )}

                {currentFruit ? (
                  <img 
                    src={currentFruit.image_url} 
                    alt="Active Fruit" 
                    className="w-12 h-12 object-contain"
                  />
                ) : (
                  <span className="text-[10px] text-neutral-600">IDLE</span>
                )}
                <div className="text-[8px] bg-neutral-900 text-neutral-400 w-full text-center py-0.5 border-t border-neutral-800">
                  CAMERA SENSOR
                </div>
              </div>
              <span className="text-[9px] text-neutral-400 mt-1 font-bold">[SCANNER]</span>
            </div>

            {/* Stage 3: SORTING GATE 1 (ACCEPTED) */}
            <div className="relative z-10 flex flex-col items-center">
              <div className={`w-14 h-14 border flex items-center justify-center text-center p-1 transition-all ${
                gateStates.gate1 
                  ? 'bg-white text-black border-white font-bold scale-110' 
                  : 'bg-neutral-900 border-neutral-700 text-neutral-400'
              }`}>
                <span className="text-[9px]">GATE 1<br/>ACCEPTED</span>
              </div>
              <span className="text-[8px] text-neutral-500 mt-1">DIVERTER A</span>
            </div>

            {/* Stage 4: SORTING GATE 2 (SECONDARY) */}
            <div className="relative z-10 flex flex-col items-center">
              <div className={`w-14 h-14 border flex items-center justify-center text-center p-1 transition-all ${
                gateStates.gate2 
                  ? 'bg-neutral-200 text-black border-neutral-300 font-bold scale-110' 
                  : 'bg-neutral-900 border-neutral-700 text-neutral-400'
              }`}>
                <span className="text-[9px]">GATE 2<br/>SECONDARY</span>
              </div>
              <span className="text-[8px] text-neutral-500 mt-1">DIVERTER B</span>
            </div>

            {/* Stage 5: SORTING GATE 3 (REJECTED) */}
            <div className="relative z-10 flex flex-col items-center">
              <div className={`w-14 h-14 border flex items-center justify-center text-center p-1 transition-all ${
                gateStates.gate3 
                  ? 'bg-neutral-400 text-black border-white font-bold scale-110' 
                  : 'bg-neutral-900 border-neutral-700 text-neutral-400'
              }`}>
                <span className="text-[9px]">GATE 3<br/>REJECTED</span>
              </div>
              <span className="text-[8px] text-neutral-500 mt-1">EJECTOR C</span>
            </div>

          </div>

          {/* Belt status pill */}
          <div className="mt-2 flex items-center justify-between text-[11px] text-neutral-500">
            <span>DRIVE: BRUSHLESS SERVO MOTOR (SPEED: {speed}X)</span>
            <span>STATUS: {isRunning ? '[RUNNING]' : '[PAUSED]'}</span>
          </div>
        </div>

        {/* Active Inspection Telemetry HUD & Output Bins */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          
          {/* Active Unit Telemetry Card */}
          <div className="bg-black border border-neutral-800 p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-neutral-900 pb-2 mb-3">
                <span className="text-xs font-bold uppercase text-white tracking-wider">
                  Optical Decision Telemetry
                </span>
                <span className="text-[11px] text-neutral-400">
                  {currentFruit ? currentFruit.item_id : 'NO ITEM IN SCANNER'}
                </span>
              </div>

              {currentFruit ? (
                <div className="space-y-3">
                  <div className="flex items-center space-x-4">
                    <img 
                      src={currentFruit.image_url} 
                      alt="Sample" 
                      className="w-16 h-16 object-contain border border-neutral-800 bg-neutral-950 p-1"
                    />
                    <div className="space-y-1">
                      <div className="text-xs">
                        <span className="text-neutral-500">TYPE: </span>
                        <span className="text-white font-bold">{currentFruit.fruit_type.toUpperCase()}</span>
                      </div>
                      <div className="text-xs">
                        <span className="text-neutral-500">PREDICTED QUALITY: </span>
                        <span className="border border-neutral-600 px-1.5 py-0.2 text-[11px] font-bold text-white ml-1">
                          [{currentFruit.predicted_class.toUpperCase()}]
                        </span>
                      </div>
                      <div className="text-xs">
                        <span className="text-neutral-500">CONFIDENCE: </span>
                        <span className="text-white">{Math.round(currentFruit.confidence * 100)}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Softmax Probability Bars */}
                  <div className="pt-2 border-t border-neutral-900 space-y-1.5 text-[11px]">
                    {Object.entries(currentFruit.probabilities || {}).map(([cName, prob]) => (
                      <div key={cName} className="flex items-center justify-between">
                        <span className="text-neutral-400 w-24">{cName.toUpperCase()}:</span>
                        <div className="flex-1 mx-2 bg-neutral-900 h-1.5 border border-neutral-800">
                          <div 
                            className="bg-white h-full" 
                            style={{ width: `${Math.round(prob * 100)}%` }}
                          />
                        </div>
                        <span className="text-white w-10 text-right">{Math.round(prob * 100)}%</span>
                      </div>
                    ))}
                  </div>

                  <div className="pt-2 border-t border-neutral-900 text-xs">
                    <span className="text-neutral-500">ROUTING ACTION: </span>
                    <span className="text-neutral-200">{currentFruit.action}</span>
                  </div>
                </div>
              ) : (
                <div className="py-10 text-center text-xs text-neutral-600">
                  Press [START SIMULATION] or [STEP] to feed fruit onto the conveyor.
                </div>
              )}
            </div>

            <div className="mt-4 pt-2 border-t border-neutral-900 text-[10px] text-neutral-500 flex justify-between">
              <span>ALGORITHM: EfficientNetB0 CNN</span>
              <span>CYCLE TIME: 85ms</span>
            </div>
          </div>

          {/* Industrial Bins Live State */}
          <div className="bg-black border border-neutral-800 p-4 space-y-3">
            <div className="border-b border-neutral-900 pb-2 flex justify-between items-center">
              <span className="text-xs font-bold uppercase text-white tracking-wider">
                Industrial Collection Bins
              </span>
              <span className="text-[10px] text-neutral-500">LIVE DIVERSIONS</span>
            </div>

            {/* Bin 1: Accepted */}
            <div className="border border-neutral-800 p-2.5 bg-neutral-950">
              <div className="flex justify-between items-center text-xs">
                <span className="text-white font-bold">[BIN 1] ACCEPTED FOR PACKAGING</span>
                <span className="text-white border border-neutral-700 px-1.5 py-0.5">
                  COUNT: {stats.good_count}
                </span>
              </div>
              <p className="text-[10px] text-neutral-500 mt-1">Direct destination: Retail Packing Unit A</p>
              <div className="mt-2 flex items-center space-x-1 overflow-x-auto text-[10px] text-neutral-400">
                {recentBins.accepted.length > 0 ? (
                  recentBins.accepted.map((it, idx) => (
                    <span key={idx} className="border border-neutral-800 px-1.5 py-0.5">
                      {it.type} ({it.conf}%)
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-600 text-[10px]">No items routed yet</span>
                )}
              </div>
            </div>

            {/* Bin 2: Secondary Inspection */}
            <div className="border border-neutral-800 p-2.5 bg-neutral-950">
              <div className="flex justify-between items-center text-xs">
                <span className="text-neutral-300 font-bold">[BIN 2] SECONDARY MANUAL GRADING</span>
                <span className="text-neutral-300 border border-neutral-700 px-1.5 py-0.5">
                  COUNT: {stats.damaged_count}
                </span>
              </div>
              <p className="text-[10px] text-neutral-500 mt-1">Direct destination: Juicing &amp; Trimming Line</p>
              <div className="mt-2 flex items-center space-x-1 overflow-x-auto text-[10px] text-neutral-400">
                {recentBins.secondary.length > 0 ? (
                  recentBins.secondary.map((it, idx) => (
                    <span key={idx} className="border border-neutral-800 px-1.5 py-0.5">
                      {it.type} ({it.conf}%)
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-600 text-[10px]">No items routed yet</span>
                )}
              </div>
            </div>

            {/* Bin 3: Rejected */}
            <div className="border border-neutral-800 p-2.5 bg-neutral-950">
              <div className="flex justify-between items-center text-xs">
                <span className="text-neutral-400 font-bold">[BIN 3] ORGANIC SPOILAGE REJECTION</span>
                <span className="text-neutral-400 border border-neutral-700 px-1.5 py-0.5">
                  COUNT: {stats.rotten_count}
                </span>
              </div>
              <p className="text-[10px] text-neutral-500 mt-1">Direct destination: Bio-Waste / Composting Silo</p>
              <div className="mt-2 flex items-center space-x-1 overflow-x-auto text-[10px] text-neutral-400">
                {recentBins.rejected.length > 0 ? (
                  recentBins.rejected.map((it, idx) => (
                    <span key={idx} className="border border-neutral-800 px-1.5 py-0.5">
                      {it.type} ({it.conf}%)
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-600 text-[10px]">No items routed yet</span>
                )}
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}

