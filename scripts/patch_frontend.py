"""
Patch frontend ConveyorSimulation.jsx and Analytics.jsx to support fruit conveyor lines and dynamic confusion matrix.
"""

from pathlib import Path

def patch_conveyor():
    path = Path("frontend/src/pages/ConveyorSimulation.jsx")
    content = path.read_text(encoding="utf-8")

    t1 = "const [fruitFeed, setFruitFeed] = useState('random');"
    r1 = "const [fruitFeed, setFruitFeed] = useState('random');\n  const [conveyorMode, setConveyorMode] = useState('all');"
    if t1 in content and "conveyorMode" not in content:
        content = content.replace(t1, r1, 1)

    t2 = "const result = await getSimulationSample(fruitFeed);"
    r2 = "const result = await getSimulationSample(fruitFeed, conveyorMode);"
    if t2 in content:
        content = content.replace(t2, r2, 1)

    t3 = "{/* Feed Selector */}"
    r3 = """{/* Line Mode Selector */}
          <select 
            value={conveyorMode}
            onChange={(e) => setConveyorMode(e.target.value)}
            className="px-2 py-1 bg-black border border-neutral-700 text-xs text-neutral-300 font-mono focus:outline-none focus:border-white cursor-pointer"
          >
            <option value="all">LINE: [ALL PRODUCE MIX]</option>
            <option value="apple">LINE: [APPLE DEDICATED]</option>
            <option value="banana">LINE: [BANANA DEDICATED]</option>
          </select>

          {/* Feed Selector */}"""
    if t3 in content and "LINE: [APPLE DEDICATED]" not in content:
        content = content.replace(t3, r3, 1)

    path.write_text(content, encoding="utf-8")
    print("Patched ConveyorSimulation.jsx successfully.")

def patch_analytics():
    path = Path("frontend/src/pages/Analytics.jsx")
    content = path.read_text(encoding="utf-8")

    # Dynamic confusion matrix rendering
    old_matrix_code = """          <div className="border border-neutral-800 p-4 bg-neutral-950">
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div className="p-2 text-neutral-500 text-[10px]">ACTUAL \\ PRED</div>
              <div className="p-2 border border-neutral-800 text-neutral-400 text-[10px]">PRED GOOD</div>
              <div className="p-2 border border-neutral-800 text-neutral-400 text-[10px]">PRED DAMAGED</div>
              <div className="p-2 border border-neutral-800 text-neutral-400 text-[10px]">PRED ROTTEN</div>

              {/* Row 1: True Good */}
              <div className="p-2 text-neutral-400 text-[10px] flex items-center justify-center">TRUE GOOD</div>
              <div className="p-2 bg-white text-black font-bold">96</div>
              <div className="p-2 bg-neutral-900 text-neutral-300">3</div>
              <div className="p-2 bg-neutral-900 text-neutral-300">1</div>

              {/* Row 2: True Damaged */}
              <div className="p-2 text-neutral-400 text-[10px] flex items-center justify-center">TRUE DAMAGED</div>
              <div className="p-2 bg-neutral-900 text-neutral-300">4</div>
              <div className="p-2 bg-white text-black font-bold">92</div>
              <div className="p-2 bg-neutral-900 text-neutral-300">4</div>

              {/* Row 3: True Rotten */}
              <div className="p-2 text-neutral-400 text-[10px] flex items-center justify-center">TRUE ROTTEN</div>
              <div className="p-2 bg-neutral-950 text-neutral-600 font-bold border border-neutral-800">0</div>
              <div className="p-2 bg-neutral-900 text-neutral-300">3</div>
              <div className="p-2 bg-white text-black font-bold">97</div>
            </div>
          </div>"""

    new_matrix_code = """          <div className="border border-neutral-800 p-4 bg-neutral-950 overflow-x-auto">
            {modelMetrics?.confusion_matrix?.labels && modelMetrics?.confusion_matrix?.matrix ? (
              <table className="w-full text-center text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 text-neutral-500 text-[10px] border border-neutral-800 bg-black">ACTUAL \\ PRED</th>
                    {modelMetrics.confusion_matrix.labels.map((lbl) => (
                      <th key={lbl} className="p-2 border border-neutral-800 text-neutral-300 text-[10px] font-mono bg-black">
                        {lbl.replace('_', ' ').toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {modelMetrics.confusion_matrix.matrix.map((row, rIdx) => {
                    const rowLabel = modelMetrics.confusion_matrix.labels[rIdx];
                    return (
                      <tr key={rowLabel}>
                        <td className="p-2 border border-neutral-800 text-neutral-300 text-[10px] font-mono bg-black text-left">
                          {rowLabel.replace('_', ' ').toUpperCase()}
                        </td>
                        {row.map((val, cIdx) => {
                          const isDiagonal = rIdx === cIdx;
                          return (
                            <td
                              key={cIdx}
                              className={`p-2 border border-neutral-800 font-mono ${
                                isDiagonal
                                  ? 'bg-white text-black font-bold'
                                  : val > 0
                                  ? 'bg-neutral-900 text-neutral-300'
                                  : 'bg-neutral-950 text-neutral-600'
                              }`}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : (
              <div className="text-center text-xs text-neutral-500 py-4">Confusion matrix loading...</div>
            )}
          </div>"""

    # Normalize CRLF
    content_norm = content.replace("\r\n", "\n")
    old_norm = old_matrix_code.replace("\r\n", "\n")

    if old_norm in content_norm:
        content_norm = content_norm.replace(old_norm, new_matrix_code, 1)
        path.write_text(content_norm, encoding="utf-8")
        print("Patched Analytics.jsx successfully.")
    else:
        print("Warning: old_matrix_code not found in Analytics.jsx")

if __name__ == "__main__":
    patch_conveyor()
    patch_analytics()

