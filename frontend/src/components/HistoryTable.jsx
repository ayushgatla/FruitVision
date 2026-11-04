import React, { useState } from 'react';

export default function HistoryTable({ 
  records = [], 
  onDelete, 
  title = "Inspection Records",
  minimal = false 
}) {
  const [selectedImage, setSelectedImage] = useState(null);

  const formatTime = (ts) => {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
             ' ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } catch {
      return ts;
    }
  };

  const getClassBadge = (cls) => {
    if (cls === 'Good') {
      return 'border-neutral-500 text-white bg-neutral-900';
    } else if (cls === 'Damaged') {
      return 'border-neutral-600 text-neutral-300 bg-neutral-900';
    } else if (cls === 'Not Fruits' || cls === 'Not Fruit') {
      return 'border-neutral-700 text-neutral-400 bg-neutral-950 font-bold';
    } else {
      return 'border-white text-black bg-white font-bold';
    }
  };

  return (
    <div className="bg-black border border-neutral-800 font-mono">
      <div className="px-4 py-3 border-b border-neutral-800 flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-wider text-white">
          {title} ({records.length})
        </h3>
      </div>

      {records.length === 0 ? (
        <div className="py-8 text-center text-xs text-neutral-500">
          No fruit inspection records logged yet.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-neutral-950 text-neutral-500 text-[11px] border-b border-neutral-800">
              <tr>
                <th className="py-2.5 px-3">ID</th>
                <th className="py-2.5 px-3">TIMESTAMP</th>
                <th className="py-2.5 px-3">QUALITY</th>
                <th className="py-2.5 px-3">CONFIDENCE</th>
                <th className="py-2.5 px-3">STATUS</th>
                <th className="py-2.5 px-3">ITEMS</th>
                <th className="py-2.5 px-3 text-center">EVIDENCE</th>
                {!minimal && <th className="py-2.5 px-3 text-right">ACTION</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-900 text-neutral-300">
              {records.map((r) => (
                <tr key={r.id} className="hover:bg-neutral-950 transition-colors">
                  <td className="py-2.5 px-3 text-neutral-400">#{r.id}</td>
                  <td className="py-2.5 px-3 text-[11px]">
                    {formatTime(r.timestamp)}
                  </td>
                  <td className="py-2.5 px-3">
                    <span className={`border px-1.5 py-0.5 text-[10px] ${getClassBadge(r.predicted_class)}`}>
                      [{r.predicted_class?.toUpperCase()}]
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    {Math.round(r.confidence * 100)}%
                  </td>
                  <td className="py-2.5 px-3 text-[11px] text-neutral-400 truncate max-w-[140px]">
                    {r.inspection_status}
                  </td>
                  <td className="py-2.5 px-3 text-neutral-400">
                    {r.fruit_count || 1}
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    {(r.evidence_path || r.image_path) ? (
                      <button
                        onClick={() => setSelectedImage(r.evidence_path || r.image_path)}
                        className="text-white hover:underline text-[11px] cursor-pointer"
                      >
                        [VIEW]
                      </button>
                    ) : (
                      <span className="text-neutral-700">-</span>
                    )}
                  </td>
                  {!minimal && (
                    <td className="py-2.5 px-3 text-right">
                      {onDelete && (
                        <button
                          onClick={() => onDelete(r.id)}
                          className="text-neutral-500 hover:text-white cursor-pointer"
                          title="Delete record"
                        >
                          [DELETE]
                        </button>
                      )}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modal for evidence preview */}
      {selectedImage && (
        <div 
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div 
            className="bg-black border border-neutral-700 max-w-2xl w-full p-4 relative"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-3 mb-3 border-b border-neutral-800">
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                Inspection Evidence Snapshot
              </span>
              <button 
                onClick={() => setSelectedImage(null)}
                className="text-neutral-500 hover:text-white text-xs cursor-pointer"
              >
                [CLOSE]
              </button>
            </div>
            <div className="flex items-center justify-center bg-neutral-950 border border-neutral-900 p-2">
              <img 
                src={selectedImage} 
                alt="Inspection Evidence" 
                className="max-h-[70vh] object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

