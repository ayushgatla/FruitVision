import React, { useState } from 'react';
import HistoryTable from '../components/HistoryTable';

export default function History({ 
  records = [], 
  onDeleteRecord, 
  onClearAll, 
  onRefresh 
}) {
  const [search, setSearch] = useState('');
  const [filterCategory, setFilterCategory] = useState('ALL');

  const filteredRecords = records.filter((r) => {
    const matchesSearch = 
      r.id.toString().includes(search) ||
      (r.notes && r.notes.toLowerCase().includes(search.toLowerCase())) ||
      (r.predicted_class && r.predicted_class.toLowerCase().includes(search.toLowerCase())) ||
      (r.inspection_status && r.inspection_status.toLowerCase().includes(search.toLowerCase()));

    const matchesCategory = 
      filterCategory === 'ALL' || 
      r.predicted_class?.toUpperCase() === filterCategory.toUpperCase();

    return matchesSearch && matchesCategory;
  });

  const handleExportCSV = () => {
    if (filteredRecords.length === 0) return;
    const headers = ["ID", "Timestamp", "Class", "Confidence", "Status", "Items", "Notes"];
    const rows = filteredRecords.map(r => [
      r.id,
      r.timestamp,
      r.predicted_class,
      r.confidence,
      r.inspection_status,
      r.fruit_count || 1,
      `"${r.notes || ''}"`
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `FruitVision_Audit_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-4 font-mono">
      
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-neutral-800">
        <div>
          <h1 className="text-base font-bold text-white uppercase tracking-wider">
            Inspection Audit Logs
          </h1>
          <p className="text-xs text-neutral-500">
            Indexed records stored persistently in SQLite database
          </p>
        </div>

        <div className="flex items-center space-x-2 text-xs">
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="px-2.5 py-1 bg-black hover:bg-neutral-900 border border-neutral-700 text-neutral-300 cursor-pointer"
              title="Refresh Logs"
            >
              [REFRESH]
            </button>
          )}

          <button
            onClick={handleExportCSV}
            disabled={filteredRecords.length === 0}
            className="px-2.5 py-1 bg-black hover:bg-neutral-900 border border-neutral-700 text-neutral-300 cursor-pointer disabled:opacity-40"
          >
            [EXPORT CSV]
          </button>

          {onClearAll && records.length > 0 && (
            <button
              onClick={() => {
                if (window.confirm("Purge all fruit inspection audit logs?")) {
                  onClearAll();
                }
              }}
              className="px-2.5 py-1 bg-black hover:bg-neutral-900 border border-neutral-700 text-white cursor-pointer"
            >
              [PURGE ALL]
            </button>
          )}
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-black border border-neutral-800 p-3">
        
        {/* Search */}
        <div className="flex-1 max-w-sm">
          <input
            type="text"
            placeholder="SEARCH BY ID, CLASS, STATUS..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-3 py-1.5 bg-black border border-neutral-700 text-xs text-white placeholder-neutral-600 focus:outline-none focus:border-white font-mono"
          />
        </div>

        {/* Filter Pills */}
        <div className="flex items-center space-x-1 text-xs">
          {[
            { id: 'ALL', label: '[ALL]' },
            { id: 'Good', label: '[GOOD]' },
            { id: 'Damaged', label: '[DAMAGED]' },
            { id: 'Rotten', label: '[ROTTEN]' },
            { id: 'Not Fruits', label: '[NOT FRUITS]' },
          ].map((pill) => (
            <button
              key={pill.id}
              onClick={() => setFilterCategory(pill.id)}
              className={`px-2 py-1 text-[11px] font-mono transition cursor-pointer ${
                filterCategory === pill.id
                  ? 'bg-white text-black font-bold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              {pill.label}
            </button>
          ))}
        </div>

      </div>

      {/* Full Records Table */}
      <HistoryTable
        title="Optical Audit Evidence Log"
        records={filteredRecords}
        onDelete={onDeleteRecord}
      />

    </div>
  );
}

