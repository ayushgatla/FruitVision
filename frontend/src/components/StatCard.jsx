import React from 'react';

export default function StatCard({ title, value, subtitle }) {
  return (
    <div className="p-4 bg-black border border-neutral-800 font-mono">
      <div className="text-[11px] text-neutral-400 uppercase tracking-wider">
        {title}
      </div>
      <div className="mt-2 text-2xl font-bold text-white tracking-tight">
        {value}
      </div>
      {subtitle && (
        <p className="mt-1 text-[11px] text-neutral-500 truncate">{subtitle}</p>
      )}
    </div>
  );
}

