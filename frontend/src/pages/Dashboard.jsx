import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  CartesianGrid
} from 'recharts';
import StatCard from '../components/StatCard';
import HistoryTable from '../components/HistoryTable';

export default function Dashboard({ 
  analytics, 
  records, 
  setCurrentTab, 
  onDeleteRecord 
}) {
  const acceptanceRate = analytics?.acceptance_rate ?? 100;
  const totalDefects = (analytics?.damaged_fruits ?? 0) + (analytics?.rotten_fruits ?? 0);

  return (
    <div className="space-y-6 font-mono">
      
      {/* Top Title & Actions */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-3 border-b border-neutral-800">
        <div>
          <h1 className="text-lg font-bold text-white tracking-wider uppercase">System Dashboard</h1>
          <p className="text-xs text-neutral-500 mt-0.5">Automated Optical Fruit Grading &amp; Real-Time Telemetry</p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setCurrentTab('conveyor')}
            className="px-3 py-1.5 bg-white text-black hover:bg-neutral-200 text-xs font-bold transition cursor-pointer"
          >
            CONVEYOR SIMULATION
          </button>
          <button
            onClick={() => setCurrentTab('inspect')}
            className="px-3 py-1.5 bg-black text-white border border-neutral-700 hover:bg-neutral-900 text-xs transition cursor-pointer"
          >
            INSPECT MEDIA
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          title="Total Inspected"
          value={analytics?.total_inspected ?? 0}
          subtitle="Processed fruit items"
        />
        <StatCard
          title="Acceptance Rate"
          value={`${acceptanceRate}%`}
          subtitle="Target threshold: 90%"
        />
        <StatCard
          title="Defects Identified"
          value={totalDefects}
          subtitle="Damaged + Rotten occurrences"
        />
        <StatCard
          title="Active Alerts"
          value={analytics?.active_alerts ?? 0}
          subtitle="Last 60 minutes"
        />
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        
        {/* Daily Quality Acceptance Trend Line Chart */}
        <div className="lg:col-span-2 bg-black border border-neutral-800 p-4">
          <div className="flex items-center justify-between mb-3 border-b border-neutral-900 pb-2">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-white">
                Quality Acceptance Trajectory
              </h3>
              <p className="text-[10px] text-neutral-500">Daily optical inspection performance (7 days)</p>
            </div>
            <span className="text-[11px] text-neutral-400 border border-neutral-800 px-2 py-0.5">
              AVG: {acceptanceRate}%
            </span>
          </div>

          <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={analytics?.daily_trends || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="#222222" strokeDasharray="2 2" />
                <XAxis dataKey="date" stroke="#666666" fontSize={10} tickLine={false} />
                <YAxis stroke="#666666" fontSize={10} tickLine={false} domain={[70, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#000000', borderColor: '#444444', borderRadius: '0px', fontSize: '11px', color: '#ffffff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="acceptance_rate" 
                  stroke="#ffffff" 
                  strokeWidth={2}
                  dot={{ r: 2, fill: '#ffffff' }}
                  name="Acceptance %"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quality Breakdown Card */}
        <div className="bg-black border border-neutral-800 p-4 flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-white mb-1 border-b border-neutral-900 pb-2">
              Quality Breakdown
            </h3>
            <p className="text-[10px] text-neutral-500 mb-4 mt-2">Distribution across 3 grading classes</p>

            <div className="space-y-4 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-neutral-400">GOOD (ACCEPTED)</span>
                  <span className="text-white">{analytics?.acceptance_rate ?? 94}%</span>
                </div>
                <div className="w-full bg-neutral-900 h-1 border border-neutral-800">
                  <div 
                    className="bg-white h-full" 
                    style={{ width: `${analytics?.acceptance_rate ?? 94}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-neutral-400">DAMAGED (SECONDARY)</span>
                  <span className="text-white">{analytics?.secondary_rate ?? 4}%</span>
                </div>
                <div className="w-full bg-neutral-900 h-1 border border-neutral-800">
                  <div 
                    className="bg-neutral-400 h-full" 
                    style={{ width: `${analytics?.secondary_rate ?? 4}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-neutral-400">ROTTEN (REJECTED)</span>
                  <span className="text-white">{analytics?.rejection_rate ?? 2}%</span>
                </div>
                <div className="w-full bg-neutral-900 h-1 border border-neutral-800">
                  <div 
                    className="bg-neutral-600 h-full" 
                    style={{ width: `${Math.min(100, (analytics?.rejection_rate ?? 2) * 5)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-neutral-900 text-[10px] text-neutral-500 flex justify-between">
            <span>STANDARD: ISO 22000</span>
            <span className="text-neutral-300">ACTIVE PROTOCOL</span>
          </div>
        </div>

      </div>

      {/* Recent Inspection Records */}
      <HistoryTable
        title="Recent Inspection Logs"
        records={records.slice(0, 5)}
        onDelete={onDeleteRecord}
        minimal={true}
      />

    </div>
  );
}

