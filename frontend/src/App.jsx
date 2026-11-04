import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import ConveyorSimulation from './pages/ConveyorSimulation';
import InspectionHub from './pages/InspectionHub';
import LiveMonitor from './pages/LiveMonitor';
import History from './pages/History';
import Analytics from './pages/Analytics';
import { 
  getHealth, 
  getAnalyticsSummary, 
  getHistory, 
  deleteHistoryRecord, 
  purgeAllHistory,
  reloadModel
} from './services/api';

export default function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [health, setHealth] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [records, setRecords] = useState([]);

  const fetchData = async () => {
    try {
      const [hData, aData, rData] = await Promise.all([
        getHealth().catch(() => null),
        getAnalyticsSummary().catch(() => null),
        getHistory({ limit: 50 }).catch(() => ({ records: [] }))
      ]);
      setHealth(hData);
      setAnalytics(aData);
      setRecords(rData?.records || []);
    } catch (err) {
      console.error("Data fetch error:", err);
    }
  };

  useEffect(() => {
    fetchData();
    const timer = setInterval(() => {
      getAnalyticsSummary().then(setAnalytics).catch(() => {});
      getHistory({ limit: 50 }).then((res) => setRecords(res?.records || [])).catch(() => {});
    }, 12000);
    return () => clearInterval(timer);
  }, []);

  const handleDeleteRecord = async (id) => {
    try {
      await deleteHistoryRecord(id);
      setRecords((prev) => prev.filter((r) => r.id !== id));
      getAnalyticsSummary().then(setAnalytics).catch(() => {});
    } catch (err) {
      console.error(err);
    }
  };

  const handleClearAllHistory = async () => {
    try {
      await purgeAllHistory();
      setRecords([]);
      getAnalyticsSummary().then(setAnalytics).catch(() => {});
    } catch (err) {
      console.error(err);
    }
  };

  const handleReloadModel = async () => {
    try {
      const res = await reloadModel();
      getHealth().then(setHealth);
      alert(res.status === 'success' ? 'Custom EfficientNetB0 weights reloaded.' : 'Using base model weights.');
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col font-mono selection:bg-white selection:text-black">
      <Header
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        health={health}
        activeAlertsCount={analytics?.active_alerts ?? 0}
        onReloadModel={handleReloadModel}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
        {currentTab === 'dashboard' && (
          <Dashboard
            analytics={analytics}
            records={records}
            setCurrentTab={setCurrentTab}
            onDeleteRecord={handleDeleteRecord}
          />
        )}

        {currentTab === 'conveyor' && (
          <ConveyorSimulation
            onSimulationUpdate={fetchData}
          />
        )}

        {currentTab === 'inspect' && (
          <InspectionHub
            onInspectionDone={fetchData}
          />
        )}

        {currentTab === 'live' && (
          <LiveMonitor
            onAlertLogged={fetchData}
          />
        )}

        {currentTab === 'history' && (
          <History
            records={records}
            onDeleteRecord={handleDeleteRecord}
            onClearAll={handleClearAllHistory}
            onRefresh={fetchData}
          />
        )}

        {currentTab === 'analytics' && (
          <Analytics
            analytics={analytics}
          />
        )}
      </main>

      <footer className="border-t border-neutral-900 bg-black py-4 px-6 text-center text-xs text-neutral-600 font-mono">
        <p>FruitVision AI — Intelligent Fruit Quality Inspection &amp; Sorting System</p>
      </footer>
    </div>
  );
}

