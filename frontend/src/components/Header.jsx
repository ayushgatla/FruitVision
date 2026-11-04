import React from 'react';

export default function Header({ 
  currentTab, 
  setCurrentTab, 
  health, 
  activeAlertsCount = 0,
  onReloadModel 
}) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'conveyor', label: 'Conveyor Simulation' },
    { id: 'inspect', label: 'Inspection Hub' },
    { id: 'live', label: 'Live Monitor' },
    { id: 'history', label: 'Audit History', badge: activeAlertsCount },
    { id: 'analytics', label: 'Analytics' },
  ];

  return (
    <header className="border-b border-neutral-800 bg-black sticky top-0 z-50 font-mono">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between h-14">
        
        {/* Brand Title */}
        <div 
          className="cursor-pointer" 
          onClick={() => setCurrentTab('dashboard')}
        >
          <div className="font-bold text-white text-sm tracking-wider uppercase flex items-center space-x-2">
            <span className="inline-block w-2 h-2 bg-white"></span>
            <span>FruitVision AI</span>
          </div>
          <div className="text-[10px] text-neutral-500 uppercase tracking-widest">
            Fruit Quality Inspection &amp; Sorting
          </div>
        </div>

        {/* Desktop Navigation Tabs */}
        <nav className="hidden md:flex items-center space-x-1">
          {navItems.map((item) => {
            const active = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentTab(item.id)}
                className={`px-3 py-1.5 text-xs transition-colors cursor-pointer ${
                  active
                    ? 'text-white border-b-2 border-white font-bold'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <span>{item.label}</span>
                {item.badge > 0 && (
                  <span className="ml-1.5 px-1 py-0.2 text-[10px] bg-white text-black font-bold">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Minimal System Telemetry */}
        <div className="flex items-center space-x-3 text-xs">
          <div className="hidden sm:flex items-center space-x-2 text-[11px] text-neutral-400 border border-neutral-800 px-2.5 py-1">
            <span>{health?.device || 'CPU'}</span>
            <span className="text-neutral-700">/</span>
            <span>EFFICIENTNET-B0</span>
            <span className="text-neutral-700">/</span>
            <span className="text-neutral-300">
              {health?.using_custom_weights ? 'CUSTOM WEIGHTS' : 'BASE MODEL'}
            </span>
            {onReloadModel && (
              <button 
                title="Hot-reload weights from disk" 
                onClick={onReloadModel}
                className="text-neutral-500 hover:text-white ml-1 cursor-pointer font-bold"
              >
                [RELOAD]
              </button>
            )}
          </div>

          <div className="border border-neutral-700 px-2 py-1 text-[11px] text-white">
            {health?.status === 'healthy' ? '[ONLINE]' : '[OFFLINE]'}
          </div>
        </div>

      </div>

      {/* Mobile Nav row */}
      <div className="flex md:hidden px-4 py-2 bg-black border-t border-neutral-900 overflow-x-auto space-x-2">
        {navItems.map((item) => {
          const active = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentTab(item.id)}
              className={`px-2.5 py-1 text-xs whitespace-nowrap cursor-pointer ${
                active ? 'text-white border-b border-white font-bold' : 'text-neutral-500'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </div>
    </header>
  );
}

