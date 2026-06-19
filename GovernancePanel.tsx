'use client';

import { useState } from 'react';

export default function GovernancePanel() {
  const [pending, setPending] = useState<string | null>(null);

  const sendControl = async (endpoint: string, actionName: string) => {
    if (!confirm(`Are you sure you want to ${actionName}?`)) return;
    
    setPending(actionName);
    try {
      const res = await fetch(`http://localhost:8000/api/control/${endpoint}`, {
        method: 'POST',
      });
      if (res.ok) alert(`Success: ${actionName} executed.`);
    } catch (err) {
      alert('Governance action failed. Check connection.');
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="bg-slate-900 border border-red-900/50 rounded-lg p-6">
      <h2 className="text-sm font-bold text-red-400 uppercase mb-4 tracking-widest flex items-center gap-2">
        <span className="animate-pulse">⚠️</span> Governance & Override
      </h2>
      
      <div className="space-y-3">
        <button 
          onClick={() => sendControl('pause-buying', 'PAUSE BUYING')}
          disabled={!!pending}
          className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 rounded text-xs font-bold border border-slate-700 transition-colors"
        >
          PAUSE BUY SIGNALS
        </button>

        <button 
          onClick={() => sendControl('force-safe-mode', 'ENABLE SAFE MODE')}
          disabled={!!pending}
          className="w-full bg-blue-900/20 hover:bg-blue-900/40 text-blue-400 py-2 rounded text-xs font-bold border border-blue-900/50 transition-colors"
        >
          FORCE SAFE MODE
        </button>

        <button 
          onClick={() => sendControl('cancel-open-orders', 'CANCEL ALL ORDERS')}
          disabled={!!pending}
          className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 rounded text-xs font-bold border border-slate-700 transition-colors"
        >
          CANCEL OPEN ORDERS
        </button>

        <button 
          onClick={() => sendControl('pause-all', 'EMERGENCY HALT')}
          disabled={!!pending}
          className="w-full bg-red-600 hover:bg-red-700 text-white py-3 rounded text-xs font-bold shadow-lg shadow-red-900/20 transition-all uppercase"
        >
          🚨 EMERGENCY HALT 🚨
        </button>
      </div>
    </div>
  );
}