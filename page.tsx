'use client';

import { useEffect, useState } from 'react';
import { SystemStatus, AIDecision, Position, ActivityEvent } from '@/types/api';
import GovernancePanel from '@/components/GovernancePanel';

export default function CommandCenter() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [aiState, setAiState] = useState<AIDecision | null>(null);
  const [portfolio, setPortfolio] = useState<Position[]>([]);
  const [activity, setActivity] = useState<ActivityEvent[]>([]);
  const [isOffline, setIsOffline] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [connError, setConnError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [resStatus, resAi, resPortfolio, resActivity] = await Promise.all([
        fetch('http://localhost:8000/api/status'),
        fetch('http://localhost:8000/api/ai-state'),
        fetch('http://localhost:8000/api/portfolio'),
        fetch('http://localhost:8000/api/activity'),
      ]);

      setStatus(await resStatus.json());
      setAiState(await resAi.json());
      setPortfolio(await resPortfolio.json());
      setActivity(await resActivity.json());
      setIsOffline(false);
      setLastSync(new Date());
      setConnError(null);
    } catch (err) {
      console.error('Failed to sync with API Server:', err);
      setIsOffline(true);
      setConnError(err instanceof Error ? err.message : 'Unknown Connection Error');
    }
  };

  useEffect(() => {
    fetchData(); // Initial load
    const interval = setInterval(fetchData, 2000); // 2s Polling
    return () => clearInterval(interval);
  }, []);

  if (!status) {
    return (
      <div className="bg-slate-950 text-white h-screen flex flex-col items-center justify-center font-mono p-6">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
        <p className="text-xl font-bold mb-2">Initializing Mission Control...</p>
        <p className="text-slate-500 text-sm mb-6">Waiting for API at http://localhost:8000</p>
        {connError && (
          <div className="bg-red-900/20 border border-red-500/50 p-4 rounded text-red-400 text-xs max-w-md">
            <p className="font-bold mb-1">Diagnostic Output:</p>
            <code>{connError}</code>
          </div>
        )}
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-200 p-4 font-mono relative">
      {/* Offline Overlay */}
      {isOffline && (
        <div className="fixed top-0 left-0 w-full bg-red-600 text-white text-[10px] text-center py-1 z-50 font-bold uppercase tracking-widest animate-pulse">
          ⚠️ Connection to Trading Engine Lost — Data may be stale ⚠️
        </div>
      )}

      {/* SystemStateBar */}
      <div className="grid grid-cols-4 gap-4 mb-6 bg-slate-900 border border-slate-800 p-4 rounded-lg">
        <div>
          <div className="flex items-center gap-2">
             <div className={`w-2 h-2 rounded-full ${isOffline ? 'bg-red-500' : 'bg-green-500 animate-pulse'}`} />
             <p className="text-[10px] text-slate-500 uppercase">Engine Link</p>
          </div>
          <p className="text-xs text-slate-500 uppercase">Net Asset Value</p>
          <p className="text-2xl font-bold text-green-400">${status.nav.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500 uppercase">System Mode</p>
          <p className={`text-2xl font-bold ${status.mode === 'READY' ? 'text-blue-400' : 'text-yellow-500 animate-pulse'}`}>
            {status.mode}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 uppercase">Growth Session</p>
          <p className="text-2xl font-bold text-green-400">+{status.growth_pct}%</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500 uppercase">Health Status</p>
          <p className={`text-lg font-bold ${status.health === 'HEALTHY' ? 'text-green-500' : 'text-red-500'}`}>
            ● {status.health}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* AIBrainPanel */}
        <section className="col-span-8 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
            <h2 className="text-sm font-bold text-slate-400 uppercase mb-4 tracking-widest">AI Brain: Latest Rationale</h2>
            {aiState ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-2xl font-bold text-white">{aiState.symbol}</span>
                  <span className={`px-3 py-1 rounded text-xs font-bold ${aiState.action === 'BUY' ? 'bg-green-900 text-green-300' : 'bg-slate-800 text-slate-400'}`}>
                    {aiState.action} (Conf: {(aiState.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
                <p className="text-slate-400 italic text-sm">"{aiState.rationale}"</p>
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(aiState.gates).map(([gate, ok]) => (
                    <div key={gate} className={`text-[10px] p-2 rounded border ${ok ? 'border-green-800 bg-green-950/30 text-green-400' : 'border-red-900 bg-red-950/30 text-red-400'}`}>
                      {gate.toUpperCase()}: {ok ? 'PASSED' : 'REJECTED'}
                    </div>
                  ))}
                </div>
              </div>
            ) : <p className="text-slate-600">No active signals.</p>}
          </div>
          
          {/* ActivityTimeline Placeholder */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 h-64 overflow-y-auto">
            <h2 className="text-sm font-bold text-slate-400 uppercase mb-4 tracking-widest">Activity Feed</h2>
            {activity.map(ev => (
              <div key={ev.id} className="text-xs py-2 border-b border-slate-800 flex gap-4">
                <span className="text-slate-600">{new Date(ev.timestamp).toLocaleTimeString()}</span>
                <span className={ev.type === 'THROTTLE' ? 'text-yellow-500' : 'text-slate-300'}>{ev.message}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Governance & Controls */}
        <section className="col-span-4 space-y-6">
          <GovernancePanel />
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Portfolio</h2>
              {lastSync && <span className="text-[10px] text-slate-600">Sync: {lastSync.toLocaleTimeString()}</span>}
            </div>
            <div className="space-y-3">
              {portfolio.map(pos => (
                <div key={pos.symbol} className="bg-slate-950 p-3 rounded border border-slate-800">
                  <div className="flex justify-between font-bold">
                    <span>{pos.symbol}</span>
                    <span className={pos.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}>{pos.pnl_pct}%</span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">Value: ${pos.value_usdt} | AI: {pos.ai_guidance}</div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}