import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Activity, Play, Cpu, Database, Zap, Clock, Terminal, BarChart3, Layers, Brain, Grid, Box, Sparkles, RefreshCw } from 'lucide-react'
import { cn } from './lib/utils'

// --- Types ---
interface TrainConfig {
  dataset: string
  epochs: number
  batch_size: number
  seq_len: number
  n_models: number
  device: string
}

interface WorkerStatus {
  rank: number
  model: string
  status: string
  duration: number
}

interface TrainingStatus {
  active: boolean
  progress: number
  logs: string[]
  workers: WorkerStatus[]
}

interface Results {
  ready: boolean
  metrics?: {
    auc_roc: number
    pr_auc: number
  }
  plot_data?: {
    scores: number[]
    labels: number[]
  }
}

const API_URL = 'http://localhost:8000'

// Add Ngrok bypass header for Axios
axios.defaults.headers.common['ngrok-skip-browser-warning'] = 'true'

// --- Model Config ---
const MODEL_INFO: Record<string, { icon: any; color: string; bgColor: string; description: string }> = {
  lstm: {
    icon: Brain,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10 border-purple-500/20',
    description: 'Long Short-Term Memory'
  },
  cnn: {
    icon: Grid,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10 border-blue-500/20',
    description: 'Convolutional Neural Network'
  },
  dense: {
    icon: Box,
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10 border-emerald-500/20',
    description: 'Fully Connected Network'
  },
  transformer: {
    icon: Sparkles,
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10 border-amber-500/20',
    description: 'Self-Attention Architecture'
  },
  gru: {
    icon: RefreshCw,
    color: 'text-rose-400',
    bgColor: 'bg-rose-500/10 border-rose-500/20',
    description: 'Gated Recurrent Unit'
  }
}

// --- Components ---

const BentoBox = ({ children, className, title, icon: Icon }: { children: React.ReactNode, className?: string, title?: string, icon?: any }) => (
  <div className={cn("bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 backdrop-blur-sm flex flex-col", className)}>
    {title && (
      <div className="flex items-center gap-2 mb-4 text-zinc-400">
        {Icon && <Icon size={16} />}
        <h3 className="text-sm font-medium uppercase tracking-wider">{title}</h3>
      </div>
    )}
    {children}
  </div>
)

const MetricCard = ({ label, value, colorClass }: { label: string, value: string, colorClass: string }) => (
  <div className="bg-zinc-950/50 rounded-xl p-4 border border-zinc-800/50">
    <div className="text-zinc-500 text-xs font-medium uppercase tracking-wide mb-1">{label}</div>
    <div className={cn("text-2xl font-bold font-mono", colorClass)}>{value}</div>
  </div>
)

const StatusBadge = ({ status }: { status: string }) => {
  const isSuccess = status === 'success'
  const isError = status === 'error' || status === 'failed'
  const isTraining = status === 'training'

  return (
    <span className={cn(
      "px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide border",
      isSuccess ? "bg-emerald-950/30 text-emerald-400 border-emerald-900/50" :
        isError ? "bg-rose-950/30 text-rose-400 border-rose-900/50" :
          isTraining ? "bg-amber-950/30 text-amber-400 border-amber-900/50 animate-pulse" :
            "bg-zinc-800 text-zinc-400 border-zinc-700"
    )}>
      {status}
    </span>
  )
}

const ModelBadge = ({ model }: { model: string }) => {
  const info = MODEL_INFO[model.toLowerCase()] || MODEL_INFO.dense
  const Icon = info.icon

  return (
    <div className={cn("flex items-center gap-2 px-2 py-1 rounded-lg border", info.bgColor)}>
      <Icon size={12} className={info.color} />
      <span className={cn("text-xs font-medium uppercase", info.color)}>{model}</span>
    </div>
  )
}

const DataInspector = ({ dataset }: { dataset: string }) => {
  const [data, setData] = useState<{ normal: number[], anomaly: number[] } | null>(null)

  const fetchData = async () => {
    try {
      const res = await axios.get(`${API_URL}/sample-data?dataset=${dataset}`)
      setData(res.data)
    } catch (e) {
      console.error("Failed to fetch sample data", e)
    }
  }

  useEffect(() => {
    fetchData()
  }, [dataset])

  if (!data) return null

  const chartData = data.normal.map((v, i) => ({
    index: i,
    normal: v,
    anomaly: data.anomaly[i] || 0
  }))

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs text-zinc-500 font-mono">
          Displaying raw samples from <span className="text-zinc-300">{dataset}</span>
        </div>
        <button
          onClick={fetchData}
          className="text-[10px] bg-zinc-800 hover:bg-zinc-700 text-zinc-400 px-2 py-1 rounded border border-zinc-700 transition-colors"
        >
          Refresh Samples
        </button>
      </div>

      <div className="flex-1 min-h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#09090b',
                borderColor: '#27272a',
                color: '#f4f4f5',
                fontSize: '12px'
              }}
            />
            <Line type="monotone" dataKey="normal" stroke="#34d399" strokeWidth={2} dot={false} name="Normal Pattern" />
            <Line type="monotone" dataKey="anomaly" stroke="#f472b6" strokeWidth={2} dot={false} name="Anomaly Pattern" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-4 mt-2 justify-center">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Normal</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-pink-400" />
          <span className="text-[10px] text-zinc-400 uppercase tracking-wider">Anomaly</span>
        </div>
      </div>
    </div>
  )
}

// --- Model Overview Component ---
const ModelOverview = () => {
  const models = ['lstm', 'cnn', 'dense', 'transformer', 'gru']

  return (
    <div className="grid grid-cols-5 gap-3">
      {models.map((model) => {
        const info = MODEL_INFO[model]
        const Icon = info.icon
        return (
          <div
            key={model}
            className={cn(
              "flex flex-col items-center justify-center p-4 rounded-xl border transition-all hover:scale-105",
              info.bgColor
            )}
          >
            <Icon size={24} className={info.color} />
            <span className={cn("text-sm font-bold uppercase mt-2", info.color)}>{model}</span>
            <span className="text-[9px] text-zinc-500 text-center mt-1">{info.description}</span>
          </div>
        )
      })}
    </div>
  )
}

// --- Main App ---

function App() {
  const [config, setConfig] = useState<TrainConfig>({
    dataset: 'nyc_taxi',
    epochs: 10,
    batch_size: 32,
    seq_len: 64,
    n_models: 5,  // Default to 5 models
    device: 'cpu'
  })

  const [status, setStatus] = useState<TrainingStatus>({
    active: false,
    progress: 0,
    logs: [],
    workers: []
  })

  const [results, setResults] = useState<Results>({ ready: false })
  const [chartData, setChartData] = useState<any[]>([])
  const [backendUp, setBackendUp] = useState(false)
  const hasFetchedResults = useRef(false)

  // SSE Connection
  useEffect(() => {
    let eventSource: EventSource | null = null;

    const connectSSE = () => {
      eventSource = new EventSource(`${API_URL}/stream-status`)

      eventSource.onopen = () => {
        setBackendUp(true)
      }

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setStatus(data)
        } catch (e) {
          console.error("Failed to parse SSE data", e)
        }
      }

      eventSource.onerror = (err) => {
        console.error("SSE Error", err)
        setBackendUp(false)
        eventSource?.close()
        // Retry connection after 3s
        setTimeout(connectSSE, 3000)
      }
    }

    connectSSE()

    return () => {
      eventSource?.close()
    }
  }, [])

  // Effect to fetch results when training completes
  useEffect(() => {
    if (!status.active && status.progress === 100 && !results.ready && !hasFetchedResults.current) {
      hasFetchedResults.current = true
      fetchResults()
    }
  }, [status.active, status.progress, results.ready])

  const fetchResults = async () => {
    try {
      const res = await axios.get(`${API_URL}/results`)
      if (res.data.ready) {
        setResults(res.data)
        // Downsample for chart if needed, but backend does it too.
        // Just mapping for Recharts
        const data = res.data.plot_data.scores.map((score: number, i: number) => ({
          index: i,
          score: score,
          label: res.data.plot_data.labels[i]
        }))
        setChartData(data)
      }
    } catch (e) {
      console.error("Failed to fetch results", e)
    }
  }

  const startTraining = async () => {
    try {
      setResults({ ready: false })
      setChartData([])
      hasFetchedResults.current = false
      await axios.post(`${API_URL}/train`, config)
    } catch (e) {
      alert("Failed to start training")
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-4 md:p-8 font-sans selection:bg-blue-500/30">
      <div className="max-w-7xl mx-auto space-y-4">

        {/* --- Hero Section --- */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
          <BentoBox className="md:col-span-8 lg:col-span-9 justify-center min-h-[160px] relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative z-10">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
                  <Layers size={24} />
                </div>
                <h1 className="text-3xl font-bold tracking-tight text-white">
                  Anomaly Detection <span className="text-zinc-500">Framework</span>
                </h1>
              </div>
              <p className="text-zinc-400 max-w-2xl text-lg">
                Research-grade parallel ensemble learning for time-series analysis.
                Supports <span className="text-purple-400 font-medium">LSTM</span>, <span className="text-blue-400 font-medium">CNN</span>, <span className="text-emerald-400 font-medium">Dense</span>, <span className="text-amber-400 font-medium">Transformer</span>, <span className="text-rose-400 font-medium">GRU</span> architectures.
              </p>
            </div>
          </BentoBox>

          <BentoBox className="md:col-span-4 lg:col-span-3 justify-center items-center text-center">
            <div className={cn(
              "w-3 h-3 rounded-full mb-3 shadow-[0_0_10px_currentColor]",
              backendUp ? "bg-emerald-500 text-emerald-500" : "bg-rose-500 text-rose-500"
            )} />
            <div className="text-sm font-medium text-zinc-300">
              {backendUp ? "System Online" : "Backend Offline"}
            </div>
            {!backendUp && API_URL.includes('ngrok') && (
              <a
                href={API_URL}
                target="_blank"
                rel="noreferrer"
                className="text-[10px] text-blue-400 hover:underline mt-1 block"
              >
                Click to bypass Ngrok warning
              </a>
            )}
            <div className="text-xs text-zinc-500 mt-1 font-mono">{API_URL}</div>
          </BentoBox>
        </div>

        {/* --- Model Architectures Overview --- */}
        <BentoBox title="Ensemble Architectures" icon={Layers}>
          <ModelOverview />
        </BentoBox>

        {/* --- Main Content Grid --- */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">

          {/* Config Sidebar */}
          <BentoBox title="Configuration" icon={Zap} className="md:col-span-4 lg:col-span-3 h-full">
            <div className="space-y-5">
              <div className="space-y-1.5">
                <label className="text-xs text-zinc-500 font-medium ml-1">Dataset</label>
                <div className="relative">
                  <Database className="absolute left-3 top-2.5 text-zinc-600" size={14} />
                  <select
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg pl-9 pr-3 py-2 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500 outline-none appearance-none transition-all hover:border-zinc-700"
                    value={config.dataset}
                    onChange={(e) => setConfig({ ...config, dataset: e.target.value })}
                    disabled={status.active}
                  >
                    <option value="nyc_taxi">NYC Taxi (Real)</option>
                    <option value="synthetic">Synthetic (Sine)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-xs text-zinc-500 font-medium ml-1">Epochs</label>
                  <input
                    type="number"
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:ring-1 focus:ring-blue-500 outline-none transition-all hover:border-zinc-700"
                    value={config.epochs}
                    onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                    disabled={status.active}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs text-zinc-500 font-medium ml-1">Batch Size</label>
                  <input
                    type="number"
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:ring-1 focus:ring-blue-500 outline-none transition-all hover:border-zinc-700"
                    value={config.batch_size}
                    onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                    disabled={status.active}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs text-zinc-500 font-medium ml-1">Number of Models</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm focus:ring-1 focus:ring-blue-500 outline-none transition-all hover:border-zinc-700"
                  value={config.n_models}
                  onChange={(e) => setConfig({ ...config, n_models: parseInt(e.target.value) })}
                  disabled={status.active}
                />
                <p className="text-[10px] text-zinc-600 ml-1">Models cycle through: LSTM → CNN → Dense → Transformer → GRU</p>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs text-zinc-500 font-medium ml-1">Compute Device</label>
                <div className="relative">
                  <Cpu className="absolute left-3 top-2.5 text-zinc-600" size={14} />
                  <select
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg pl-9 pr-3 py-2 text-sm focus:ring-1 focus:ring-blue-500 outline-none appearance-none transition-all hover:border-zinc-700"
                    value={config.device}
                    onChange={(e) => setConfig({ ...config, device: e.target.value })}
                    disabled={status.active}
                  >
                    <option value="cpu">CPU (Standard)</option>
                    <option value="cuda">GPU (CUDA)</option>
                  </select>
                </div>
              </div>

              <div className="pt-4">
                <button
                  onClick={startTraining}
                  disabled={status.active || !backendUp}
                  className={cn(
                    "w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all duration-300",
                    status.active
                      ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                      : "bg-white text-black hover:bg-zinc-200 hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-white/5"
                  )}
                >
                  {status.active ? (
                    <>
                      <Activity className="animate-spin" size={18} />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play size={18} fill="currentColor" />
                      Start Pipeline
                    </>
                  )}
                </button>
              </div>
            </div>
          </BentoBox>

          {/* Chart & Metrics Area */}
          <div className="md:col-span-8 lg:col-span-9 grid grid-cols-1 md:grid-cols-2 gap-4">

            {/* Metrics Row */}
            <BentoBox title="Performance Metrics" icon={BarChart3} className="col-span-1 md:col-span-2">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  label="AUC-ROC"
                  value={results.metrics ? results.metrics.auc_roc.toFixed(4) : '--'}
                  colorClass="text-emerald-400"
                />
                <MetricCard
                  label="PR-AUC"
                  value={results.metrics ? results.metrics.pr_auc.toFixed(4) : '--'}
                  colorClass="text-blue-400"
                />
                <MetricCard
                  label="Models"
                  value={config.n_models.toString()}
                  colorClass="text-zinc-100"
                />
                <MetricCard
                  label="Status"
                  value={status.active ? "Running" : "Idle"}
                  colorClass={status.active ? "text-amber-400" : "text-zinc-500"}
                />
              </div>
            </BentoBox>

            {/* Main Chart */}
            <BentoBox className="col-span-1 md:col-span-2 min-h-[400px]">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2 text-zinc-400">
                  <Activity size={16} />
                  <h3 className="text-sm font-medium uppercase tracking-wider">Anomaly Score Visualization</h3>
                </div>
                {results.ready && (
                  <span className="text-[10px] bg-emerald-900/30 text-emerald-400 px-2 py-1 rounded-full border border-emerald-800/50 uppercase tracking-wide font-medium">
                    Evaluation Complete
                  </span>
                )}
              </div>

              <div className="flex-1 w-full min-h-[300px]">
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                      <XAxis
                        dataKey="index"
                        stroke="#52525b"
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        stroke="#52525b"
                        tick={{ fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#09090b',
                          borderColor: '#27272a',
                          color: '#f4f4f5',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                        }}
                        itemStyle={{ fontSize: '12px' }}
                        labelStyle={{ fontSize: '12px', color: '#a1a1aa' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: '#3b82f6' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-zinc-700 space-y-3">
                    <div className="p-4 bg-zinc-900 rounded-full">
                      <Activity size={32} className="opacity-50" />
                    </div>
                    <p className="text-sm font-medium">No visualization data available</p>
                    <p className="text-xs">Run the training pipeline to generate scores</p>
                  </div>
                )}
              </div>
            </BentoBox>
          </div>
        </div>

        {/* --- Data Inspector ---
        <div className="grid grid-cols-1">
          <BentoBox title="Data Inspector" icon={Database} className="min-h-[300px]">
            <DataInspector dataset={config.dataset} />
          </BentoBox>
        </div> */}

        {/* --- Bottom Grid --- */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">

          {/* Worker Status */}
          <BentoBox title="Parallel Workers" icon={Cpu} className="md:col-span-6 h-[300px]">
            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
              <table className="w-full text-sm text-left text-zinc-400">
                <thead className="text-xs text-zinc-500 uppercase bg-zinc-900/50 sticky top-0 backdrop-blur-md">
                  <tr>
                    <th className="px-4 py-3 font-medium">Rank</th>
                    <th className="px-4 py-3 font-medium">Model</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                    <th className="px-4 py-3 font-medium text-right">Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50">
                  {status.workers && status.workers.length > 0 ? (
                    status.workers.map((w) => (
                      <tr key={w.rank} className="group hover:bg-zinc-800/30 transition-colors">
                        <td className="px-4 py-3 font-mono text-xs text-zinc-500">#{w.rank}</td>
                        <td className="px-4 py-3">
                          <ModelBadge model={w.model} />
                        </td>
                        <td className="px-4 py-3">
                          <StatusBadge status={w.status} />
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs text-zinc-400">
                          {w.duration > 0 ? (
                            <span className="flex items-center justify-end gap-1">
                              <Clock size={10} />
                              {w.duration.toFixed(2)}s
                            </span>
                          ) : '-'}
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={4} className="px-4 py-8 text-center text-zinc-600 text-xs">
                        No active workers. Start training to spawn processes.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </BentoBox>

          {/* Logs */}
          <BentoBox title="System Logs" icon={Terminal} className="md:col-span-6 h-[300px]">
            <div className="flex-1 overflow-y-auto font-mono text-[11px] text-zinc-400 space-y-1.5 bg-zinc-950/50 p-4 rounded-xl border border-zinc-900 custom-scrollbar">
              {status.logs.length === 0 && (
                <div className="flex items-center gap-2 text-zinc-700">
                  <span className="animate-pulse">_</span>
                  <span>Waiting for system output...</span>
                </div>
              )}
              {status.logs.map((log, i) => (
                <div key={i} className="flex gap-3 group">
                  <span className={cn(
                    "break-all",
                    log.includes("Error") ? "text-rose-400" :
                      log.includes("Complete") ? "text-emerald-400" :
                        log.includes("Starting") ? "text-blue-400" :
                          "text-zinc-300"
                  )}>
                    {log}
                  </span>
                </div>
              ))}
            </div>
          </BentoBox>
        </div>
      </div>
    </div>
  )
}

export default App