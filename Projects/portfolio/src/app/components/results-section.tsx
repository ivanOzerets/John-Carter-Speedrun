"use client";
import React, { useState, useEffect, useRef } from "react";
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell
} from "recharts";

// ── types ──
interface HistoryPoint { epoch: number; "train/acc": number; "val/acc": number; "train/loss": number; "val/loss": number; }
interface ScratchRun { name: string; config: { learning_rate: number; num_conv_layers: number; base_filters: number; dropout: number; }; history: HistoryPoint[]; summary: Record<string, number>; }
interface TransferRun { name: string; strategy: string; model: string; history: HistoryPoint[]; summary: Record<string, number>; }
interface SklearnRun { name: string; summary: Record<string, number>; }
interface PcaData { x: number[]; y: number[]; labels: number[]; }
interface MisclassifiedItem { index: number; path: string; predicted: string; true: string; filename: string; }

// ── constants ──
const LR_VALUES = [0.0001, 0.0005, 0.001];
const DEPTH_VALUES = [4, 5, 6];
const FILTER_VALUES = [32, 64, 128];
const DROPOUT_VALUES = [0.3, 0.5, 0.7];
const STRATEGY_COLORS: Record<string, string> = { frozen: "#82aaff", gradual: "#00ffc8", finetune: "#ffaa00" };
const MODEL_SHORT: Record<string, string> = { resnet18: "ResNet18", resnet50: "ResNet50", resnet101: "ResNet101", mobilenet_v3_small: "MobileNet-S", mobilenet_v3_large: "MobileNet-L", efficientnet_b0: "EffNet-B0", efficientnet_b3: "EffNet-B3" };
const CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"];
const CLASS_SHORT: Record<string, string> = { buildings: "build.", forest: "forest", glacier: "glacier", mountain: "mount.", sea: "sea", street: "street" };
const SKLEARN_CLASSIFIERS = [{ key: "SVM", label: "SVM (RBF)" }, { key: "XGBoost", label: "XGBoost" }, { key: "LogisticRegression", label: "logistic reg." }, { key: "KNN", label: "KNN" }, { key: "RandomForest", label: "random forest" }];
const CLASS_COLORS: Record<number, string> = { 0: "#82aaff", 1: "#c3e88d", 2: "#89ddff", 3: "#c792ea", 4: "#00ffc8", 5: "#ffaa00" };

// ── slider ──
function SnapSlider({ label, values, value, onChange }: { label: string; values: number[]; value: number; onChange: (v: number) => void; }) {
    const trackRef = useRef<HTMLDivElement>(null);
    const [dragging, setDragging] = useState(false);
    const [visualPct, setVisualPct] = useState(values.indexOf(value) / (values.length - 1) * 100);

    const getPct = (clientX: number) => {
        if (!trackRef.current) return 0;
        const rect = trackRef.current.getBoundingClientRect();
        return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    };
    const snapToNearest = (pct: number) => Math.max(0, Math.min(values.length - 1, Math.round(pct * (values.length - 1))));

    useEffect(() => {
        if (!dragging) return;
        const handleMove = (e: MouseEvent) => setVisualPct(getPct(e.clientX) * 100);
        const handleUp = (e: MouseEvent) => {
            const idx = snapToNearest(getPct(e.clientX));
            onChange(values[idx]);
            setVisualPct(idx / (values.length - 1) * 100);
            setDragging(false);
        };
        window.addEventListener("mousemove", handleMove);
        window.addEventListener("mouseup", handleUp);
        return () => { window.removeEventListener("mousemove", handleMove); window.removeEventListener("mouseup", handleUp); };
    }, [dragging]);

    useEffect(() => { if (!dragging) setVisualPct(values.indexOf(value) / (values.length - 1) * 100); }, [value, dragging]);
    const snappedIdx = snapToNearest(visualPct / 100);

    return (
        <div className="mb-7">
            <span className="text-[10px] tracking-[0.08em] text-white/40 block mb-2.5">{label}</span>
            <div ref={trackRef} onMouseDown={e => { setDragging(true); setVisualPct(getPct(e.clientX) * 100); }}
                className="relative h-5 mx-3 cursor-pointer">
                <div className="absolute top-1/2 left-0 right-0 h-px bg-white/[0.08] -translate-y-1/2 pointer-events-none" />
                {values.map((_, i) => (
                    <div key={i} className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-1 h-1 rounded-full bg-white/15 pointer-events-none"
                        style={{ left: `${i / (values.length - 1) * 100}%` }} />
                ))}
                <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-accent pointer-events-none z-[1]"
                    style={{ left: `${visualPct}%` }} />
            </div>
            <div className="relative h-4 mx-3 mt-1.5">
                {values.map((v, i) => (
                    <span key={v} className="absolute -translate-x-1/2 text-[9px] whitespace-nowrap transition-colors"
                        style={{ left: `${i / (values.length - 1) * 100}%`, color: i === snappedIdx ? "#00ffc8" : "rgba(240,240,240,0.25)" }}>
                        {v}
                    </span>
                ))}
            </div>
        </div>
    );
}

// ── chart ──
function TrainingChart({ data, dataKey, label, color, yLabel }: { data: HistoryPoint[]; dataKey: keyof HistoryPoint; label: string; color: string; yLabel: string; }) {
    const formatted = data.map(d => ({ epoch: d.epoch, value: d[dataKey] }));
    return (
        <div className="bg-black/20 rounded-lg p-3 border border-white/[0.06]">
            <p className="text-[9px] tracking-[0.1em] uppercase text-white/30 mb-2">{label}</p>
            <ResponsiveContainer width="100%" height={180}>
                <LineChart data={formatted}>
                    <XAxis dataKey="epoch" tick={{ fontSize: 8, fill: "rgba(240,240,240,0.3)" }} tickLine={false} axisLine={false}
                        label={{ value: "epoch", position: "insideBottom", offset: -2, fontSize: 8, fill: "rgba(240,240,240,0.2)" }} />
                    <YAxis tick={{ fontSize: 8, fill: "rgba(240,240,240,0.3)" }} tickFormatter={(v: number) => v.toFixed(2)}
                        tickLine={false} axisLine={false} width={48} domain={["auto", "auto"]}
                        label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 8, fontSize: 8, fill: "rgba(240,240,240,0.2)" }} />
                    <Tooltip contentStyle={{ background: "#111", border: "0.5px solid rgba(255,255,255,0.1)", borderRadius: 4, fontSize: 10, padding: "4px 8px" }}
                        formatter={(value: unknown) => [(value as number).toFixed(3), yLabel]} labelFormatter={l => `epoch ${l}`}
                        labelStyle={{ color: "rgba(240,240,240,0.4)", marginBottom: 2 }} itemStyle={{ color }} />
                    <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

// ── confusion matrix ──
function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
    if (!matrix) return <div className="text-white/30 text-[11px]">No confusion matrix data</div>;
    const max = Math.max(...matrix.flat());
    return (
        <div>
            <div className="grid gap-px mt-2" style={{ gridTemplateColumns: `32px repeat(${CLASS_NAMES.length}, 1fr)` }}>
                <div />
                {CLASS_NAMES.map(c => <div key={c} className="text-[8px] text-white/40 text-center tracking-[0.05em] pb-1">{c.slice(0, 3)}</div>)}
                {CLASS_NAMES.map((row, i) => (
                    <React.Fragment key={row}>
                        <div className="text-[8px] text-white/40 flex items-center tracking-[0.05em]">{row.slice(0, 3)}</div>
                        {CLASS_NAMES.map((_, j) => {
                            const val = matrix[i]?.[j] ?? 0;
                            const intensity = val / max;
                            return (
                                <div key={j} title={`${CLASS_NAMES[i]} → ${CLASS_NAMES[j]}: ${val}`}
                                    className="aspect-square rounded-sm flex items-center justify-center text-[9px]"
                                    style={{
                                        background: i === j ? `rgba(0,255,200,${0.1 + intensity * 0.8})` : `rgba(255,100,100,${intensity * 0.6})`,
                                        color: intensity > 0.3 ? "#f0f0f0" : "rgba(240,240,240,0.4)",
                                    }}>{val}</div>
                            );
                        })}
                    </React.Fragment>
                ))}
            </div>
            <p className="text-[9px] text-white/20 mt-2 tracking-[0.05em]">teal = correct · red = misclassified</p>
        </div>
    );
}

// ── PCA plot ──
function PcaPlot({ data }: { data: PcaData }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [hoveredClass, setHoveredClass] = useState<number | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !data) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        const xs = data.x, ys = data.y;
        const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
        const pad = 20;
        const toX = (x: number) => pad + (x - minX) / (maxX - minX) * (W - pad * 2);
        const toY = (y: number) => H - pad - (y - minY) / (maxY - minY) * (H - pad * 2);
        for (let i = 0; i < xs.length; i++) {
            const label = data.labels[i];
            const isHovered = hoveredClass === null || hoveredClass === label;
            ctx.beginPath();
            ctx.arc(toX(xs[i]), toY(ys[i]), 2.5, 0, Math.PI * 2);
            ctx.fillStyle = isHovered ? CLASS_COLORS[label] : "rgba(255,255,255,0.08)";
            ctx.globalAlpha = isHovered ? 0.7 : 1;
            ctx.fill();
            ctx.globalAlpha = 1;
        }
    }, [data, hoveredClass]);

    return (
        <div>
            <canvas ref={canvasRef} width={350} height={280}
                className="w-full h-auto rounded-lg bg-black/20 border border-white/[0.06]" />
            <div className="flex flex-wrap gap-2 mt-2">
                {Object.entries(CLASS_SHORT).map(([key, name], i) => (
                    <div key={key} onMouseEnter={() => setHoveredClass(i)} onMouseLeave={() => setHoveredClass(null)}
                        className="flex items-center gap-1 cursor-pointer">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ background: CLASS_COLORS[i] }} />
                        <span className="text-[9px] tracking-[0.05em] transition-colors"
                            style={{ color: hoveredClass === i ? CLASS_COLORS[i] : "rgba(240,240,240,0.4)" }}>{name}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ── misclassified ──
function MisclassifiedGrid({ items }: { items: MisclassifiedItem[] }) {
    return (
        <div className="flex gap-2 overflow-x-auto pb-2">
            {items.map((item, i) => (
                <div key={i} className="relative shrink-0 w-[120px] sm:w-[140px]">
                    <img src={`/misclassified/${item.filename}`} alt={item.true}
                        className="w-full h-[120px] sm:h-[140px] object-cover rounded-md block" />
                    <div className="absolute bottom-0 left-0 right-0 bg-black/75 rounded-b-md px-1.5 py-1">
                        <div className="text-[8px] text-white/50">true: <span style={{ color: CLASS_COLORS[CLASS_NAMES.indexOf(item.true)] }}>{item.true}</span></div>
                        <div className="text-[8px] text-white/50">pred: <span className="text-red-400">{item.predicted}</span></div>
                    </div>
                </div>
            ))}
        </div>
    );
}

// ── main ──
export default function ResultsSection() {
    const [activeTab, setActiveTab] = useState<"scratch" | "transfer" | "sklearn">("scratch");
    const [scratchRuns, setScratchRuns] = useState<ScratchRun[]>([]);
    const [transferRuns, setTransferRuns] = useState<TransferRun[]>([]);
    const [sklearnRuns, setSklearnRuns] = useState<SklearnRun[]>([]);
    const [loading, setLoading] = useState(true);
    const [lr, setLr] = useState(0.0005);
    const [depth, setDepth] = useState(6);
    const [filters, setFilters] = useState(32);
    const [dropout, setDropout] = useState(0.7);
    const [selectedRun, setSelectedRun] = useState<string | null>(null);
    const [activeStrategy, setActiveStrategy] = useState<"frozen" | "gradual" | "finetune">("gradual");
    const [hoveredModel, setHoveredModel] = useState<string | null>(null);
    const [selectedClassifier, setSelectedClassifier] = useState("SVM");
    const [pcaData, setPcaData] = useState<PcaData | null>(null);
    const [misclassified, setMisclassified] = useState<MisclassifiedItem[]>([]);

    useEffect(() => {
        Promise.all([
            fetch("/data/scratch_runs.json").then(r => r.json()),
            fetch("/data/transfer_runs.json").then(r => r.json()),
            fetch("/data/sklearn_runs.json").then(r => r.json()),
            fetch("/data/pca_data.json").then(r => r.json()),
        ]).then(([scratch, transfer, sklearn, pca]) => {
            setScratchRuns(scratch); setTransferRuns(transfer); setSklearnRuns(sklearn); setPcaData(pca);
            setMisclassified(sklearn[0]?.summary?.svm_misclassified || []);
            setLoading(false);
        });
    }, []);

    const selectedScratchRun = scratchRuns.find(r =>
        Math.abs(r.config.learning_rate - lr) < 1e-9 && r.config.num_conv_layers === depth &&
        r.config.base_filters === filters && Math.abs(r.config.dropout - dropout) < 1e-9
    );
    const strategyRuns = transferRuns.filter(r => r.strategy === activeStrategy).sort((a, b) => (b.summary["test/acc"] || 0) - (a.summary["test/acc"] || 0));
    const displayedRun = selectedRun ? transferRuns.find(r => r.name === selectedRun) : strategyRuns[0];
    const sklearnSummary = sklearnRuns[0]?.summary || {};
    const confusionMatrix = sklearnSummary[`${selectedClassifier}/confusion_matrix`] as unknown as number[][];
    const isShowingBest = lr === 0.0005 && depth === 6 && filters === 32 && dropout === 0.7;

    return (
        <div className="glass p-4 sm:p-7 mb-4 lg:h-[38rem] lg:overflow-hidden">

            {/* Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6">
                <p className="text-[10px] tracking-[0.15em] uppercase text-white/25">Results</p>
                <div className="flex gap-1.5">
                    {(["scratch", "transfer", "sklearn"] as const).map(tab => (
                        <button key={tab} className={`tab-btn ${activeTab === tab ? "active" : ""}`} onClick={() => setActiveTab(tab)}>
                            {tab === "scratch" ? "Scratch CNN" : tab === "transfer" ? "Transfer" : "Sklearn"}
                        </button>
                    ))}
                </div>
            </div>

            {loading && <div className="text-center py-12 text-white/30 text-[11px]">loading data...</div>}

            {/* ── SCRATCH ── */}
            {!loading && activeTab === "scratch" && (
                <div className="grid grid-cols-1 lg:grid-cols-[1fr_2.5fr] gap-6">
                    <div>
                        <p className="text-[9px] tracking-[0.1em] mb-4 uppercase text-white/25">Hyperparameters</p>
                        <div className="flex items-baseline gap-2 mb-4">
                            <span className="text-[2rem] font-serif text-accent">
                                {selectedScratchRun ? ((selectedScratchRun.summary["test/acc"] || 0) * 100).toFixed(1) : "—"}%
                            </span>
                            <span className="text-[9px] text-white/30">test accuracy</span>
                            <span onClick={() => { setLr(0.0005); setDepth(6); setFilters(32); setDropout(0.7); }}
                                className={`text-[9px] ml-auto tracking-[0.05em] transition-colors ${isShowingBest ? "text-white/30 cursor-default pointer-events-none" : "text-accent cursor-pointer"}`}>
                                show best
                            </span>
                        </div>
                        <SnapSlider label="Learning rate" values={LR_VALUES} value={lr} onChange={setLr} />
                        <SnapSlider label="Conv layers" values={DEPTH_VALUES} value={depth} onChange={setDepth} />
                        <SnapSlider label="Base filters" values={FILTER_VALUES} value={filters} onChange={setFilters} />
                        <SnapSlider label="Dropout" values={DROPOUT_VALUES} value={dropout} onChange={setDropout} />
                    </div>
                    <div>
                        {selectedScratchRun && (
                            <p className="text-[9px] tracking-[0.1em] uppercase text-white/25 mb-3">
                                lr={selectedScratchRun.config.learning_rate} · depth={selectedScratchRun.config.num_conv_layers} · filters={selectedScratchRun.config.base_filters} · dropout={selectedScratchRun.config.dropout}
                            </p>
                        )}
                        {selectedScratchRun ? (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                <TrainingChart data={selectedScratchRun.history} dataKey="val/acc" label="Val accuracy" color="#00ffc8" yLabel="Accuracy" />
                                <TrainingChart data={selectedScratchRun.history} dataKey="train/acc" label="Train accuracy" color="#82aaff" yLabel="Accuracy" />
                                <TrainingChart data={selectedScratchRun.history} dataKey="val/loss" label="Val loss" color="#ffaa00" yLabel="Loss" />
                                <TrainingChart data={selectedScratchRun.history} dataKey="train/loss" label="Train loss" color="#c792ea" yLabel="Loss" />
                            </div>
                        ) : (
                            <div className="h-full flex items-center justify-center text-white/20 text-[11px]">select a hyperparameter combination</div>
                        )}
                    </div>
                </div>
            )}

            {/* ── TRANSFER ── */}
            {!loading && activeTab === "transfer" && (
                <div className="grid grid-cols-1 lg:grid-cols-[1fr_2.5fr] gap-6">
                    <div>
                        <p className="text-sm tracking-[0.1em] uppercase mb-3 -mt-1" style={{ color: STRATEGY_COLORS[activeStrategy] }}>
                            {activeStrategy === "frozen" ? "Linear probing" : activeStrategy === "gradual" ? "Staged unfreezing" : "Full fine-tuning"}
                        </p>
                        <p className="text-[9px] tracking-[0.1em] uppercase text-white/25 mb-3">Models by test acc</p>
                        {strategyRuns.map(run => {
                            const isActive = selectedRun === run.name || (!selectedRun && strategyRuns[0]?.name === run.name);
                            return (
                                <div key={run.name} onClick={() => setSelectedRun(run.name)}
                                    onMouseEnter={() => setHoveredModel(run.name)} onMouseLeave={() => setHoveredModel(null)}
                                    className="px-3 py-2 rounded-md mb-1 cursor-pointer transition-all"
                                    style={{
                                        background: isActive ? "rgba(255,255,255,0.05)" : hoveredModel === run.name ? "rgba(255,255,255,0.03)" : "transparent",
                                        border: `0.5px solid ${isActive ? STRATEGY_COLORS[activeStrategy] : hoveredModel === run.name ? "rgba(255,255,255,0.08)" : "transparent"}`,
                                    }}>
                                    <div className="flex justify-between items-center">
                                        <span className="text-[11px] text-foreground">{MODEL_SHORT[run.model] || run.model}</span>
                                        <span className="text-[11px]" style={{ color: STRATEGY_COLORS[activeStrategy] }}>
                                            {((run.summary["test/acc"] || 0) * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                        <div className="flex items-center justify-center gap-2">
                            <span onClick={() => { setActiveStrategy(prev => prev === "gradual" ? "frozen" : prev === "finetune" ? "gradual" : "frozen"); setSelectedRun(null); }}
                                className="text-2xl text-white/50 cursor-pointer">‹</span>
                            {(["frozen", "gradual", "finetune"] as const).map(s => (
                                <div key={s} onClick={() => { setActiveStrategy(s); setSelectedRun(null); }}
                                    className="rounded-full cursor-pointer transition-all mt-0.5"
                                    style={{ width: s === activeStrategy ? 8 : 5, height: s === activeStrategy ? 8 : 5, background: s === activeStrategy ? STRATEGY_COLORS[s] : "rgba(255,255,255,0.2)" }} />
                            ))}
                            <span onClick={() => { setActiveStrategy(prev => prev === "frozen" ? "gradual" : prev === "gradual" ? "finetune" : "finetune"); setSelectedRun(null); }}
                                className="text-2xl text-white/50 cursor-pointer">›</span>
                        </div>
                    </div>
                    <div>
                        {displayedRun && (
                            <>
                                <p className="text-[9px] tracking-[0.1em] uppercase text-white/25 mb-3">
                                    {MODEL_SHORT[displayedRun.model] || displayedRun.model} — {displayedRun.strategy}
                                </p>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                    <TrainingChart data={displayedRun.history} dataKey="val/acc" label="Val accuracy" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Accuracy" />
                                    <TrainingChart data={displayedRun.history} dataKey="train/acc" label="Train accuracy" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Accuracy" />
                                    <TrainingChart data={displayedRun.history} dataKey="val/loss" label="Val loss" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Loss" />
                                    <TrainingChart data={displayedRun.history} dataKey="train/loss" label="Train loss" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Loss" />
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}

            {/* ── SKLEARN ── */}
            {!loading && activeTab === "sklearn" && (
                <div className="flex flex-col gap-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8 items-start">
                        <div>
                            <p className="text-[9px] tracking-[0.1em] uppercase text-white/25 mb-3">Classifier accuracy</p>
                            <ResponsiveContainer width="100%" height={267}>
                                <BarChart data={SKLEARN_CLASSIFIERS.map(({ key, label }) => ({
                                    name: label, acc: parseFloat(((sklearnSummary[`${key}/test_acc`] || 0) * 100).toFixed(1)), selected: key === selectedClassifier,
                                }))}
                                    onClick={(data: any) => {
                                        if (data?.activePayload?.[0]) {
                                            const found = SKLEARN_CLASSIFIERS.find(c => c.label === data.activePayload[0].payload.name);
                                            if (found) setSelectedClassifier(found.key);
                                        }
                                    }}>
                                    <XAxis height={21} dataKey="name" tick={{ fontSize: 8, fill: "rgba(240,240,240,0.4)" }} tickLine={false} axisLine={false} tickMargin={8} />
                                    <YAxis width={20} domain={[88, 95]} tick={{ fontSize: 8, fill: "rgba(240,240,240,0.3)" }} tickLine={false} axisLine={false} tickFormatter={v => `${v}%`} />
                                    <Tooltip contentStyle={{ background: "#111", border: "0.5px solid rgba(255,255,255,0.1)", borderRadius: 4, fontSize: 10 }}
                                        formatter={(v: unknown) => [`${v}%`, "accuracy"]} labelFormatter={() => ""} cursor={{ fill: "rgba(255,255,255,0.03)" }} itemStyle={{ color: "#f0f0f0" }} />
                                    <Bar dataKey="acc" radius={[3, 3, 0, 0]}>
                                        {SKLEARN_CLASSIFIERS.map(({ key }) => <Cell key={key} fill={key === selectedClassifier ? "#00ffc8" : "rgba(255,255,255,0.15)"} />)}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div>
                            <p className="text-[9px] tracking-[0.1em] uppercase text-white/25 mb-3">ResNet50 feature space — PCA</p>
                            {pcaData && <PcaPlot data={pcaData} />}
                        </div>
                        <div>
                            <p className="text-[9px] tracking-[0.1em] uppercase text-white/25 mb-3">{selectedClassifier} — confusion matrix</p>
                            <div className="max-w-[260px]">
                                <ConfusionMatrix matrix={confusionMatrix} />
                            </div>
                        </div>
                    </div>
                    <div className="mt-4">
                        <p className="text-[9px] tracking-[0.1em] uppercase text-white/25 mb-3">SVM misclassified samples</p>
                        <MisclassifiedGrid items={misclassified} />
                    </div>
                </div>
            )}
        </div>
    );
}
