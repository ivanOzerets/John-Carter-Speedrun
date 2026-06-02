"use client";
import React, { useState, useEffect, useRef } from "react";
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
    BarChart,
    Bar,
    Cell
} from "recharts";

// ── types ──────────────────────────────────────────────────────────────────
interface HistoryPoint {
    epoch: number;
    "train/acc": number;
    "val/acc": number;
    "train/loss": number;
    "val/loss": number;
}

interface ScratchRun {
    name: string;
    config: {
        learning_rate: number;
        num_conv_layers: number;
        base_filters: number;
        dropout: number;
    };
    history: HistoryPoint[];
    summary: Record<string, number>;
}

interface TransferRun {
    name: string;
    strategy: string;
    model: string;
    history: HistoryPoint[];
    summary: Record<string, number>;
}

interface SklearnRun {
    name: string;
    summary: Record<string, number>;
}

interface PcaData {
    x: number[];
    y: number[];
    labels: number[];
}

interface MisclassifiedItem {
    index: number;
    path: string;
    predicted: string;
    true: string;
    filename: string;
}

// ── constants ──────────────────────────────────────────────────────────────
const LR_VALUES = [0.0001, 0.0005, 0.001];
const DEPTH_VALUES = [4, 5, 6];
const FILTER_VALUES = [32, 64, 128];
const DROPOUT_VALUES = [0.3, 0.5, 0.7];

const STRATEGY_COLORS: Record<string, string> = {
    frozen: "#82aaff",
    gradual: "#00ffc8",
    finetune: "#ffaa00",
};

const MODEL_SHORT: Record<string, string> = {
    resnet18: "ResNet18",
    resnet50: "ResNet50",
    resnet101: "ResNet101",
    mobilenet_v3_small: "MobileNet-S",
    mobilenet_v3_large: "MobileNet-L",
    efficientnet_b0: "EffNet-B0",
    efficientnet_b3: "EffNet-B3",
};

const CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"];
const CLASS_SHORT: Record<string, string> = {
    buildings: "build.", forest: "forest", glacier: "glacier",
    mountain: "mount.", sea: "sea", street: "street"
};

const SKLEARN_CLASSIFIERS = [
    { key: "SVM", label: "SVM (RBF)" },
    { key: "XGBoost", label: "XGBoost" },
    { key: "LogisticRegression", label: "logistic reg." },
    { key: "KNN", label: "KNN" },
    { key: "RandomForest", label: "random forest" },
];

const CLASS_COLORS: Record<number, string> = {
    0: "#82aaff",   // buildings
    1: "#c3e88d",   // forest
    2: "#89ddff",   // glacier
    3: "#c792ea",   // mountain
    4: "#00ffc8",   // sea
    5: "#ffaa00",   // street
}

// ── slider component ───────────────────────────────────────────────────────
function SnapSlider({ label, values, value, onChange }: {
    label: string;
    values: number[];
    value: number;
    onChange: (v: number) => void;
}) {
    const trackRef = useRef<HTMLDivElement>(null);
    const [dragging, setDragging] = useState(false);
    const [visualPct, setVisualPct] = useState(values.indexOf(value) / (values.length - 1) * 100);

    const getPct = (clientX: number) => {
        if (!trackRef.current) return 0;
        const rect = trackRef.current.getBoundingClientRect();
        return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    };

    const snapToNearest = (pct: number) =>
        Math.max(0, Math.min(values.length - 1, Math.round(pct * (values.length - 1))));

    useEffect(() => {
        if (!dragging) return;
        const handleMove = (e: MouseEvent) => {
            const pct = getPct(e.clientX);
            setVisualPct(pct * 100);
        };
        const handleUp = (e: MouseEvent) => {
            const idx = snapToNearest(getPct(e.clientX));
            onChange(values[idx]);
            setVisualPct(idx / (values.length - 1) * 100);
            setDragging(false);
        };
        window.addEventListener("mousemove", handleMove);
        window.addEventListener("mouseup", handleUp);
        return () => {
            window.removeEventListener("mousemove", handleMove);
            window.removeEventListener("mouseup", handleUp);
        };
    }, [dragging]);

    useEffect(() => {
        if (!dragging) {
            setVisualPct(values.indexOf(value) / (values.length - 1) * 100);
        }
    }, [value, dragging]);

    const snappedIdx = snapToNearest(visualPct / 100);

    return (
        <div style={{ marginBottom: "1.75rem" }}>
            <span style={{ fontSize: 10, letterSpacing: "0.08em", color: "rgba(240,240,240,0.4)", display: "block", marginBottom: 10 }}>
                {label}
            </span>
            {/* track area */}
            <div
                ref={trackRef}
                onMouseDown={e => { setDragging(true); setVisualPct(getPct(e.clientX) * 100); }}
                style={{ position: "relative", height: 20, margin: "0 12px", cursor: "pointer" }}
            >
                {/* line */}
                <div style={{
                    position: "absolute", top: "50%", left: 0, right: 0,
                    height: 1, background: "rgba(255,255,255,0.08)",
                    transform: "translateY(-50%)", pointerEvents: "none"
                }} />
                {/* snap marker dots */}
                {values.map((_, i) => (
                    <div key={i} style={{
                        position: "absolute",
                        left: `${i / (values.length - 1) * 100}%`,
                        top: "50%",
                        transform: "translate(-50%, -50%)",
                        width: 4, height: 4,
                        borderRadius: "50%",
                        background: "rgba(255,255,255,0.15)",
                        pointerEvents: "none",
                    }} />
                ))}
                {/* active dot */}
                <div style={{
                    position: "absolute",
                    left: `${visualPct}%`,
                    top: "50%",
                    transform: "translate(-50%, -50%)",
                    width: 10, height: 10,
                    borderRadius: "50%",
                    background: "#00ffc8",
                    pointerEvents: "none",
                    zIndex: 1,
                }} />
            </div>
            {/* labels — separate div below track */}
            <div style={{ position: "relative", height: 16, margin: "6px 12px 0" }}>
                {values.map((v, i) => (
                    <span key={v} style={{
                        position: "absolute",
                        left: `${i / (values.length - 1) * 100}%`,
                        transform: "translateX(-50%)",
                        fontSize: 9,
                        color: i === snappedIdx ? "#00ffc8" : "rgba(240,240,240,0.25)",
                        transition: "color 0.15s",
                        whiteSpace: "nowrap",
                    }}>
                        {v}
                    </span>
                ))}
            </div>
        </div>
    );
}

// ── chart component ────────────────────────────────────────────────────────
function TrainingChart({
    data, dataKey, label, color, yLabel
}: {
    data: HistoryPoint[];
    dataKey: keyof HistoryPoint;
    label: string;
    color: string;
    yLabel: string;
}) {
    const formatted = data.map(d => ({ epoch: d.epoch, value: d[dataKey] }));
    return (
        <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "0.75rem", border: "0.5px solid rgba(255,255,255,0.06)" }}>
            <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.3)", marginBottom: 8 }}>{label}</p>
            <ResponsiveContainer width="100%" height={180}>
                <LineChart data={formatted}>
                    <XAxis dataKey="epoch"
                        tick={{ fontSize: 8, fill: "rgba(240,240,240,0.3)" }}
                        tickLine={false} axisLine={false}
                        label={{ value: "epoch", position: "insideBottom", offset: -2, fontSize: 8, fill: "rgba(240,240,240,0.2)" }}
                    />
                    <YAxis
                        tick={{ fontSize: 8, fill: "rgba(240,240,240,0.3)" }}
                        tickFormatter={(v: number) => v.toFixed(2)}
                        tickLine={false} axisLine={false} width={48}
                        domain={["auto", "auto"]}
                        label={{ value: yLabel, angle: -90, position: "insideLeft", offset: 8, fontSize: 8, fill: "rgba(240,240,240,0.2)" }}
                    />
                    <Tooltip
                        contentStyle={{
                            background: "#111",
                            border: "0.5px solid rgba(255,255,255,0.1)",
                            borderRadius: 4,
                            fontSize: 10,
                            padding: "4px 8px"
                        }}
                        formatter={(value: unknown) => [(value as number).toFixed(3), yLabel]}
                        labelFormatter={(label) => `epoch ${label}`}
                        labelStyle={{ color: "rgba(240,240,240,0.4)", marginBottom: 2 }}
                        itemStyle={{ color }}
                    />
                    <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

// ── confusion matrix ───────────────────────────────────────────────────────
function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
    if (!matrix) return <div style={{ color: "rgba(240,240,240,0.3)", fontSize: 11 }}>No confusion matrix data</div>;
    const max = Math.max(...matrix.flat());
    return (
        <div>
            <div style={{ display: "grid", gridTemplateColumns: `32px repeat(${CLASS_NAMES.length}, 1fr)`, gap: 1, marginTop: 8 }}>
                <div />
                {CLASS_NAMES.map(c => (
                    <div key={c} style={{ fontSize: 8, color: "rgba(240,240,240,0.4)", textAlign: "center", letterSpacing: "0.05em", paddingBottom: 4 }}>
                        {c.slice(0, 3)}
                    </div>
                ))}
                {CLASS_NAMES.map((row, i) => (
                    <React.Fragment key={row}>
                        <div key={row} style={{ fontSize: 8, color: "rgba(240,240,240,0.4)", display: "flex", alignItems: "center", letterSpacing: "0.05em" }}>
                            {row.slice(0, 3)}
                        </div>
                        {CLASS_NAMES.map((_, j) => {
                            const val = matrix[i]?.[j] ?? 0;
                            const intensity = val / max;
                            const isCorrect = i === j;
                            return (
                                <div
                                    key={j}
                                    title={`${CLASS_NAMES[i]} → ${CLASS_NAMES[j]}: ${val}`}
                                    style={{
                                        aspectRatio: "1",
                                        borderRadius: 3,
                                        background: isCorrect
                                            ? `rgba(0,255,200,${0.1 + intensity * 0.8})`
                                            : `rgba(255,100,100,${intensity * 0.6})`,
                                        display: "flex", alignItems: "center", justifyContent: "center",
                                        fontSize: 9, color: intensity > 0.3 ? "#f0f0f0" : "rgba(240,240,240,0.4)",
                                    }}
                                >
                                    {val}
                                </div>
                            );
                        })}
                    </React.Fragment>
                ))}
            </div>
            <p style={{ fontSize: 9, color: "rgba(240,240,240,0.2)", marginTop: 8, letterSpacing: "0.05em" }}>
                teal = correct · red = misclassified
            </p>
        </div>
    );
}

// ── PCA plot ───────────────────────────────────────────────────────────────
function PcaPlot({ data }: { data: PcaData }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [hoveredClass, setHoveredClass] = useState<number | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !data) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const xs = data.x;
        const ys = data.y;
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);
        const pad = 20;

        const toCanvasX = (x: number) => pad + (x - minX) / (maxX - minX) * (W - pad * 2);
        const toCanvasY = (y: number) => H - pad - (y - minY) / (maxY - minY) * (H - pad * 2);

        for (let i = 0; i < xs.length; i++) {
            const label = data.labels[i];
            const isHovered = hoveredClass === null || hoveredClass === label;
            ctx.beginPath();
            ctx.arc(toCanvasX(xs[i]), toCanvasY(ys[i]), 2.5, 0, Math.PI * 2);
            ctx.fillStyle = isHovered ? CLASS_COLORS[label] : "rgba(255,255,255,0.08)";
            ctx.globalAlpha = isHovered ? 0.7 : 1;
            ctx.fill();
            ctx.globalAlpha = 1;
        }
    }, [data, hoveredClass]);

    return (
        <div>
            <canvas ref={canvasRef} width={350} height={280}
                style={{ width: "100%", height: "auto", borderRadius: 8, background: "rgba(0,0,0,0.2)", border: "0.5px solid rgba(255,255,255,0.06)" }}
            />
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: 8 }}>
                {Object.entries(CLASS_SHORT).map(([key, name], i) => (
                    <div key={key} onMouseEnter={() => setHoveredClass(i)} onMouseLeave={() => setHoveredClass(null)}
                        style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: CLASS_COLORS[i] }} />
                        <span style={{ fontSize: 9, color: hoveredClass === i ? CLASS_COLORS[i] : "rgba(240,240,240,0.4)", letterSpacing: "0.05em" }}>
                            {name}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ── missclassified images list ─────────────────────────────────────────────
function MisclassifiedGrid({ items }: { items: MisclassifiedItem[] }) {
    return (
        <div style={{
            display: "flex",
            gap: "0.5rem",
            overflowX: "auto",
            paddingBottom: "0.5rem",
        }}>
            {items.map((item, i) => (
                <div key={i} style={{ position: "relative", flexShrink: 0, width: 140 }}>
                    <img
                        src={`/misclassified/${item.filename}`}
                        alt={item.true}
                        style={{ width: 140, height: 140, objectFit: "cover", borderRadius: 6, display: "block" }}
                    />
                    <div style={{
                        position: "absolute", bottom: 0, left: 0, right: 0,
                        background: "rgba(0,0,0,0.75)",
                        borderRadius: "0 0 6px 6px",
                        padding: "3px 5px",
                    }}>
                        <div style={{ fontSize: 8, color: "rgba(240,240,240,0.5)" }}>
                            true: <span style={{ color: CLASS_COLORS[CLASS_NAMES.indexOf(item.true)] }}>{item.true}</span>
                        </div>
                        <div style={{ fontSize: 8, color: "rgba(240,240,240,0.5)" }}>
                            pred: <span style={{ color: "#ff6464" }}>{item.predicted}</span>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}

// ── main results component ─────────────────────────────────────────────────
export default function ResultsSection() {
    const [activeTab, setActiveTab] = useState<"scratch" | "transfer" | "sklearn">("scratch");
    const [scratchRuns, setScratchRuns] = useState<ScratchRun[]>([]);
    const [transferRuns, setTransferRuns] = useState<TransferRun[]>([]);
    const [sklearnRuns, setSklearnRuns] = useState<SklearnRun[]>([]);
    const [loading, setLoading] = useState(true);

    // scratch CNN slider state
    const [lr, setLr] = useState(0.0005);
    const [depth, setDepth] = useState(6);
    const [filters, setFilters] = useState(32);
    const [dropout, setDropout] = useState(0.7);

    // transfer state
    const [selectedRun, setSelectedRun] = useState<string | null>(null);
    const [activeStrategy, setActiveStrategy] = useState<"frozen" | "gradual" | "finetune">("gradual");
    const [hoveredModel, setHoveredModel] = useState<string | null>(null);

    // sklearn state
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
            setScratchRuns(scratch);
            setTransferRuns(transfer);
            setSklearnRuns(sklearn);
            setPcaData(pca);
            setMisclassified(sklearn[0]?.summary?.svm_misclassified || []);
            setLoading(false);
        });
    }, []);

    const selectedScratchRun = scratchRuns.find(r =>
        Math.abs(r.config.learning_rate - lr) < 1e-9 &&
        r.config.num_conv_layers === depth &&
        r.config.base_filters === filters &&
        Math.abs(r.config.dropout - dropout) < 1e-9
    );

    const strategyRuns = transferRuns
        .filter(r => r.strategy === activeStrategy)
        .sort((a, b) => (b.summary["test/acc"] || 0) - (a.summary["test/acc"] || 0));

    const displayedRun = selectedRun
        ? transferRuns.find(r => r.name === selectedRun)
        : strategyRuns[0];

    const sklearnSummary = sklearnRuns[0]?.summary || {};
    const cmKey = `${selectedClassifier}/confusion_matrix`;
    const confusionMatrix = sklearnSummary[cmKey] as unknown as number[][];
    const isShowingBest = lr === 0.0005 && depth === 6 && filters === 32 && dropout === 0.7;

    return (
        <div className="glass" style={{ padding: "1.75rem", marginBottom: "1rem", height: "38rem", overflow: "hidden" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem" }}>
                <p style={{ fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)" }}>
                    Results
                </p>
                <div style={{ display: "flex", gap: 6 }}>
                    {(["scratch", "transfer", "sklearn"] as const).map(tab => (
                        <button
                            key={tab}
                            className={`tab-btn ${activeTab === tab ? "active" : ""}`}
                            onClick={() => setActiveTab(tab)}
                        >
                            {tab === "scratch" ? "Scratch CNN" : tab === "transfer" ? "Transfer" : "Sklearn"}
                        </button>
                    ))}
                </div>
            </div>

            {loading && (
                <div style={{ textAlign: "center", padding: "3rem", color: "rgba(240,240,240,0.3)", fontSize: 11 }}>
                    loading data...
                </div>
            )}

            {/* ── SCRATCH CNN ── */}
            {!loading && activeTab === "scratch" && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 2.5fr", gap: "1.5rem" }}>
                    <div>
                        <p style={{ fontSize: 9, letterSpacing: "0.1em", marginBottom: "1rem", textTransform: "uppercase", color: "rgba(240,240,240,0.25)" }}>
                            Hyperparameters
                        </p>

                        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginBottom: "1rem" }}>
                            <span style={{ fontSize: "2rem", fontFamily: "'Instrument Serif', serif", color: "#00ffc8" }}>
                                {selectedScratchRun ? ((selectedScratchRun.summary["test/acc"] || 0) * 100).toFixed(1) : "—"}%
                            </span>
                            <span style={{ fontSize: 9, color: "rgba(240,240,240,0.3)" }}>test accuracy</span>
                            <span
                                onClick={() => { setLr(0.0005); setDepth(6); setFilters(32); setDropout(0.7); }}
                                style={{
                                    fontSize: 9, 
                                    color: isShowingBest ? "rgba(240,240,240,0.3)" : "#00ffc8", 
                                    cursor: isShowingBest ? "default" : "pointer",
                                    marginLeft: "auto", 
                                    transition: "color 0.2s",
                                    letterSpacing: "0.05em",
                                    pointerEvents: isShowingBest ? "none" : "auto"
                                }}
                            >
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
                            <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "0.75rem" }}>
                                lr={selectedScratchRun.config.learning_rate} · depth={selectedScratchRun.config.num_conv_layers} · filters={selectedScratchRun.config.base_filters} · dropout={selectedScratchRun.config.dropout}
                            </p>
                        )}
                        {selectedScratchRun ? (
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                                <TrainingChart data={selectedScratchRun.history} dataKey="val/acc" label="Val accuracy" color="#00ffc8" yLabel="Accuracy" />
                                <TrainingChart data={selectedScratchRun.history} dataKey="train/acc" label="Train accuracy" color="#82aaff" yLabel="Accuracy" />
                                <TrainingChart data={selectedScratchRun.history} dataKey="val/loss" label="Val loss" color="#ffaa00" yLabel="Loss" />
                                <TrainingChart data={selectedScratchRun.history} dataKey="train/loss" label="Train loss" color="#c792ea" yLabel="Loss" />
                            </div>
                        ) : (
                            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "rgba(240,240,240,0.2)", fontSize: 11 }}>
                                select a hyperparameter combination
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ── TRANSFER ── */}
            {!loading && activeTab === "transfer" && (
                <div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 2.5fr", gap: "1.5rem" }}>
                        <div>
                            {/* Active strategy title */}
                            <p style={{
                                fontSize: 14, letterSpacing: "0.1em", textTransform: "uppercase",
                                color: STRATEGY_COLORS[activeStrategy], marginBottom: "0.75rem",
                                marginTop: "-0.25rem"
                            }}>
                                {activeStrategy === "frozen" ? "Linear probing" : activeStrategy === "gradual" ? "Staged unfreezing" : "Full fine-tuning"}
                            </p>

                            {/* Model list */}
                            <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "0.75rem" }}>
                                Models by test acc
                            </p>
                            {strategyRuns.map(run => {
                                const isActive = selectedRun === run.name || (!selectedRun && strategyRuns[0]?.name === run.name);
                                return (
                                    <div
                                        key={run.name}
                                        onClick={() => setSelectedRun(run.name)}
                                        onMouseEnter={() => setHoveredModel(run.name)}
                                        onMouseLeave={() => setHoveredModel(null)}
                                        style={{
                                            padding: "0.5rem 0.75rem",
                                            borderRadius: 6,
                                            marginBottom: 4,
                                            cursor: "pointer",
                                            background: isActive
                                                ? "rgba(255,255,255,0.05)"
                                                : hoveredModel === run.name
                                                    ? "rgba(255,255,255,0.03)"
                                                    : "transparent",
                                            border: `0.5px solid ${isActive
                                                    ? STRATEGY_COLORS[activeStrategy]
                                                    : hoveredModel === run.name
                                                        ? "rgba(255,255,255,0.08)"
                                                        : "transparent"
                                                }`,
                                            transition: "all 0.15s",
                                        }}
                                    >
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                            <span style={{ fontSize: 11, color: "#f0f0f0" }}>{MODEL_SHORT[run.model] || run.model}</span>
                                            <span style={{ fontSize: 11, color: STRATEGY_COLORS[activeStrategy] }}>
                                                {((run.summary["test/acc"] || 0) * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}

                            {/* Pagination dots */}
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                                <span
                                    onClick={() => { setActiveStrategy(prev => prev === "gradual" ? "frozen" : prev === "finetune" ? "gradual" : "frozen"); setSelectedRun(null); }}
                                    style={{ fontSize: 24, color: "rgba(240,240,240,0.5)", cursor: "pointer" }}
                                >‹</span>
                                {(["frozen", "gradual", "finetune"] as const).map(s => (
                                    <div
                                        key={s}
                                        onClick={() => { setActiveStrategy(s); setSelectedRun(null); }}
                                        style={{
                                            width: s === activeStrategy ? 8 : 5,
                                            height: s === activeStrategy ? 8 : 5,
                                            borderRadius: "50%",
                                            background: s === activeStrategy ? STRATEGY_COLORS[s] : "rgba(255,255,255,0.2)",
                                            cursor: "pointer",
                                            transition: "all 0.2s",
                                            marginTop: 2,
                                        }}
                                    />
                                ))}
                                <span
                                    onClick={() => { setActiveStrategy(prev => prev === "frozen" ? "gradual" : prev === "gradual" ? "finetune" : "finetune"); setSelectedRun(null); }}
                                    style={{ fontSize: 24, color: "rgba(240,240,240,0.5)", cursor: "pointer" }}
                                >›</span>
                            </div>
                        </div>

                        {/* Right — charts */}
                        <div>
                            {displayedRun && (
                                <>
                                    <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "0.75rem" }}>
                                        {MODEL_SHORT[displayedRun.model] || displayedRun.model} — {displayedRun.strategy}
                                    </p>
                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                                        <TrainingChart data={displayedRun.history} dataKey="val/acc" label="Val accuracy" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Accuracy" />
                                        <TrainingChart data={displayedRun.history} dataKey="train/acc" label="Train accuracy" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Accuracy" />
                                        <TrainingChart data={displayedRun.history} dataKey="val/loss" label="Val loss" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Loss" />
                                        <TrainingChart data={displayedRun.history} dataKey="train/loss" label="Train loss" color={STRATEGY_COLORS[displayedRun.strategy]} yLabel="Loss" />
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* ── SKLEARN ── */}
            {!loading && activeTab === "sklearn" && (
                <div style={{ display: "flex", gridTemplateColumns: "1fr 1fr", flexDirection: "column", gap: "1rem" }}>

                    {/* Top row — classifier bars + PCA */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "2rem", alignItems: "start" }}>
                        <div style={{ marginLeft: "1.5rem" }}>
                            <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "0.75rem" }}>
                                Classifier accuracy
                            </p>
                            <ResponsiveContainer width="100%" height={267}>
                                <BarChart
                                    data={SKLEARN_CLASSIFIERS.map(({ key, label }) => ({
                                        name: label,
                                        acc: parseFloat(((sklearnSummary[`${key}/test_acc`] || 0) * 100).toFixed(1)),
                                        selected: key === selectedClassifier,
                                    }))}
                                    onClick={(data: any) => {
                                        if (data?.activePayload?.[0]) {
                                            const name = data.activePayload[0].payload.name;
                                            const found = SKLEARN_CLASSIFIERS.find(c => c.label === name);
                                            if (found) setSelectedClassifier(found.key);
                                        }
                                    }}
                                >
                                    <XAxis height={21} dataKey="name" tick={{ fontSize: 8, fill: "rgba(240,240,240,0.4)" }} tickLine={false} axisLine={false} tickMargin={8} />
                                    <YAxis width={20} domain={[88, 95]} tick={{ fontSize: 8, fill: "rgba(240,240,240,0.3)" }} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
                                    <Tooltip
                                        contentStyle={{ background: "#111", border: "0.5px solid rgba(255,255,255,0.1)", borderRadius: 4, fontSize: 10 }}
                                        formatter={(v: unknown) => [`${v}%`, "accuracy"]}
                                        labelFormatter={() => ""}
                                        cursor={{ fill: "rgba(255,255,255,0.03)" }}
                                        itemStyle={{ color: "#f0f0f0" }}
                                    />
                                    <Bar dataKey="acc" radius={[3, 3, 0, 0]}>
                                        {SKLEARN_CLASSIFIERS.map(({ key }) => (
                                            <Cell key={key} fill={key === selectedClassifier ? "#00ffc8" : "rgba(255,255,255,0.15)"} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div style={{ marginLeft: "1.5rem" }}>
                            <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "0.75rem" }}>
                                ResNet50 feature space — PCA
                            </p>
                            {pcaData && <PcaPlot data={pcaData} />}
                        </div>

                        <div style={{ marginLeft: "2rem" }}>
                            <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "0.75rem", marginLeft: "-0.25rem" }}>
                                {selectedClassifier} — confusion matrix
                            </p>
                            <div style={{ maxWidth: 260 }}>
                                <ConfusionMatrix matrix={confusionMatrix} />
                            </div>
                        </div>
                    </div>

                    {/* Confusion matrix + misclassified */}
                    <div style={{ marginTop: "1rem" }}>
                        <p style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "0.75rem" }}>
                            SVM misclassified samples
                        </p>
                        <MisclassifiedGrid items={misclassified} />
                    </div>
                </div>
            )}
        </div>
    );
}
