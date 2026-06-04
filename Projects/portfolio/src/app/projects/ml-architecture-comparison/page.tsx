"use client";
import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import WaveGrid from "@/app/components/wave-grid";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import "highlight.js/styles/atom-one-dark.css";
import ResultsSection from "@/app/components/results-section";
import ReactMarkdown from "react-markdown";

hljs.registerLanguage("python", python);

const PYTORCH_CODE = `def train(model, train_loader, val_loader, criterion,
          optimizer, epochs, device, save_path,
          epoch_offset=0, patience=10):

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_correct, train_total = 0.0, 0, 0
        model.train()

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        wandb.log({
            "train/loss": train_loss / len(train_loader),
            "train/acc": train_correct / train_total,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "epoch": epoch + 1 + epoch_offset
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return epoch + 1`;

const TENSORFLOW_CODE = `def compile_and_fit(model, train_dataset, val_dataset,
                    config, save_path, lr,
                    initial_epoch=0):

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        WandbMetricsLogger(),
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_loss',
            save_best_only=True
        )
    ]

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=config.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=initial_epoch + EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    return len(history.history['loss'])`;

const CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"];

export default function MlArchitectureComparison() {
  const [framework, setFramework] = useState<"pytorch" | "tensorflow">("pytorch");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{
    class: string;
    confidence: number;
    scores: Record<string, number>;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (file: File) => {
    setImagePreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(
        "https://ivanozerets-ml-architecture-comparison.hf.space/predict",
        { method: "POST", body: formData }
      );
      const data = await res.json();
      setPrediction(data);
    } catch {
      setError("Failed to connect to API. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  const [notes, setNotes] = useState("");
  const [notesOpen, setNotesOpen] = useState(false);

  useEffect(() => {
    fetch("/notes/ml-architecture-comparison.md")
    .then(r => r.text())
    .then(setNotes);
  }, []);

  return (
    <main style={{
      minHeight: "100vh",
      background: "#080808",
      color: "#f0f0f0",
      fontFamily: "'DM Mono', monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .glass {
          background: rgba(255,255,255,0.03);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 0.5px solid rgba(255,255,255,0.08);
          border-radius: 16px;
        }

        .tab-btn {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          padding: 5px 14px;
          border: 0.5px solid rgba(255,255,255,0.1);
          border-radius: 4px;
          background: transparent;
          color: rgba(240,240,240,0.4);
          cursor: pointer;
          transition: all 0.2s;
        }
        .tab-btn.active {
          background: rgba(0,255,200,0.08);
          border-color: rgba(0,255,200,0.3);
          color: #00ffc8;
        }

        .drop-zone {
          border: 0.5px dashed rgba(255,255,255,0.15);
          border-radius: 12px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: border-color 0.2s, background 0.2s;
          gap: 8px;
        }
        .drop-zone:hover {
          border-color: rgba(0,255,200,0.3);
          background: rgba(0,255,200,0.02);
        }

        .bar-track {
          flex: 1;
          height: 4px;
          background: rgba(255,255,255,0.06);
          border-radius: 2px;
          overflow: hidden;
        }
        .bar-fill {
          height: 100%;
          border-radius: 2px;
          background: #00ffc8;
          transition: width 0.5s ease;
        }

        pre {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          line-height: 1.6;
          color: rgba(240,240,240,0.7);
          overflow-x: auto;
          white-space: pre;
        }

        .keyword { color: #c792ea; }
        .string { color: #c3e88d; }
        .comment { color: rgba(240,240,240,0.3); }
        .function { color: #82aaff; }
        .number { color: #f78c6c; }

        .notes-content h1 {
          font-size: 13px;
          color: rgba(240,240,240,0.7);
          margin-bottom: 0.5rem;
          margin-top: 1.25rem;
          font-weight: 500;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          font-family: 'DM Mono', monospace;
        }
        .notes-content h2 {
          font-size: 12px;
          color: rgba(240,240,240,0.5);
          margin-bottom: 0.4rem;
          margin-top: 1rem;
          font-weight: 500;
          letter-spacing: 0.08em;
          font-family: 'DM Mono', monospace;
        }
        .notes-content h3 {
          font-size: 11px;
          color: rgba(240,240,240,0.4);
          margin-bottom: 0.3rem;
          margin-top: 0.75rem;
          font-weight: 400;
          letter-spacing: 0.06em;
          font-family: 'DM Mono', monospace;
        }
        .notes-content p {
          margin-bottom: 0.75rem;
        }
        .notes-content ul, .notes-content ol {
          padding-left: 1rem;
          margin-bottom: 0.75rem;
          list-style-type: disc;
        }
        .notes-content li {
          margin-bottom: 0.25rem;
        }

      `}</style>

      <WaveGrid />

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "2rem clamp(1rem, 4vw, 3rem) 4rem" }}>

        {/* Back */}
        <Link href="/" style={{
          color: "rgba(240,240,240,0.3)",
          textDecoration: "none",
          fontSize: 11,
          letterSpacing: "0.08em",
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          marginBottom: "2rem",
          transition: "color 0.2s",
        }}
          onMouseEnter={e => (e.currentTarget.style.color = "#00ffc8")}
          onMouseLeave={e => (e.currentTarget.style.color = "rgba(240,240,240,0.3)")}
        >
          ← back
        </Link>

        {/* Header */}
        <div style={{ marginBottom: "2.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#00ffc8", display: "inline-block" }} />
            <span style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(240,240,240,0.3)" }}>complete</span>
          </div>
          <h1 style={{
            fontFamily: "'Instrument Serif', serif",
            fontSize: "clamp(2rem, 4vw, 3.2rem)",
            fontWeight: 400,
            letterSpacing: "-0.01em",
            marginBottom: "0.5rem",
          }}>
            ML Architecture Comparison
          </h1>
          <p style={{ fontSize: 13, color: "rgba(240,240,240,0.4)", lineHeight: 1.7, maxWidth: 600 }}>
            PyTorch vs TensorFlow vs Scikit-learn experiments. Scratch CNN to fine-tuning pretrained models on Intel's scene classification dataset.
          </p>
        </div>

        {/* Top two columns */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem", alignItems: "stretch" }}>

          {/* Left — Demo */}
          <div className="glass" style={{ padding: "1.75rem" }}>

            {/* Header row */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <p style={{ fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)", marginBottom: "1.25rem" }}>
                Live demo
              </p>
              {loading && (
                <p style={{ fontSize: 10, color: "rgba(240,240,240,0.3)", letterSpacing: "0.08em", textAlign: "center", marginBottom: "1.25rem" }}>running inference...</p>
              )}
              {error && (
                <p style={{ fontSize: 10, color: "rgba(255,100,100,0.7)", letterSpacing: "0.05em" }}>{error}</p>
              )}
            </div>

            {/* Drop zone */}
            <div
              className="drop-zone"
              style={{ height: 260, marginBottom: "1.25rem", position: "relative", overflow: "hidden" }}
              onClick={() => fileRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={e => e.preventDefault()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="preview" style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 12, background: "#111" }} />
              ) : (
                <>
                  <span style={{ fontSize: 20, color: "rgba(240,240,240,0.2)" }}>↑</span>
                  <span style={{ fontSize: 11, color: "rgba(240,240,240,0.25)", letterSpacing: "0.05em" }}>drop image or click to upload</span>
                </>
              )}
            </div>
            <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
              onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])} />

            {/* Results */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: "0.5rem" }}>
              {CLASS_NAMES.map(cls => (
                <div key={cls} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 10, width: 70, color: prediction?.class === cls ? "#f0f0f0" : "rgba(240,240,240,0.35)", letterSpacing: "0.04em" }}>
                    {cls}
                  </span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{
                      width: prediction ? `${(prediction.scores[cls] || 0) * 100}%` : "0%",
                      opacity: prediction?.class === cls ? 1 : 0.4,
                    }} />
                  </div>
                  <span style={{ fontSize: 10, width: 36, textAlign: "right", color: prediction?.class === cls ? "#00ffc8" : "rgba(240,240,240,0.3)" }}>
                    {prediction ? `${((prediction.scores[cls] || 0) * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Right — Code */}
          <div className="glass" style={{ padding: "1.75rem", display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
              <p style={{ fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)" }}>
                Training loop
              </p>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <a
                  href="https://github.com/ivanOzerets/John-Carter-Speedrun/tree/main/Projects/ml-architecture-comparison"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ fontSize: 10, color: "rgba(240,240,240,0.3)", textDecoration: "none", letterSpacing: "0.08em", transition: "color 0.2s" }}
                  onMouseEnter={e => (e.currentTarget.style.color = "#00ffc8")}
                  onMouseLeave={e => (e.currentTarget.style.color = "rgba(240,240,240,0.3)")}
                >
                  link to repo
                </a>
              </div>
            </div>

            {/* Framework tabs */}
            <div style={{ display: "flex", gap: 6, marginBottom: "1rem" }}>
              <button className={`tab-btn ${framework === "pytorch" ? "active" : ""}`}
                onClick={() => setFramework("pytorch")}>PyTorch</button>
              <button className={`tab-btn ${framework === "tensorflow" ? "active" : ""}`}
                onClick={() => setFramework("tensorflow")}>TensorFlow</button>
            </div>

            {/* Code block */}
            <div className="code-block" style={{
              background: "rgba(0,0,0,0.3)",
              borderRadius: 8,
              border: "0.5px solid rgba(255,255,255,0.06)",
              padding: "1rem",
              overflow: "auto",
              height: "23rem"
            }}>
              <pre dangerouslySetInnerHTML={{
                __html: hljs.highlight(
                  framework === "pytorch" ? PYTORCH_CODE : TENSORFLOW_CODE,
                  { language: 'python' }
                ).value
              }} />
            </div>
          </div>
        </div>

        <ResultsSection />

        {/* Personal notes */}
        <div className="glass" style={{ padding: "1.75rem" }}>
          <div
            onClick={() => setNotesOpen(!notesOpen)}
            style={{ 
              display: "flex", 
              justifyContent: "space-between", 
              alignItems: "center", 
              cursor: "pointer",
              userSelect: "none",
            }}
            onMouseEnter={e => {
              e.currentTarget.querySelector('p')!.style.color = "rgba(0,255,200,0.5)";
              e.currentTarget.querySelector('span')!.style.color = "rgba(0,255,200,0.5)";
            }}
            onMouseLeave={e => {
              e.currentTarget.querySelector('p')!.style.color = "rgba(240,240,240,0.25)";
              e.currentTarget.querySelector('span')!.style.color = "rgba(240,240,240,0.25)";
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <p style={{ fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase", color: "rgba(240,240,240,0.25)" }}>
                Notes
              </p>
              <span style={{ 
                fontSize: 14, 
                color: "rgba(240,240,240,0.25)", 
                lineHeight: 1,
                marginBottom: "4px",
              }}>
                {notesOpen ? "▴" : "▾"}
              </span>
            </div>
          </div>
          {notesOpen && (
            <div className="notes-content" style={{ fontSize: 12, color: "rgba(240,240,240,0.35)", lineHeight: 1.8, marginTop: "1rem" }}>
              <ReactMarkdown>{notes}</ReactMarkdown>
            </div>
          )}
        </div>

      </div>
    </main>
  );
}
