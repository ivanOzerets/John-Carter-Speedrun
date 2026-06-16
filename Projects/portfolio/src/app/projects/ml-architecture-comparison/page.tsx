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
      setPrediction(await res.json());
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
    fetch("/notes/ml-architecture-comparison.md").then(r => r.text()).then(setNotes);
  }, []);

  return (
    <main className="min-h-screen bg-background text-foreground font-mono">
      <WaveGrid />

      <div className="max-w-[1200px] mx-auto px-4 sm:px-8 lg:px-12 pt-8 pb-16">

        {/* Back */}
        <Link href="/" className="text-white/30 no-underline text-[11px] tracking-[0.08em] inline-flex items-center gap-1.5 mb-8 hover:text-accent transition-colors">
          ← back
        </Link>

        {/* Header */}
        <div className="mb-10">
          <div className="flex items-center gap-3 mb-2">
            <span className="w-1.5 h-1.5 rounded-full bg-accent inline-block" />
            <span className="text-[9px] tracking-[0.15em] uppercase text-white/30">complete</span>
          </div>
          <h1 className="font-serif text-2xl sm:text-3xl lg:text-[3.2rem] font-normal tracking-tight mb-2">
            ML Architecture Comparison
          </h1>
          <p className="text-[11px] sm:text-[13px] text-white/40 leading-relaxed max-w-[600px]">
            PyTorch vs TensorFlow vs Scikit-learn experiments. Scratch CNN to fine-tuning pretrained models on Intel's scene classification dataset.
          </p>
        </div>

        {/* Top grid — demo + code */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">

          {/* Demo */}
          <div className="glass p-6 sm:p-7">
            <div className="flex justify-between items-center">
              <p className="text-[10px] tracking-[0.15em] uppercase text-white/25 mb-5">Live demo</p>
              {loading && <p className="text-[10px] text-white/30 tracking-[0.08em] mb-5">running inference...</p>}
              {error && <p className="text-[10px] text-red-400/70 tracking-[0.05em]">{error}</p>}
            </div>

            <div
              className="border border-dashed border-white/15 rounded-xl flex flex-col items-center justify-center cursor-pointer transition-colors hover:border-accent/30 hover:bg-accent/[0.02] h-[220px] sm:h-[260px] mb-5 relative overflow-hidden"
              onClick={() => fileRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={e => e.preventDefault()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="preview" className="w-full h-full object-contain rounded-xl bg-[#111]" />
              ) : (
                <>
                  <span className="text-xl text-white/20">↑</span>
                  <span className="text-[11px] text-white/25 tracking-[0.05em]">drop image or click to upload</span>
                </>
              )}
            </div>
            <input ref={fileRef} type="file" accept="image/*" className="hidden"
              onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])} />

            <div className="flex flex-col gap-2 mt-2">
              {CLASS_NAMES.map(cls => (
                <div key={cls} className="flex items-center gap-2.5">
                  <span className={`text-[10px] w-[70px] tracking-[0.04em] ${prediction?.class === cls ? "text-foreground" : "text-white/35"
                    }`}>{cls}</span>
                  <div className="flex-1 h-1 bg-white/[0.06] rounded-sm overflow-hidden">
                    <div className="h-full rounded-sm bg-accent transition-all duration-500"
                      style={{
                        width: prediction ? `${(prediction.scores[cls] || 0) * 100}%` : "0%",
                        opacity: prediction?.class === cls ? 1 : 0.4,
                      }} />
                  </div>
                  <span className={`text-[10px] w-9 text-right ${prediction?.class === cls ? "text-accent" : "text-white/30"
                    }`}>
                    {prediction ? `${((prediction.scores[cls] || 0) * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Code */}
          <div className="glass p-6 sm:p-7 flex flex-col">
            <div className="flex justify-between items-center mb-5">
              <p className="text-[10px] tracking-[0.15em] uppercase text-white/25">Training loop</p>
              <a
                href="https://github.com/ivanOzerets/John-Carter-Speedrun/tree/main/Projects/ml-architecture-comparison"
                target="_blank" rel="noopener noreferrer"
                className="text-[10px] text-white/30 no-underline tracking-[0.08em] hover:text-accent transition-colors"
              >link to repo</a>
            </div>
            <div className="flex gap-1.5 mb-4">
              <button className={`tab-btn ${framework === "pytorch" ? "active" : ""}`}
                onClick={() => setFramework("pytorch")}>PyTorch</button>
              <button className={`tab-btn ${framework === "tensorflow" ? "active" : ""}`}
                onClick={() => setFramework("tensorflow")}>TensorFlow</button>
            </div>
            <div className="bg-black/30 rounded-lg border border-white/[0.06] p-4 overflow-auto h-[23rem] sm:h-[23rem]">
              <pre className="font-mono text-[10px] leading-relaxed text-white/70 whitespace-pre"
                dangerouslySetInnerHTML={{
                  __html: hljs.highlight(
                    framework === "pytorch" ? PYTORCH_CODE : TENSORFLOW_CODE,
                    { language: 'python' }
                  ).value
                }} />
            </div>
          </div>
        </div>

        <ResultsSection />

        {/* Notes */}
        <div className="glass p-6 sm:p-7">
          <div
            onClick={() => setNotesOpen(!notesOpen)}
            className="flex justify-between items-center cursor-pointer select-none group"
          >
            <div className="flex items-center gap-2">
              <p className="text-[10px] tracking-[0.15em] uppercase text-white/25 group-hover:text-accent/50 transition-colors">
                Notes
              </p>
              <span className="text-sm text-white/25 leading-none mb-1 group-hover:text-accent/50 transition-colors">
                {notesOpen ? "▴" : "▾"}
              </span>
            </div>
          </div>
          {notesOpen && (
            <div className="notes-content text-xs text-white/35 leading-[1.8] mt-4">
              <ReactMarkdown>{notes}</ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
