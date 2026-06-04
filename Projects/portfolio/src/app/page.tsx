"use client";
import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import WaveGrid from "@/app/components/wave-grid";

const PROJECTS = [
  {
    slug: "personal-research-cli",
    title: "Personal Research CLI",
    description: "A personal AI research assistant. In goes a list of YouTube links on a subjuct. Out pops summaries, highlights, and relavant follow-up.",
    tags: ["Python", "AI", "CLI", "LLM"],
    status: "planned",
    disabled: true,
  },
  {
    slug: "re-zero-productivity-system",
    title: "Re-Zero Productivity System",
    description: "An implementation of Mark Forster's Resistance Zero system. Using psychology to build momentum and beat procrastination.",
    tags: ["React Native", "TypeScript"],
    status: "in progress",
    disabled: true,
  },
  {
    slug: "ml-architecture-comparison",
    title: "ML Architecture Comparison",
    description: "PyTorch vs TensorFlow vs Scikit-learn. Scratch CNN to fine-tuning pretrained models on Intel's scene classification dataset.",
    tags: ["Python", "PyTorch", "TensorFlow", "sklearn", "FastAPI"],
    status: "complete",
    disabled: false,
  },
];

export default function Home() {
  const [hovered, setHovered] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const copyEmail = (e: React.MouseEvent) => {
    e.preventDefault();
    navigator.clipboard.writeText("ivan.ozerets@gmail.com");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <main style={{ minHeight: "100vh", background: "#080808", color: "#f0f0f0", position: "relative" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        ::selection { background: rgba(0,255,200,0.15); }

        .accent { color: #00ffc8; }
        .muted { color: rgba(240,240,240,0.38); }

        .glass-card {
          background: rgba(255,255,255,0.03);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 0.5px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          transition: background 0.3s, border-color 0.3s;
        }
        .glass-card:hover {
          background: rgba(255,255,255,0.06);
          border-color: rgba(0,255,200,0.2);
        }

        .project-link {
          text-decoration: none;
          color: inherit;
          display: block;
        }

        .tag {
          font-size: 9px;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          padding: 2px 7px;
          border: 0.5px solid rgba(255,255,255,0.1);
          border-radius: 3px;
          color: rgba(240,240,240,0.4);
          font-family: 'DM Mono', monospace;
          transition: border-color 0.2s, color 0.2s;
        }
        .project-link:hover .tag {
          border-color: rgba(0,255,200,0.2);
          color: rgba(0,255,200,0.55);
        }

        .status-dot {
          width: 5px;
          height: 5px;
          border-radius: 50%;
          display: inline-block;
          margin-right: 5px;
        }
        .dot-complete { background: #00ffc8; }
        .dot-progress { background: #ffaa00; animation: blink 2s infinite; }
        .dot-planned { background: rgba(240,240,240,0.25); }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

        .arrow {
          font-size: 16px;
          color: rgba(240,240,240,0.18);
          transition: color 0.2s, transform 0.2s;
        }
        .project-link:hover .arrow {
          color: #00ffc8;
          transform: translate(3px, -3px);
        }
      `}</style>

      <WaveGrid />

      <div style={{ position: "relative", zIndex: 1 }}>

        {/* Hero */}
        <section style={{
          padding: "2rem clamp(1.5rem, 6vw, 6rem) 4rem",
          maxWidth: "1400px",
          margin: "0 auto",
        }}>

        {/* Identity row */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "2rem",
          marginBottom: "2.5rem",
          flexWrap: "wrap",
        }}>
          <div style={{
            width: 110,
            height: 110,
            borderRadius: "50%",
            overflow: "hidden",
            border: "0.5px solid rgba(255,255,255,0.12)",
            flexShrink: 0,
          }}>
            <Image src="/photo.jpg" alt="Ivan Ozerets" width={110} height={110}
              style={{ objectFit: "cover", width: "100%", height: "100%" }} />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "1.25rem", flexWrap: "wrap" }}>
            <h1 style={{
              fontFamily: "'Instrument Serif', serif",
              fontSize: "clamp(2.8rem, 5vw, 4.2rem)",
              fontWeight: 400,
              letterSpacing: "-0.01em",
              lineHeight: 1,
            }}>
              Ivan Ozerets
            </h1>
            <span className="muted" style={{ fontSize: "clamp(0.8rem, 2vw, 1rem)", paddingTop: "0.5rem" }}>|</span>
            <span className="muted" style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: "clamp(0.8rem, 2vw, 1rem)",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              paddingTop: "0.7rem",
            }}>
              ML Engineer
            </span>
          </div>
        </div>

        {/* Bio */}
        <p style={{
          fontFamily: "'DM Mono', monospace",
          fontSize: 15,
          lineHeight: 1.9,
          color: "rgba(240,240,240,0.45)",
          maxWidth: 640,
          marginBottom: "1.75rem",
        }}>
          Interested in anything data science, machine learning, and AI.
          Projects for personal development and competency in the field.
          Always looking to chat with like-minded folks (m-dash) feel free to reach out.
        </p>

        {/* Contact */}
        <div style={{
          fontFamily: "'DM Mono', monospace",
          fontSize: 13,
          letterSpacing: "0.04em",
          marginBottom: "5rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.5rem",
        }}>
          <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
            <span style={{ color: "rgba(240,240,240,0.25)" }}>email:</span>
            <a href="#" onClick={copyEmail}
              style={{ color: "#00ffc8", textDecoration: "none", cursor: "pointer"}}>
              ivan (dot) ozerets [at] gmail (dot) com
            </a>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <span style={{ color: "rgba(240,240,240,0.25)" }}>github:</span>
            <a href="https://github.com/ivanOzerets" target="_blank" rel="noopener noreferrer"
              style={{ color: "#00ffc8", textDecoration: "none", cursor: "pointer" }}>
              ivanOzerets
            </a>
          </div>
        </div>

          {/* Projects label */}
          <p style={{
            fontFamily: "'DM Mono', monospace",
            fontSize: 10,
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            color: "rgba(240,240,240,0.25)",
            marginBottom: "1rem",
          }}>
            Projects
          </p>

          {/* Projects grid */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "1rem",
          }}>
            {PROJECTS.map((p) => {
              const card = (
                <div className="glass-card" style={{ padding: "1.75rem", height: "100%", position: "relative" }}>
                  {/* Status */}
                  <div style={{ display: "flex", alignItems: "center", marginBottom: "1.25rem" }}>
                    <span className={`status-dot ${
                      p.status === "complete" ? "dot-complete" :
                      p.status === "in progress" ? "dot-progress" : "dot-planned"
                    }`} />
                    <span style={{
                      fontFamily: "'DM Mono', monospace",
                      fontSize: 9,
                      letterSpacing: "0.12em",
                      textTransform: "uppercase",
                      color: "rgba(240,240,240,0.3)",
                    }}>
                      {p.status}
                    </span>
                  </div>
                  {/* Title */}
                  <h2 style={{
                    fontFamily: "'Instrument Serif', serif",
                    fontSize: "1.4rem",
                    fontWeight: 400,
                    marginBottom: "0.75rem",
                    letterSpacing: "-0.01em",
                    color: "#f0f0f0",
                  }}>
                    {p.title}
                  </h2>
                  {/* Demo preview */}
                  <div style={{
                    width: "100%",
                    aspectRatio: "16/7",
                    borderRadius: 8,
                    border: "0.5px solid rgba(255,255,255,0.08)",
                    background: "rgba(255,255,255,0.02)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: "1.25rem",
                  }}>
                    <span style={{
                      fontFamily: "'DM Mono', monospace",
                      fontSize: 9,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                      color: "rgba(240,240,240,0.18)",
                    }}>
                      preview
                    </span>
                  </div>
                  {/* Description */}
                  <p style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: 12,
                    lineHeight: 1.7,
                    color: "rgba(240,240,240,0.45)",
                    marginBottom: "1.5rem",
                  }}>
                    {p.description}
                  </p>
                  {/* Tags */}
                  <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: "0.5rem" }}>
                    {p.tags.map((tag) => (
                      <span key={tag} className="tag">{tag}</span>
                    ))}
                  </div>
                </div>
              );

              return p.disabled ? (
                <div key={p.slug} style={{ cursor: "default", pointerEvents: "none" }}>
                  {card}
                </div>
              ) : (
                <Link
                  key={p.slug}
                  href={`/projects/${p.slug}`}
                  className="project-link"
                  onMouseEnter={() => setHovered(p.slug)}
                  onMouseLeave={() => setHovered(null)}
                >
                  {card}
                </Link>
              );
            })}
          </div>

        </section>
      </div>

    {copied && (
      <div style={{
        position: "fixed",
        bottom: "2rem",
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,255,200,0.08)",
        border: "0.5px solid rgba(0,255,200,0.25)",
        color: "#00ffc8",
        fontFamily: "'DM Mono', monospace",
        fontSize: 11,
        letterSpacing: "0.05em",
        padding: "0.4rem 0.9rem",
        borderRadius: 4,
        zIndex: 100,
        backdropFilter: "blur(8px)",
        whiteSpace: "nowrap",
      }}>
        copied to clipboard
      </div>
    )}
      
    </main>
  );
}
