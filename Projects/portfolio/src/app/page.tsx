"use client";
import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import WaveGrid from "@/app/components/wave-grid";

const ORDER_CLASSES = [
  "order-3 lg:order-none",
  "order-2 lg:order-none",
  "order-1 lg:order-none",
]

const PROJECTS = [
  {
    slug: "personal-research-cli",
    title: "Personal Research CLI",
    description: "A personal AI research assistant. In goes a list of links and out pops summaries, highlights, and relevant follow-up.",
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

const COURSES = [
  {
    slug: "ml-in-production",
    title: "Machine Learning in Production",
    description: "MLOps practices for deploying and maintaining ML systems. Learning to understand production-grade pipelines.",
    tags: ["MLOps", "Python", "Deployment", "Monitoring"],
    status: "planned",
    disabled: true,
    provider: "Coursera · DeepLearning.AI",
  },
  {
    slug: "deep-learning-specialization",
    title: "Deep Learning Specialization",
    description: "Neural networks, CNNs, RNNs, and transformers. Follow-up course by Andrew Ng on modern deep learning.",
    tags: ["Python", "TensorFlow", "CNNs", "RNNs", "Transformers"],
    status: "complete",
    disabled: false,
    provider: "Coursera · DeepLearning.AI",
  },
  {
    slug: "ml-specialization",
    title: "Machine Learning Specialization",
    description: "The most recommended ML course, by Andrew Ng. Supervised, unsupervised, and reinforcement learning basics.",
    tags: ["Python", "NumPy", "Classification", "Regression", "Supervised & Unsupervised Learning"],
    status: "complete",
    disabled: false,
    provider: "Coursera · DeepLearning.AI",
  },
];

function ProjectCard({ p }: { p: typeof PROJECTS[0] }) {
  return (
    <div className="glass-card p-6 sm:p-7 h-full relative">

      {/* Status */}
      <div className="flex items-center mb-5">
        <span className={`status-dot ${p.status === "complete" ? "dot-complete" :
            p.status === "in progress" ? "dot-progress" : "dot-planned"
          }`} />
        <span className="font-mono text-[9px] tracking-[0.12em] uppercase text-white/30">
          {p.status}
        </span>
      </div>

      {/* Title */}
      <h2 className="font-serif text-lg sm:text-xl font-normal mb-3 tracking-tight text-foreground">
        {p.title}
      </h2>

      {/* Preview */}
      <div className="w-full aspect-video rounded-lg border border-white/[0.08] overflow-hidden mb-5">
        {p.slug === "ml-architecture-comparison" ? (
          <video
            src="/ml-architecture-comparison-preview-video.mp4"
            autoPlay loop muted playsInline
            className="w-full h-full object-cover block"
          />
        ) : (
          <div className="w-full h-full bg-white/[0.02] flex items-center justify-center">
            <span className="font-mono text-[9px] tracking-[0.1em] uppercase text-white/[0.18]">
              preview
            </span>
          </div>
        )}
      </div>

      {/* Description */}
      <p className="font-mono text-[11px] sm:text-xs leading-relaxed text-white/45 mb-6">
        {p.description}
      </p>

      {/* Tags */}
      <div className="flex gap-[5px] flex-wrap">
        {p.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
      </div>
    </div>
  );
}

function CourseCard({ c }: { c: typeof COURSES[0] }) {
  return (
    <div className="glass-card glass-card-course p-6 sm:p-7 h-full relative">
      {/* Status */}
      <div className="flex items-center mb-5">
        <span className={`status-dot ${c.status === "complete" ? "dot-complete-course" :
            c.status === "in progress" ? "dot-progress" : "dot-planned"
          }`} />
        <span className="font-mono text-[9px] tracking-[0.12em] uppercase text-white/30">
          {c.status}
        </span>
      </div>

      {/* Title */}
      <h2 className="font-serif text-lg sm:text-xl font-normal mb-1 tracking-tight text-foreground">
        {c.title}
      </h2>

      {/* Provider */}
      <p className="font-mono text-[9px] tracking-[0.08em] text-accent-purple/50 uppercase mb-4">
        {c.provider}
      </p>

      {/* Description */}
      <p className="font-mono text-[11px] sm:text-xs leading-relaxed text-white/45 mb-6">
        {c.description}
      </p>

      {/* Tags */}
      <div className="flex gap-[5px] flex-wrap">
        {c.tags.map(tag => <span key={tag} className="tag">{tag}</span>)}
      </div>
    </div>
  );
}

export default function Home() {
  const [copied, setCopied] = useState(false);

  const copyEmail = (e: React.MouseEvent) => {
    e.preventDefault();
    navigator.clipboard.writeText("ivan.ozerets@gmail.com");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <main className="min-h-screen text-foreground relative">
      <WaveGrid />

      <div className="relative">
        <section className="px-4 sm:px-8 lg:px-24 pt-8 pb-16 max-w-[1400px] mx-auto">

          {/* ── Identity ── */}
          <div className="flex items-center gap-6 sm:gap-8 mb-10">
            {/* Photo */}
            <div className="w-[80px] h-[80px] sm:w-[110px] sm:h-[110px] rounded-full overflow-hidden border border-white/[0.12] shrink-0">
              <Image src="/photo.JPG" alt="Ivan Ozerets" width={110} height={110} className="object-cover w-full h-full" />
            </div>
            {/* Name + role stacked on mobile, inline on desktop */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-0 sm:gap-5">
              <h1 className="font-serif text-3xl sm:text-5xl lg:text-[4.2rem] font-normal tracking-tight leading-none">
                Ivan Ozerets
              </h1>
              <div className="flex items-center gap-2 sm:gap-4 mt-1 sm:mt-0">
                <span className="hidden sm:inline text-white/[0.38] text-sm sm:text-base pt-2">|</span>
                <span className="font-mono text-[10px] sm:text-sm lg:text-base tracking-[0.08em] uppercase text-white/[0.38] sm:pt-3">
                  ML Engineer
                </span>
              </div>
            </div>
          </div>

          {/* ── Bio ── */}
          <p className="font-mono text-[12px] sm:text-[15px] leading-[1.9] text-white/45 max-w-[640px] mb-7">
            Heavily interested in anything data science, machine learning, and AI.
            Here you'll find my projects, course notes, and apps I've developed for personal development and competency in the field.
            Feel free to reach out (m-dash) I'm always looking to chat with like-minded folks.
          </p>

          {/* ── Contact ── */}
          <div className="font-mono text-[11px] sm:text-[13px] tracking-[0.04em] mb-8 sm:mb-12 flex flex-col gap-2">
            <div className="flex gap-4 items-center">
              <span className="text-white/25">email:</span>
              <a href="#" onClick={copyEmail} className="text-accent no-underline cursor-pointer">
                ivan (dot) ozerets [at] gmail (dot) com
              </a>
            </div>
            <div className="flex gap-2 items-center">
              <span className="text-white/25">github:</span>
              <a href="https://github.com/ivanOzerets" target="_blank" rel="noopener noreferrer"
                className="text-accent no-underline cursor-pointer">
                ivanOzerets
              </a>
            </div>
          </div>

          {/* ── Projects ── */}
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/25 mb-4">
            Projects
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8 sm:mb-12">
            {PROJECTS.map((p, i) => (
              p.disabled ? (
                <div key={p.slug} className={`cursor-default pointer-events-none ${ORDER_CLASSES[i]}`}>
                  <ProjectCard p={p} />
                </div>
              ) : (
                <Link key={p.slug} href={`/projects/${p.slug}`}
                  className={`no-underline text-inherit block group ${ORDER_CLASSES[i]}`}>
                  <ProjectCard p={p} />
                </Link>
              )
            ))}
          </div>

          {/* ── Courses ── */}
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/25 mb-4">
            Courses
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {COURSES.map((p, i) => {
              const reverseOrder = COURSES.length - i; // 3, 2, 1
              return p.disabled ? (
                <div key={p.slug} className={`cursor-default pointer-events-none order-${reverseOrder} lg:order-none`}>
                  <CourseCard c={p} />
                </div>
              ) : (
                <Link key={p.slug} href={`/courses/${p.slug}`}
                  className={`no-underline text-inherit block group order-${reverseOrder} lg:order-none`}>
                  <CourseCard c={p} />
                </Link>
              );
            })}
          </div>

        </section>
      </div>

      {/* ── Copied toast ── */}
      {copied && (
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 bg-accent/[0.08] border border-accent/25 text-accent font-mono text-[11px] tracking-[0.05em] px-4 py-2 rounded z-[100] backdrop-blur-lg whitespace-nowrap">
          copied to clipboard
        </div>
      )}
    </main>
  );
}
