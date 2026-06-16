"use client";
import { useEffect, useRef } from "react";

export default function WaveGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const isMobile = window.innerWidth < 768;

    const drawFrame = (t: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const size = isMobile ? 32 : 48;
      const cols = Math.ceil(canvas.width / size) + 2;
      const rows = Math.ceil(canvas.height / size) + 2;
      const T = isMobile ? t / 4000 : t / 1000;
      for (let r = -1; r < rows; r++) {
        for (let c = -1; c < cols; c++) {
          const bx = c * size;
          const by = r * size;
          const w1 = Math.sin(bx / 180 + T * 0.5) * 7;
          const w2 = Math.sin(by / 140 - T * 0.4) * 6;
          const w3 = Math.sin((bx * 0.8 + by * 0.6) / 220 + T * 0.3) * 5;
          const w4 = Math.sin((bx * 0.5 - by * 0.9) / 190 - T * 0.6) * 4;
          const w5 = Math.sin((bx * 1.1 + by * 0.3) / 160 + T * 0.2) * 3;
          const wave = w1 + w2 + w3 + w4 + w5;
          const interference = (Math.abs(w1) + Math.abs(w2)) / 13;
          const opacity = isMobile
            ? 0.01 + Math.abs(wave) * 0.001 + interference * 0.02
            : 0.04 + Math.abs(wave) * 0.004 + interference * 0.06;
          ctx.strokeStyle = `rgba(255,255,255,${Math.min(opacity, isMobile ? 0.06 : 0.2)})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.rect(bx + wave * 0.2, by + wave * 0.15, size, size);
          ctx.stroke();
        }
      }
    };

    let lastWidth = 0;

    const resize = () => {
      const newWidth = window.innerWidth;
      if (newWidth !== lastWidth) {
        lastWidth = newWidth;
        canvas.width = newWidth;
        canvas.height = window.innerHeight;
      }
    };
    resize();

    let lastFrame = 0;
    const draw = (t: number) => {
      animRef.current = requestAnimationFrame(draw);
      if (isMobile && t - lastFrame < 42) return; // ~24fps on mobile
      lastFrame = t;
      drawFrame(t);
    };
    animRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas ref={canvasRef} style={{
      position: "fixed", inset: 0, width: "100%", height: "100%",
      zIndex: 0, pointerEvents: "none",
    }} />
  );
}