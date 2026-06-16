"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import WaveGrid from "@/app/components/wave-grid";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/atom-one-dark.css";
import { cleanMarkdown } from "@/app/utils/clean-markdown";

export interface Lab {
    title: string;
    file: string;
}

export interface Week {
    title: string;
    file: string | null;
    labs?: Lab[];
}

export interface Course {
    title: string;
    weeks: Week[];
}

interface CoursePageProps {
    title: string;
    provider: string;
    courses: Course[];
    basePath: string;
}

export default function CoursePage({ title, provider, courses, basePath }: CoursePageProps) {
    const [openCourses, setOpenCourses] = useState<Record<number, boolean>>({});
    const [openWeeks, setOpenWeeks] = useState<Record<string, boolean>>({});
    const [weekContent, setWeekContent] = useState<Record<string, string>>({});
    const [showTop, setShowTop] = useState(false);

    const fetchAndClean = (file: string) => {
        if (!weekContent[file]) {
            fetch(basePath + file)
                .then(r => r.text())
                .then(text => {
                    setWeekContent(prev => ({ ...prev, [file]: cleanMarkdown(text) }));
                });
        }
    };

    const toggleCourse = (ci: number) => {
        setOpenCourses(prev => ({ ...prev, [ci]: !prev[ci] }));
        courses[ci].weeks.forEach(week => {
            if (week.file) fetchAndClean(week.file);
        });
    };

    const toggleWeek = (key: string, file: string | null) => {
        setOpenWeeks(prev => ({ ...prev, [key]: !prev[key] }));
        if (file) fetchAndClean(file);
    };

    useEffect(() => {
        const onScroll = () => setShowTop(window.scrollY > 400);
        window.addEventListener("scroll", onScroll);
        return () => window.removeEventListener("scroll", onScroll);
    }, []);

    return (
        <main className="min-h-screen bg-background text-foreground font-mono">
            <WaveGrid />

            <div className="max-w-[860px] mx-auto px-4 sm:px-8 lg:px-12 pt-8 pb-24">

                {/* Back */}
                <Link href="/" className="text-white/30 no-underline text-[11px] tracking-[0.08em] inline-flex items-center gap-1.5 mb-8 hover:text-accent-purple transition-colors">
                    ← back
                </Link>

                {/* Header */}
                <div className="mb-8">
                    <p className="text-[9px] tracking-[0.15em] uppercase text-accent-purple/50 mb-2">
                        {provider}
                    </p>
                    <h1 className="font-serif text-2xl sm:text-3xl lg:text-[2.8rem] font-normal tracking-tight leading-tight">
                        {title}
                    </h1>
                </div>

                {/* Main card */}
                <div className="glass p-4 sm:p-7">
                    {courses.map((course, ci) => (
                        <div key={ci}>
                            {/* Course toggle */}
                            <div
                                className="flex items-baseline gap-2 cursor-pointer py-2 select-none group"
                                onClick={() => toggleCourse(ci)}
                            >
                                <span className="text-lg text-white/20 shrink-0">
                                    {openCourses[ci] ? "▴" : "▾"}
                                </span>
                                <span className="font-mono text-[11px] sm:text-[13px] font-medium tracking-[0.04em] text-white/70 group-hover:text-white/90 transition-colors">
                                    {course.title}
                                </span>
                            </div>

                            {/* Weeks */}
                            {openCourses[ci] && (
                                <div className="pb-2">
                                    {course.weeks.map((week, wi) => {
                                        const key = `${ci}-${wi}`;
                                        const hasNotes = !!week.file;
                                        return (
                                            <div key={key}>
                                                {/* Week toggle */}
                                                <div
                                                    className="flex items-baseline gap-2 py-1 pl-5 select-none group"
                                                    onClick={() => toggleWeek(key, week.file)}
                                                    style={{ cursor: hasNotes ? "pointer" : "default" }}
                                                >
                                                    {hasNotes && (
                                                        <span className="text-lg text-white/20 shrink-0">
                                                            {openWeeks[key] ? "▴" : "▾"}
                                                        </span>
                                                    )}
                                                    <span className={`font-mono text-[11px] tracking-[0.03em] transition-colors ${hasNotes
                                                            ? "text-white/40 group-hover:text-white/70"
                                                            : "text-white/[0.18] italic"
                                                        }`}>
                                                        {week.title}{!hasNotes ? " — coming soon" : ""}
                                                    </span>
                                                </div>

                                                {/* Week content */}
                                                {openWeeks[key] && week.file && (
                                                    weekContent[week.file] ? (
                                                        <div className="notes-content pl-6 sm:pl-10 pt-3 pb-2">
                                                            <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                                                                {weekContent[week.file]}
                                                            </ReactMarkdown>
                                                        </div>
                                                    ) : (
                                                        <div className="pl-10 py-2 text-[11px] text-white/20 font-mono">
                                                            loading...
                                                        </div>
                                                    )
                                                )}

                                                {/* Lab toggles */}
                                                {openWeeks[key] && week.labs?.map((lab, li) => {
                                                    const labKey = `${key}-lab-${li}`;
                                                    return (
                                                        <div key={labKey}>
                                                            <div
                                                                className="flex items-baseline gap-2 py-1 pl-10 cursor-pointer select-none group"
                                                                onClick={() => toggleWeek(labKey, lab.file)}
                                                            >
                                                                <span className="text-lg text-white/20 shrink-0">
                                                                    {openWeeks[labKey] ? "▴" : "▾"}
                                                                </span>
                                                                <span className="font-mono text-[10px] tracking-[0.03em] text-white/40 group-hover:text-white/70 transition-colors">
                                                                    {lab.title}
                                                                </span>
                                                            </div>
                                                            {openWeeks[labKey] && weekContent[lab.file] && (
                                                                <div className="notes-content pl-10 sm:pl-16 pt-3 pb-2">
                                                                    <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                                                                        {weekContent[lab.file]}
                                                                    </ReactMarkdown>
                                                                </div>
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Back to top */}
            {showTop && (
                <button
                    onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
                    className="fixed bottom-8 left-1/2 -translate-x-1/2 bg-white/[0.04] border border-accent-purple/30 text-accent-purple font-mono text-[10px] tracking-[0.1em] uppercase px-4 py-1.5 rounded cursor-pointer backdrop-blur-lg z-[100] hover:border-accent-purple/60 transition-colors"
                >
                    ↑ top
                </button>
            )}
        </main>
    );
}
