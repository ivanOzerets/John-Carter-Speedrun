export function cleanMarkdown(text: string): string {
    let processed = text.replace(/\t/g, "  ");
    let inFence = false;
    let fenceIndent = 0;
    const lines = processed.split('\n').map(line => {
        if (line.trim().startsWith('```')) {
            if (!inFence) {
                const leading = line.match(/^(\s*)/);
                fenceIndent = leading ? leading[1].length : 0;
                inFence = true;
            } else {
                inFence = false;
                fenceIndent = 0;
            }
            return line.trimStart();
        }
        if (inFence) {
            if (fenceIndent > 0 && line.startsWith(' '.repeat(fenceIndent))) {
                return line.slice(fenceIndent);
            }
            return line;
        }
        const leading = line.match(/^(\s+)/);
        if (leading && leading[1].length > 3) {
            return '   ' + line.trimStart();
        }
        return line;
    });
    return lines.join('\n');
}