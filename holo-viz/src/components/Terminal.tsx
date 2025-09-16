import { useEffect, useRef, useState } from "react";
import { Terminal as XTerm } from "xterm";
import { FitAddon } from "xterm-addon-fit";
import "xterm/css/xterm.css";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Square, RotateCcw } from "lucide-react";
import { getStatus, reportHtmlUrl, fetchLogs } from "@/lib/api";

type Props = {
  jobId: string | null;
  onOpenDashboard?: () => void;
};

export const Terminal = ({ jobId, onOpenDashboard }: Props) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<XTerm | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<string>("idle");
  const lastPrintedStatusRef = useRef<string>("");
  const logTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (terminalRef.current && !xtermRef.current) {
      const terminal = new XTerm({
        theme: {
          background: "transparent",
          foreground: "#00ff88",
          cursor: "#00ff88",
          black: "#1a1a2e",
          red: "#ff6b6b",
          green: "#00ff88",
          yellow: "#feca57",
          blue: "#48dbfb",
          magenta: "#ff9ff3",
          cyan: "#54a0ff",
          white: "#f1f2f6",
        },
        fontSize: 14,
        fontFamily: "Monaco, Menlo, 'Ubuntu Mono', monospace",
        cursorBlink: true,
        allowTransparency: true,
      });

      const fitAddon = new FitAddon();
      terminal.loadAddon(fitAddon);
      
      terminal.open(terminalRef.current);
      fitAddon.fit();

      // Welcome message
      terminal.writeln("🏆 Hackathon Evaluation Terminal v2.0");
      terminal.writeln("════════════════════════════════════════");
      terminal.writeln("Ready to run 17-step evaluation pipeline...");
      terminal.writeln("Upload model.h5 and dataset folder to begin");
      terminal.write("\r\n$ ");

      // Store references
      xtermRef.current = terminal;
      fitAddonRef.current = fitAddon;

      // Handle resize
      const handleResize = () => {
        if (fitAddon) {
          fitAddon.fit();
        }
      };

      window.addEventListener("resize", handleResize);
      return () => {
        window.removeEventListener("resize", handleResize);
        terminal.dispose();
      };
    }
  }, []);

  const runEvaluation = () => {
    const terminal = xtermRef.current;
    if (!terminal || isRunning) return;

    setIsRunning(true);
    if (!jobId) {
      terminal.writeln("\r\nNo job started yet. Start from Model Upload tab.");
      terminal.write("$ ");
      setIsRunning(false);
      return;
    }
    terminal.writeln(`\r\nTracking job ${jobId}...`);
    const poll = setInterval(async () => {
      try {
        const s = await getStatus(jobId);
        setStatus(s.status);
        if (lastPrintedStatusRef.current !== s.status) {
          terminal.writeln(`status: ${s.status}`);
          lastPrintedStatusRef.current = s.status;
        }
        // tail logs
        const logs = await fetchLogs(jobId);
        if (logs) {
          // Clear screen and reprint last ~200 lines for readability
          terminal.clear();
          const lines = logs.split(/\r?\n/).slice(-200);
          lines.forEach((ln) => terminal.writeln(ln));
        }
        if (s.status === "completed") {
          clearInterval(poll);
          terminal.writeln("Evaluation complete. You can view the dashboard for detailed results.");
          terminal.write("$ ");
          setIsRunning(false);
        }
      } catch {
        terminal.writeln("status: error");
      }
    }, 2000);
  };

  // Auto-track on mount if jobId exists
  useEffect(() => {
    if (!jobId || !xtermRef.current || isRunning) return;
    runEvaluation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const clearTerminal = () => {
    const terminal = xtermRef.current;
    if (terminal) {
      terminal.clear();
      terminal.writeln("🏆 Hackathon Evaluation Terminal v2.0");
      terminal.writeln("════════════════════════════════════════");
      terminal.write("$ ");
    }
  };

  return (
    <Card className="glass glow border-card-border">
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-destructive rounded-full"></div>
          <div className="w-3 h-3 bg-accent rounded-full"></div>
          <div className="w-3 h-3 bg-secondary rounded-full pulse-glow"></div>
          <span className="ml-2 text-sm font-mono text-muted-foreground">
            validation-terminal
          </span>
        </div>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={runEvaluation}
            disabled={isRunning}
            className="hover-glow"
          >
            {isRunning ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? "Tracking" : "Track Job"}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearTerminal}
            className="hover-glow"
          >
            <RotateCcw className="w-4 h-4" />
            Clear
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={() => onOpenDashboard?.()}
            className="hover-glow bg-primary text-primary-foreground px-4"
            disabled={!jobId || status !== "completed"}
          >
            Open Dashboard
          </Button>
        </div>
      </div>
      <div 
        ref={terminalRef} 
        className="h-80 p-4 terminal-glow font-mono text-sm overflow-hidden"
        style={{ minHeight: "320px" }}
      />
    </Card>
  );
};