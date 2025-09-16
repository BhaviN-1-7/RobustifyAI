import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  Download, 
  FileText, 
  BarChart3, 
  Settings, 
  CheckCircle,
  Clock,
  Eye,
  Share2,
  ExternalLink
} from "lucide-react";
import { toast } from "sonner";

// Report sections removed as requested

const exportFormats = [
  { name: "PDF Report", format: "pdf", size: "2.4 MB", icon: FileText, endpoint: "pdf" },
  { name: "HTML Dashboard", format: "html", size: "1.8 MB", icon: Eye, endpoint: "html" },
  { name: "JSON Data", format: "json", size: "0.3 MB", icon: Settings, endpoint: "json" },
];

type Props = { jobId: string | null };

export const DocumentGenerator = ({ jobId }: Props) => {
  const [generatingReport, setGeneratingReport] = useState(false);
  const [reportProgress, setReportProgress] = useState(0);
  const [ready, setReady] = useState(false);
  const [reportUrl, setReportUrl] = useState<string | null>(null);

  const generateReport = () => {
    setGeneratingReport(true);
    setReportProgress(0);

    // Simulate report generation
    const interval = setInterval(() => {
      setReportProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setGeneratingReport(false);
          toast.success("Documentation generated successfully! 📄", {
            description: "Your comprehensive validation report is ready for download.",
          });
          return 100;
        }
        return prev + 8;
      });
    }, 300);
  };

  const downloadReport = (format: { name: string; format: string; endpoint: string }) => {
    if (!jobId) {
      toast.error("No report available", {
        description: "Please run an evaluation first.",
      });
      return;
    }

    const url = `/api/report/${jobId}.${format.endpoint}`;
    const link = document.createElement('a');
    link.href = url;
    link.download = `report.${format.endpoint}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    toast.success(`Downloading ${format.name}...`, {
      description: "Your file will be saved to your downloads folder.",
    });
  };

  const previewReport = () => {
    if (!jobId) {
      toast.error("No report available", {
        description: "Please run an evaluation first.",
      });
      return;
    }

    const url = `/api/report/${jobId}.html`;
    window.open(url, '_blank');
    
    toast.success("Opening report preview...", {
      description: "Report will open in a new tab.",
    });
  };

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      if (!jobId) return;
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        setReportProgress(data.progress ?? 0);
        if (data.status === "completed") {
          setReady(true);
          setReportUrl(`/api/report/${jobId}.html`);
        }
      } catch {}
    }
    const t = setInterval(poll, 2000);
    poll();
    return () => { cancelled = true; clearInterval(t); };
  }, [jobId]);

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl font-bold gradient-text mb-2">Documentation Generator</h2>
        <p className="text-lg text-muted-foreground">
          Generate comprehensive validation reports and documentation
        </p>
      </motion.div>

      <div className="grid grid-cols-1 gap-6">

        {/* Export Options */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="glass glow border-card-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Download className="w-5 h-5 text-secondary" />
                Export & Preview Options
              </CardTitle>
              <CardDescription>
                Download your validation report in various formats or preview online
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Preview Section */}
              <div className="space-y-2 mb-6">
                <h4 className="font-medium text-foreground mb-3">Report Preview</h4>
                <Button 
                  onClick={previewReport}
                  disabled={!ready}
                  className="w-full bg-secondary hover:bg-secondary/80 text-secondary-foreground glow"
                >
                  <Eye className="w-4 h-4 mr-2" />
                  Preview Report in New Tab
                </Button>
              </div>

              <Separator className="my-4" />

              {/* Download Section */}
              <div className="space-y-2">
                <h4 className="font-medium text-foreground mb-3">Download Options</h4>
                {exportFormats.map((format, index) => (
                  <motion.div
                    key={format.format}
                    className="flex items-center justify-between p-3 rounded-lg glass border border-border hover-glow cursor-pointer"
                    onClick={() => downloadReport(format)}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <div className="flex items-center gap-3">
                      <format.icon className="w-4 h-4 text-primary" />
                      <div>
                        <div className="font-medium">{format.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {format.size}
                        </div>
                      </div>
                    </div>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="hover-glow"
                      disabled={!ready}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </motion.div>
                ))}
              </div>

              <Separator className="my-4" />

              {/* Additional Actions */}
              <div className="space-y-2">
                <Button 
                  onClick={() => {
                    if (!jobId) {
                      toast.error("No report available");
                      return;
                    }
                    navigator.clipboard.writeText(`${window.location.origin}/api/report/${jobId}.html`);
                    toast.success("Report link copied to clipboard!");
                  }}
                  variant="outline" 
                  className="w-full glass border-border hover-glow"
                  disabled={!ready}
                >
                  <Share2 className="w-4 h-4 mr-2" />
                  Copy Report Link
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Report Status */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card className="glass border-card-border text-center">
          <CardContent className="p-6">
            <div className={`text-3xl font-bold ${ready ? 'text-green-400' : 'text-muted-foreground'} mb-1`}>
              {ready ? '✓' : '○'}
            </div>
            <div className="font-medium mb-1">Report Status</div>
            <div className="text-sm text-muted-foreground">
              {ready ? 'Ready for download' : 'Waiting for evaluation'}
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass border-card-border text-center">
          <CardContent className="p-6">
            <div className="text-3xl font-bold text-primary mb-1">
              {reportProgress}%
            </div>
            <div className="font-medium mb-1">Progress</div>
            <div className="text-sm text-muted-foreground">
              Evaluation pipeline completion
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass border-card-border text-center">
          <CardContent className="p-6">
            <div className="text-3xl font-bold text-secondary mb-1">
              {jobId ? 'Active' : 'None'}
            </div>
            <div className="font-medium mb-1">Job ID</div>
            <div className="text-sm text-muted-foreground">
              {jobId ? `Job: ${jobId.slice(0, 8)}...` : 'No active job'}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};