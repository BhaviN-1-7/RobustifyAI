import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Navigation } from "@/components/Navigation";
import { Dashboard } from "@/components/Dashboard";
import { ModelUpload } from "@/components/ModelUpload";
import { Terminal } from "@/components/Terminal";
import { DocumentGenerator } from "@/components/DocumentGenerator";

const Index = () => {
  const [activeTab, setActiveTab] = useState("upload");
  const [jobId, setJobId] = useState<string | null>(null);

  // Restore jobId on refresh
  useEffect(() => {
    const saved = localStorage.getItem("robustify_job_id");
    if (saved) setJobId(saved);
  }, []);

  // Persist jobId changes
  useEffect(() => {
    if (jobId) localStorage.setItem("robustify_job_id", jobId);
  }, [jobId]);

  const renderActiveComponent = () => {
    switch (activeTab) {
      case "dashboard":
        return <Dashboard jobId={jobId} onOpenDocs={() => setActiveTab("docs")} />;
      case "upload":
        return <ModelUpload onStarted={(id: string) => { setJobId(id); setActiveTab("terminal"); }} />;
      case "terminal":
        return <Terminal jobId={jobId} onOpenDashboard={() => setActiveTab("dashboard")} />;
      case "docs":
        return <DocumentGenerator jobId={jobId} />;
      default:
        return <Dashboard jobId={jobId} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="container mx-auto px-4 py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderActiveComponent()}
          </motion.div>
        </AnimatePresence>
      </main>
      
      {/* Background gradient overlay */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>
    </div>
  );
};

export default Index;
