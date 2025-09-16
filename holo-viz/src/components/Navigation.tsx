import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart3, 
  Upload, 
  Terminal as TerminalIcon, 
  FileText, 
  Mic, 
  MicOff,
  Menu,
  X,
  Brain,
  Zap
} from "lucide-react";

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const tabs = [
  { id: "upload", label: "Model Upload", icon: Upload },
  { id: "terminal", label: "Terminal", icon: TerminalIcon },
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "docs", label: "Documentation", icon: FileText },
];

export const Navigation = ({ activeTab, setActiveTab }: NavigationProps) => {
  const [isListening, setIsListening] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleVoiceCommand = () => {
    setIsListening(!isListening);
    // In a real app, this would start/stop speech recognition
  };

  return (
    <>
      {/* Desktop Navigation */}
      <motion.nav 
        className="hidden md:flex items-center justify-between p-6 glass border-b border-border"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary rounded-full blur-md opacity-75 animate-pulse"></div>
           
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-3">
              
              <span className="text-2xl font-bold">
                <span className="text-primary">R</span><span className="bg-gradient-to-r from-blue-500 to-white text-transparent bg-clip-text">
  obustifyAI
</span>


              </span>
              <div className="px-2 py-1 bg-primary/10 rounded-full">
                <span className="text-xs font-medium text-primary">Pro</span>
              </div>
            </div>
            <span className="text-xs text-muted-foreground font-medium tracking-wide">
              ML Model Validation Platform
            </span>
          </div>
 
        </div>

        <div className="flex items-center gap-2">
          {tabs.map((tab) => (
            <Button
              key={tab.id}
              variant={activeTab === tab.id ? "default" : "ghost"}
              onClick={() => setActiveTab(tab.id)}
              className={`hover-glow ${
                activeTab === tab.id 
                  ? "bg-primary text-primary-foreground glow" 
                  : "glass text-foreground"
              }`}
            >
              <tab.icon className="w-4 h-4 mr-2" />
              {tab.label}
            </Button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleVoiceCommand}
            className={`hover-glow ${isListening ? "text-secondary pulse-glow" : ""}`}
          >
            {isListening ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
          </Button>
          <Badge variant="outline" className="text-primary border-primary">
            Live
          </Badge>
        </div>
      </motion.nav>

      {/* Mobile Navigation */}
      <motion.nav 
        className="md:hidden flex items-center justify-between p-4 glass border-b border-border"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary rounded-full blur-sm opacity-75"></div>
            <div className="relative bg-gradient-to-r from-primary to-secondary p-1.5 rounded-full">
              <Brain className="w-4 h-4 text-white" />
            </div>
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold">
              <span className="text-primary">R</span><span className="bg-gradient-to-r from-blue-500 to-white text-transparent bg-clip-text">
  obustifyAI
</span>



            </span>
            <span className="text-xs text-muted-foreground font-medium -mt-1">
              ML Validation Platform
            </span>
          </div>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="hover-glow"
        >
          {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </Button>
      </motion.nav>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <motion.div
          className="md:hidden glass border-b border-border p-4 space-y-2"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          transition={{ duration: 0.3 }}
        >
          {tabs.map((tab) => (
            <Button
              key={tab.id}
              variant={activeTab === tab.id ? "default" : "ghost"}
              onClick={() => {
                setActiveTab(tab.id);
                setIsMobileMenuOpen(false);
              }}
              className={`w-full justify-start hover-glow ${
                activeTab === tab.id 
                  ? "bg-primary text-primary-foreground glow" 
                  : "glass text-foreground"
              }`}
            >
              <tab.icon className="w-4 h-4 mr-2" />
              {tab.label}
            </Button>
          ))}
          <div className="flex items-center justify-between pt-2 border-t border-border">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleVoiceCommand}
              className={`hover-glow ${isListening ? "text-secondary pulse-glow" : ""}`}
            >
              {isListening ? <Mic className="w-4 h-4 mr-2" /> : <MicOff className="w-4 h-4 mr-2" />}
              Voice Commands
            </Button>
            <Badge variant="outline" className="text-primary border-primary">
              Live
            </Badge>
          </div>
        </motion.div>
      )}
    </>
  );
};