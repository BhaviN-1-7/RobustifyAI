import { useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Upload, Check, AlertCircle, FileText, Zap, FolderOpen, Settings } from "lucide-react";
import { toast } from "sonner";
import { startEvaluation, uploadAndEvaluate, uploadAndEvaluateFolder } from "@/lib/api";

type Props = {
  onStarted?: (jobId: string) => void;
};

export const ModelUpload = ({ onStarted }: Props) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [modelUploaded, setModelUploaded] = useState(false);
  const [datasetUploaded, setDatasetUploaded] = useState(false);
  const [datasetPath, setDatasetPath] = useState("./dataset");
  const [modelPath, setModelPath] = useState("models/catsdogs_model.h5");
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [datasetFiles, setDatasetFiles] = useState<FileList | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [seed, setSeed] = useState(1234);

  const handleUpload = async () => {
    try {
      setIsUploading(true);
      setUploadProgress(25);
      setModelUploaded(true);
      setDatasetUploaded(true);
      let res;
      res = await startEvaluation({ dataset: datasetPath, model: modelPath, seed });
      setUploadProgress(100);
      toast.success("Evaluation started", { description: `Job ${res.job_id}` });
      onStarted?.(res.job_id);
    } catch (e: any) {
      toast.error("Failed to start evaluation", { description: e?.message || String(e) });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-3xl font-bold gradient-text mb-2">Hackathon Setup</h2>
        <p className="text-lg text-muted-foreground">
          Upload your Keras model (.h5) and dataset folder to begin evaluation
        </p>
      </motion.div>

      {/* Upload Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="glass glow border-card-border">
          <CardHeader>
            <CardTitle className="gradient-text">Configure Model & Dataset Paths</CardTitle>
            <CardDescription>
              Specify local paths to your Keras model (.h5) and dataset folder for evaluation
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Model Path */}
              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center glass hover-glow">
                <FileText className="w-10 h-10 text-primary mx-auto mb-3" />
                <p className="text-lg font-medium text-foreground mb-2">
                  Model Path
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Specify the path to your Keras model (.h5 file)
                </p>
                <div className="space-y-2">
                  <div className="flex gap-2">
                    <Input 
                      type="text" 
                      placeholder="models/catsdogs_model.h5"
                      className="glass border-border flex-1"
                      value={modelPath}
                      onChange={(e) => setModelPath(e.target.value)}
                    />
                    <Button 
                      variant="outline" 
                      className="glass border-border hover-glow"
                      onClick={() => document.getElementById('model-file-picker')?.click()}
                    >
                      <FolderOpen className="w-4 h-4" />
                    </Button>
                  </div>
                  <Input 
                    id="model-file-picker"
                    type="file" 
                    accept=".h5" 
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        // If it's a .h5 file, assume it's in the models directory
                        if (file.name.endsWith('.h5')) {
                          setModelPath(`models/${file.name}`);
                        } else {
                          setModelPath(file.name);
                        }
                      }
                    }}
                  />
                </div>
              </div>

              {/* Dataset Path */}
              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center glass hover-glow">
                <FolderOpen className="w-10 h-10 text-secondary mx-auto mb-3" />
                <p className="text-lg font-medium text-foreground mb-2">
                  Dataset Path
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Specify the path to your dataset folder
                </p>
                <div className="space-y-2">
                  <div className="flex gap-2">
                    <Input 
                      type="text" 
                      placeholder="./dataset"
                      className="glass border-border flex-1"
                      value={datasetPath}
                      onChange={(e) => setDatasetPath(e.target.value)}
                    />
                    <Button 
                      variant="outline" 
                      className="glass border-border hover-glow"
                      onClick={() => document.getElementById('dataset-folder-picker')?.click()}
                    >
                      <FolderOpen className="w-4 h-4" />
                    </Button>
                  </div>
                  <input 
                    id="dataset-folder-picker"
                    type="file" 
                    webkitdirectory=""
                    className="hidden"
                    onChange={(e) => {
                      const files = e.target.files;
                      if (files && files.length > 0) {
                        const firstFile = files[0];
                        const folderPath = firstFile.webkitRelativePath.split('/')[0];
                        setDatasetPath(`./${folderPath}`);
                      }
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Upload Progress */}
            {isUploading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Uploading files...</span>
                  <span className="text-sm text-muted-foreground">{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="glow" />
              </div>
            )}

            {/* Path Status */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className={`flex items-center gap-3 p-3 rounded-lg border ${
                modelPath ? "bg-primary/10 border-primary/30" : "bg-muted/50 border-border"
              }`}>
                {modelPath ? (
                  <Check className="w-5 h-5 text-primary" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-muted-foreground" />
                )}
                <span className="text-sm font-medium">
                  {modelPath ? "Model path set" : "Model path required"}
                </span>
              </div>
              
              <div className={`flex items-center gap-3 p-3 rounded-lg border ${
                datasetPath ? "bg-secondary/10 border-secondary/30" : "bg-muted/50 border-border"
              }`}>
                {datasetPath ? (
                  <Check className="w-5 h-5 text-secondary" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-muted-foreground" />
                )}
                <span className="text-sm font-medium">
                  {datasetPath ? "Dataset path set" : "Dataset path required"}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Configuration */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card className="glass glow border-card-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-accent" />
              Evaluation Configuration
            </CardTitle>
            <CardDescription>
              Configure parameters for the 17-step evaluation pipeline
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="random-seed">Random Seed</Label>
                <Input id="random-seed" type="number" className="glass border-border" value={seed} onChange={(e) => setSeed(parseInt(e.target.value || "0"))} />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="test-split">Test Split Ratio</Label>
                <Input 
                  id="test-split" 
                  type="number" 
                  step="0.1"
                  min="0.1"
                  max="0.5"
                  placeholder="0.2"
                  className="glass border-border"
                  defaultValue="0.2"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Evaluation Components</Label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[
                  'Confusion Matrix',
                  'ROC/PR Curves', 
                  'Bootstrap CI',
                  'Baseline Comparison',
                  'Statistical Test',
                  'Calibration Plot',
                  'Robustness Tests',
                  'Explainability',
                  'Efficiency Metrics'
                ].map((component) => (
                  <div key={component} className="flex items-center space-x-2">
                    <input 
                      type="checkbox" 
                      id={component.toLowerCase().replace(/[^a-z0-9]/g, '-')} 
                      defaultChecked={true}
                      className="rounded border-border"
                    />
                    <label 
                      htmlFor={component.toLowerCase().replace(/[^a-z0-9]/g, '-')} 
                      className="text-sm"
                    >
                      {component}
                    </label>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Action Buttons */}
      <motion.div
        className="flex gap-4 justify-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Button 
          size="lg" 
          onClick={handleUpload}
          disabled={isUploading || !modelPath || !datasetPath}
          className="bg-primary hover:bg-primary-glow text-primary-foreground px-8 glow"
        >
          {isUploading ? (
            <>
              <Zap className="w-4 h-4 mr-2 animate-spin" />
              Uploading... {uploadProgress}%
            </>
          ) : (
            <>
              <Upload className="w-4 h-4 mr-2" />
              Start Evaluation
            </>
          )}
        </Button>
      </motion.div>

      {/* Requirements Info */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
      >
        <Card className="glass border-card-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-primary" />
              Hackathon Requirements
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-medium text-foreground mb-2">File Requirements:</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Keras model file (.h5 format)</li>
                  <li>• Dataset folder: class_name/image.jpg</li>
                  <li>• ASCII file names only (no spaces)</li>
                  <li>• Python 3.8 or 3.9 environment</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-foreground mb-2">Output:</h4>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• Complete report.html file</li>
                  <li>• Summary.txt (max 200 words)</li>
                  <li>• All required tables and figures</li>
                  <li>• Evaluation runs within 2 hours</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};