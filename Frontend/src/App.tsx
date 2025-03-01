import React, { useState } from 'react';
import { ImageIcon, FileText, Upload, Wand2, Sparkles, Eraser, ChevronDown, XCircle } from 'lucide-react';
import axios from 'axios';

type Mode = 'image-to-caption' | 'caption-to-image';
type Model = 'gpt-2-vision' | 'custom-image-captioning';
type ImageDto={
  image:string,
  raw:File
}
interface ErrorTooltip {
  message: string;
  id: string;
}
export default function App() {
  const [mode, setMode] = useState<Mode>('image-to-caption');
  const [model, setModel] = useState<Model>('custom-image-captioning');
  const [image, setImage] = useState<ImageDto | null>(null);
  const [generatedCap, setGeneratedCap] = useState('');
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [isModelOpen, setIsModelOpen] = useState(false);
  const [errors, setErrors] = useState<ErrorTooltip[]>([]);

  const showError = (message: string) => {
    const id = Math.random().toString(36).substr(2, 9);
    setErrors(prev => [...prev, { message, id }]);
    setTimeout(() => {
      setErrors(prev => prev.filter(error => error.id !== id));
    }, 3000);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage({image: reader.result as string, raw: file});
      };
      reader.readAsDataURL(file);
    }
  };


  const handleGenerate = async () => {
    setLoading(true);
    console.log("Generating...");
    console.log("Image:", image);
    console.log("Model:", model);
    const url = model === 'gpt-2-vision' ? 'http://127.0.0.1:8000/upload/i2c/gpt2vit' : 'http://127.0.0.1:8000/upload/i2c/8k';
    if (image) {
      const formData = new FormData();
      await formData.append('image', image.raw);
      await axios
        .post(url, formData , {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })
        .then((res) => {
          const { caption } = res.data
          setGeneratedCap(caption)
          return res.data;
        })
        .catch((err)=>{
          showError(err.message)
          console.error(err)
        });
      }
    await new Promise(resolve => setTimeout(resolve, 1500));
    setLoading(false);
  };

  const handleClear = () => {
    setImage(null);
    setCaption('');
    // Reset file input
    const fileInput = document.getElementById('imageInput') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  };

  const handleModelSelect =(selectedModel: Model) =>{
    setModel(selectedModel);
    setIsModelOpen(false);
  }



  return (
    <div className="min-h-screen bg-[conic-gradient(at_bottom_left,_var(--tw-gradient-stops))] from-slate-900 via-purple-900 to-slate-900">
      <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=1600')] opacity-10 bg-cover bg-center mix-blend-overlay" />
      
        {/* Error Tooltips */}
        <div className="fixed top-0 left-1/2 -translate-x-1/2 z-50 space-y-2 p-4 pointer-events-none">
        {errors.map((error) => (
          <div
            key={error.id}
            className="bg-red-500 text-white px-6 py-3 rounded-xl shadow-lg flex items-center gap-2 animate-slide-in-out"
          >
            <XCircle className="h-5 w-5" />
            {error.message}
          </div>
        ))}
      </div>
      <div className="relative container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 flex items-center justify-center gap-3">
            <Sparkles className="h-8 w-8 text-purple-400" />
            AI/ML Group20 Vision Studio
            <Sparkles className="h-8 w-8 text-purple-400" />
          </h1>
          <p className="text-purple-200 text-lg">Transform your vision into words, or words into vision</p>
        </div>

 {/* Mode Selection */}
 <div className="flex flex-col items-center space-y-8 mb-12">
          <div className="flex justify-center space-x-6">
            <button
              onClick={() => setMode('image-to-caption')}
              className={`group flex items-center px-8 py-4 rounded-2xl ${
                mode === 'image-to-caption'
                  ? 'bg-purple-600 shadow-lg shadow-purple-500/50'
                  : 'bg-gray-800/50 hover:bg-gray-800/80 backdrop-blur-sm'
              } transition-all duration-300 ease-out`}
            >
              <ImageIcon className={`mr-3 h-6 w-6 ${mode === 'image-to-caption' ? 'text-white' : 'text-purple-400'} group-hover:scale-110 transition-transform duration-300`} />
              <span className="text-lg font-medium">Image to Caption</span>
            </button>
            <button
              onClick={() => setMode('caption-to-image')}
              className={`group flex items-center px-8 py-4 rounded-2xl ${
                mode === 'caption-to-image'
                  ? 'bg-purple-600 shadow-lg shadow-purple-500/50'
                  : 'bg-gray-800/50 hover:bg-gray-800/80 backdrop-blur-sm'
              } transition-all duration-300 ease-out`}
            >
              <FileText className={`mr-3 h-6 w-6 ${mode === 'caption-to-image' ? 'text-white' : 'text-purple-400'} group-hover:scale-110 transition-transform duration-300`} />
              <span className="text-lg font-medium">Caption to Image</span>
            </button>
          </div>

          {/* Model Selection Dropdown */}
          <div className="relative">
            <button
              onClick={() => setIsModelOpen(!isModelOpen)}
              className="flex items-center px-6 py-3 bg-gray-800/70 hover:bg-gray-800/90 rounded-xl transition-all duration-300 text-purple-200 backdrop-blur-sm border border-gray-700/50"
            >
              <span className="mr-2">Model:</span>
              <span className="font-medium mr-2">{
                model === 'gpt-2-vision' ? 'GPT-2 Vision' : 'Custom Image-to-Caption'
              }</span>
              <ChevronDown className={`h-4 w-4 transition-transform duration-300 ${isModelOpen ? 'rotate-180' : ''}`} />
            </button>
            
            {isModelOpen && (
              <div className="absolute mt-2 w-full py-2 bg-gray-800/95 backdrop-blur-sm rounded-xl shadow-xl border border-gray-700/50 z-10">
                <button
                  onClick={() => handleModelSelect('gpt-2-vision')}
                  className={`w-full px-4 py-2 text-left hover:bg-purple-500/20 transition-colors duration-200 ${
                    model === 'gpt-2-vision' ? 'text-purple-400' : 'text-purple-200'
                  }`}
                >
                  GPT-2 Vision
                </button>
                <button
                  onClick={() => handleModelSelect('custom-image-captioning')}
                  className={`w-full px-4 py-2 text-left hover:bg-purple-500/20 transition-colors duration-200 ${
                    model === 'custom-image-captioning' ? 'text-purple-400' : 'text-purple-200'
                  }`}
                >
                  Custom Image-to-Caption
                </button>
              </div>
            )}
          </div>
        </div>

        
        <div className="max-w-5xl mx-auto backdrop-blur-xl bg-gray-900/70 rounded-3xl shadow-2xl overflow-hidden border border-gray-700">
          <div className="p-8 md:p-12">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
              {/* Input Section */}
              <div className="space-y-6">
              <div className="flex justify-between items-center mb-6">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                    {mode === 'image-to-caption' ? (
                      <>
                        <ImageIcon className="h-6 w-6 text-purple-400" />
                        Upload Image
                      </>
                    ) : (
                      <>
                        <FileText className="h-6 w-6 text-purple-400" />
                        Enter Caption
                      </>
                    )}
                  </h2>
                  {(image || caption) && (
                    <button
                      onClick={handleClear}
                      className="group flex items-center px-4 py-2 bg-gray-800/50 hover:bg-gray-800/80 rounded-xl transition-all duration-300 text-purple-400 hover:text-purple-300"
                    >
                      <Eraser className="h-5 w-5 mr-2 group-hover:rotate-12 transition-transform duration-300" />
                      <span>Clear</span>
                    </button>
                  )}
                </div>

                {mode === 'image-to-caption' ? (
                  <div className="space-y-4">
                    <div
                      className="group border-3 border-dashed border-gray-700 hover:border-purple-500 rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 relative overflow-hidden"
                      onClick={() => document.getElementById('imageInput')?.click()}
                    >
                      {image ? (
                        <img
                          src={image.image}
                          alt="Uploaded"
                          className="max-h-80 mx-auto rounded-xl shadow-lg transition-transform duration-300 group-hover:scale-105"
                        />
                      ) : (
                        <div className="space-y-4">
                          <Upload className="h-16 w-16 mx-auto text-purple-400 transition-transform duration-300 group-hover:scale-110" />
                          <p className="text-purple-200 text-lg">
                            Drop your image here or click to browse
                          </p>
                          <p className="text-gray-400 text-sm">
                            Supports JPG, PNG and GIF
                          </p>
                        </div>
                      )}
                    </div>
                    <input
                      type="file"
                      id="imageInput"
                      className="hidden"
                      accept="image/*"
                      onChange={handleImageUpload}
                    />
                  </div>
                ) : (
                  <textarea
                    value={caption}
                    onChange={(e) => setCaption(e.target.value)}
                    placeholder="Describe your imagination in detail..."
                    className="w-full h-80 px-6 py-4 rounded-2xl bg-gray-800/50 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all duration-300 resize-none text-lg"
                  />
                )}
              </div>

              {/* Output Section */}
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                  <Wand2 className="h-6 w-6 text-purple-400" />
                  {mode === 'image-to-caption' ? 'Generated Caption' : 'Generated Image'}
                </h2>
                <div className="bg-gray-800/50 rounded-2xl p-8 h-80 flex items-center justify-center border border-gray-700">
                  {loading ? (
                    <div className="flex flex-col items-center gap-4">
                      <div className="animate-spin rounded-full h-16 w-16 border-4 border-purple-500 border-t-transparent" />
                      <p className="text-purple-200 animate-pulse">Processing...</p>
                    </div>
                  ) : (
                    <div className="text-center">
                      <Sparkles className="h-16 w-16 text-purple-400/50 mx-auto mb-4" />
                      <p className="text-gray-400 text-lg">
                        {generatedCap != "" && mode === 'image-to-caption'
                          ? generatedCap : mode === 'image-to-caption'
                          ? 'Your image caption will appear here'
                          : 'Your generated image will appear here'}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="mt-12 flex justify-center">
              <button
                onClick={handleGenerate}
                disabled={loading}
                className="group flex items-center px-12 py-4 bg-purple-600 hover:bg-purple-700 rounded-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50"
              >
                <Wand2 className="mr-3 h-6 w-6 group-hover:rotate-12 transition-transform duration-300" />
                <span className="text-lg font-medium">
                  {loading ? 'Processing...' : 'Generate'}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}