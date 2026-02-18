import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { Link, useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

type SavedPersona = {
  id: string;
  name: string;
  description: string;
  text_prompt: string;
  voice_prompt: string;
  text_temperature: number;
  text_topk: number;
  audio_temperature: number;
  audio_topk: number;
};

interface HomepageProps {
  showMicrophoneAccessMessage: boolean;
  startConnection: () => Promise<void>;
  textPrompt: string;
  setTextPrompt: (value: string) => void;
  voicePrompt: string;
  setVoicePrompt: (value: string) => void;
  savedPersonas: SavedPersona[];
  onSelectPersona: (persona: SavedPersona) => void;
}

const Homepage = ({
  startConnection,
  showMicrophoneAccessMessage,
  textPrompt,
  setTextPrompt,
  voicePrompt,
  setVoicePrompt,
  savedPersonas,
  onSelectPersona,
}: HomepageProps) => {
  return (
    <div className="text-center h-screen w-screen p-4 flex flex-col items-center pt-8">
      <div className="mb-6">
        <div className="flex items-center justify-center gap-3">
          <h1 className="text-4xl text-black">PersonaPlex</h1>
        </div>
        <p className="text-sm text-gray-600 mt-2">
          Full duplex conversational AI with text and voice control.
        </p>
        <Link
          to="/admin"
          className="inline-block mt-2 text-xs text-gray-400 hover:text-[#76b900] transition-colors"
        >
          Settings &rarr;
        </Link>
      </div>

      <div className="flex flex-grow justify-center items-center flex-col gap-6 w-full min-w-[500px] max-w-2xl">
        {/* Saved Personas Selector */}
        {savedPersonas.length > 0 && (
          <div className="w-full">
            <label className="block text-left text-base font-medium text-gray-700 mb-2">
              Saved Personas:
            </label>
            <div className="flex flex-wrap gap-2">
              {savedPersonas.map((persona) => (
                <button
                  key={persona.id}
                  onClick={() => onSelectPersona(persona)}
                  className="px-3 py-1.5 text-xs bg-white hover:bg-gray-100 text-gray-700 rounded-full border border-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                  title={persona.description || persona.text_prompt}
                >
                  {persona.name}
                </button>
              ))}
              <Link
                to="/admin"
                className="px-3 py-1.5 text-xs text-gray-400 hover:text-[#76b900] rounded-full border border-dashed border-gray-300 hover:border-[#76b900] transition-colors"
              >
                + Manage
              </Link>
            </div>
          </div>
        )}

        <div className="w-full">
          <label htmlFor="text-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Text Prompt:
          </label>
          <textarea
            id="text-prompt"
            name="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="w-full h-32 min-h-[80px] max-h-64 p-3 bg-white text-black border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
            placeholder="Enter your text prompt or select a persona above..."
            maxLength={2000}
          />
          <div className="text-right text-xs text-gray-500 mt-1">
            {textPrompt.length}/2000
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="voice-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Voice:
          </label>
          <select
            id="voice-prompt"
            name="voice-prompt"
            value={voicePrompt}
            onChange={(e) => setVoicePrompt(e.target.value)}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
          >
            {VOICE_OPTIONS.map((voice) => (
              <option key={voice} value={voice}>
                {voice
                  .replace('.pt', '')
                  .replace(/^NAT/, 'NATURAL_')
                  .replace(/^VAR/, 'VARIETY_')}
              </option>
            ))}
          </select>
      </div>

        {showMicrophoneAccessMessage && (
          <p className="text-center text-red-500">Please enable your microphone before proceeding</p>
        )}

        <Button onClick={async () => await startConnection()}>Connect</Button>
    </div>
    </div>
  );
}

export const Queue:FC = () => {
  const theme = "light" as const;  // Always use light theme
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();
  const [savedPersonas, setSavedPersonas] = useState<SavedPersona[]>([]);

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);

  // Load saved personas from the admin API
  useEffect(() => {
    fetch("/api/admin/personas")
      .then((res) => (res.ok ? res.json() : []))
      .then(setSavedPersonas)
      .catch(() => setSavedPersonas([]));
  }, []);

  const handleSelectPersona = useCallback(
    (persona: SavedPersona) => {
      modelParams.setTextPrompt(persona.text_prompt);
      modelParams.setVoicePrompt(persona.voice_prompt);
      modelParams.setTextTemperature(persona.text_temperature);
      modelParams.setTextTopk(persona.text_topk);
      modelParams.setAudioTemperature(persona.audio_temperature);
      modelParams.setAudioTopk(persona.audio_topk);
    },
    [modelParams]
  );

  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
}, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
      // Prewarm decoder worker as soon as we have audio context
      // This gives WASM time to load while user grants mic access
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const startConnection = useCallback(async() => {
      await startProcessor();
      const hasAccess = await getMicrophoneAccess();
      if (hasAccess) {
      // Values are already set in modelParams, they get passed to Conversation
    }
  }, [startProcessor, getMicrophoneAccess]);

  return (
    <>
      {(hasMicrophoneAccess && audioContext.current && worklet.current) ? (
        <Conversation
        workerAddr={overrideWorkerAddr ?? ""}
        audioContext={audioContext as MutableRefObject<AudioContext|null>}
        worklet={worklet as MutableRefObject<AudioWorkletNode|null>}
        theme={theme}
        startConnection={startConnection}
        {...modelParams}
        />
      ) : (
        <Homepage
          startConnection={startConnection}
          showMicrophoneAccessMessage={showMicrophoneAccessMessage}
          textPrompt={modelParams.textPrompt}
          setTextPrompt={modelParams.setTextPrompt}
          voicePrompt={modelParams.voicePrompt}
          setVoicePrompt={modelParams.setVoicePrompt}
          savedPersonas={savedPersonas}
          onSelectPersona={handleSelectPersona}
        />
      )}
    </>
  );
};
