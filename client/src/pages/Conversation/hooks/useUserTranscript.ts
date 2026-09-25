import { useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { UserTranscriptData } from "../../../protocol/types";

export type TranscriptItem = {
  type: UserTranscriptData["type"];
  segmentId?: string;
  sequence?: number;
  text: string;
  startMs?: number | null;
  endMs?: number | null;
  confidence?: number | null;
  language?: string | null;
  provider?: string;
  utteranceId?: string;
  turnId?: string;
  timestamp: number;
};

export const useUserTranscript = () => {
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [partial, setPartial] = useState<TranscriptItem | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentUtterance, setCurrentUtterance] = useState<string>("");
  const [turnCandidate, setTurnCandidate] = useState<TranscriptItem | null>(null);
  const sequenceRef = useRef(0);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);

    let message;
    try {
      message = decodeMessage(dataArray);
    } catch {
      return;
    }

    if (message.type !== "user_transcript") return;

    const data = message.data;

    switch (data.type) {
      case "user_transcript.partial":
        setPartial({
          type: data.type,
          segmentId: data.segment_id,
          sequence: data.sequence,
          text: data.text ?? "",
          provider: data.provider,
          timestamp: Date.now(),
        });
        break;

      case "user_transcript.final":
        setPartial(null);
        setTranscripts((prev) => [
          ...prev,
          {
            type: data.type,
            segmentId: data.segment_id,
            sequence: data.sequence,
            text: data.text ?? "",
            startMs: data.start_ms,
            endMs: data.end_ms,
            confidence: data.confidence,
            language: data.language,
            provider: data.provider,
            timestamp: Date.now(),
          },
        ]);
        break;

      case "speech.started":
        setIsSpeaking(true);
        break;

      case "speech.stopped":
        setIsSpeaking(false);
        break;

      case "utterance.updated":
        setCurrentUtterance(data.text ?? "");
        break;

      case "utterance.finalized":
        setCurrentUtterance("");
        break;

      case "turn.candidate":
        setTurnCandidate({
          type: data.type,
          text: data.text ?? "",
          utteranceId: data.utterance_id,
          turnId: data.turn_id,
          timestamp: Date.now(),
        });
        break;

      case "turn.cancelled":
        setTurnCandidate(null);
        break;
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) return;
    setTranscripts([]);
    setPartial(null);
    setCurrentUtterance("");
    setTurnCandidate(null);
    sequenceRef.current = 0;
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  return { transcripts, partial, isSpeaking, currentUtterance, turnCandidate };
};
