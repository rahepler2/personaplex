import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { OrchestratorEventData } from "../../../protocol/types";

export type OrchestratorItem = OrchestratorEventData & {
  receivedAt: number;
};

export const useOrchestratorEvents = () => {
  const [events, setEvents] = useState<OrchestratorItem[]>([]);
  const [lastResponse, setLastResponse] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);

    let message;
    try {
      message = decodeMessage(dataArray);
    } catch {
      return;
    }

    if (message.type !== "orchestrator_event") return;

    const data = message.data;
    const item: OrchestratorItem = { ...data, receivedAt: Date.now() };

    setEvents((prev) => [...prev.slice(-49), item]);

    switch (data.type) {
      case "tool_call":
        setIsProcessing(true);
        break;
      case "sidecar_response":
        setIsProcessing(false);
        setLastResponse(data.data.text ?? null);
        break;
      case "error":
        setIsProcessing(false);
        break;
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) return;
    setEvents([]);
    setLastResponse(null);
    setIsProcessing(false);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  return { events, lastResponse, isProcessing };
};
