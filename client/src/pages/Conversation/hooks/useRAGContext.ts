import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { RAGContextData } from "../../../protocol/types";

export type RAGContextItem = RAGContextData & {
  timestamp: number;
};

export const useRAGContext = () => {
  const [contexts, setContexts] = useState<RAGContextItem[]>([]);
  const [ragEnabled, setRagEnabled] = useState(false);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);
    if (message.type === "rag_context") {
      setRagEnabled(true);
      setContexts(prev => [
        ...prev,
        {
          ...message.data,
          timestamp: Date.now(),
        },
      ]);
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setContexts([]);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  return { contexts, ragEnabled };
};
