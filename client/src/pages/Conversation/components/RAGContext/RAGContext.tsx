import { FC, useEffect, useRef } from "react";
import { useRAGContext, RAGContextItem } from "../../hooks/useRAGContext";

type RAGContextDisplayProps = {
  containerRef?: React.RefObject<HTMLDivElement>;
};

const RAGContextCard: FC<{ item: RAGContextItem; index: number }> = ({ item, index }) => {
  return (
    <div className="mb-2 p-2 rounded bg-base-200 border border-base-300 text-xs">
      <div className="font-semibold text-primary mb-1 flex items-center gap-1">
        <span>Knowledge</span>
        <span className="opacity-50">#{index + 1}</span>
      </div>
      <div className="opacity-70 mb-1 italic truncate">
        Query: {item.query}
      </div>
      <div className="whitespace-pre-wrap break-words">
        {item.content}
      </div>
    </div>
  );
};

export const RAGContextDisplay: FC<RAGContextDisplayProps> = ({ containerRef }) => {
  const { contexts, ragEnabled } = useRAGContext();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef?.current || scrollRef.current;
    if (el) {
      el.scroll({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [contexts]);

  if (!ragEnabled && contexts.length === 0) {
    return null;
  }

  return (
    <div ref={scrollRef} className="h-full w-full max-w-full max-h-full overflow-y-auto p-2">
      <div className="text-xs font-bold mb-2 opacity-60 uppercase tracking-wider">
        Retrieved Knowledge
      </div>
      {contexts.length === 0 ? (
        <div className="text-xs opacity-40 italic">
          Listening for relevant context...
        </div>
      ) : (
        contexts.map((ctx, i) => (
          <RAGContextCard key={i} item={ctx} index={i} />
        ))
      )}
    </div>
  );
};
