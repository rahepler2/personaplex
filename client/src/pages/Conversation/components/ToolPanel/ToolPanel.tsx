import { FC } from "react";
import { useOrchestratorEvents } from "../../hooks/useOrchestratorEvents";

export const ToolPanel: FC = () => {
  const { events, lastResponse, isProcessing } = useOrchestratorEvents();

  if (events.length === 0 && !lastResponse) {
    return null;
  }

  return (
    <div className="mt-4 p-3 rounded-lg bg-base-200 border border-base-300">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-primary">
          Tool Calling
        </span>
        {isProcessing && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-warning rounded-full animate-pulse" />
            <span className="text-xs text-warning">Processing</span>
          </span>
        )}
      </div>
      <div className="space-y-2">
        {events.map((ev, i) => {
          switch (ev.type) {
            case "tool_call":
              return (
                <div key={i} className="text-xs p-2 rounded bg-base-300">
                  <span className="font-mono text-accent">
                    {ev.data.tool}
                  </span>
                  <span className="opacity-50 ml-1">called</span>
                  {ev.data.arguments && Object.keys(ev.data.arguments).length > 0 && (
                    <pre className="mt-1 text-xs opacity-70 overflow-x-auto">
                      {JSON.stringify(ev.data.arguments, null, 2)}
                    </pre>
                  )}
                </div>
              );
            case "tool_result":
              return (
                <div
                  key={i}
                  className={`text-xs p-2 rounded ${
                    ev.data.is_error ? "bg-error/10" : "bg-success/10"
                  }`}
                >
                  <span className="font-mono">
                    {ev.data.tool}
                  </span>
                  <span className="opacity-50 ml-1">
                    {ev.data.is_error ? "error" : "result"}
                  </span>
                  {ev.data.content && (
                    <p className="mt-1 opacity-80 break-words">
                      {ev.data.content.slice(0, 300)}
                      {ev.data.content.length > 300 ? "..." : ""}
                    </p>
                  )}
                </div>
              );
            case "sidecar_response":
              return (
                <div key={i} className="text-sm p-2 rounded bg-primary/10">
                  <p>{ev.data.text}</p>
                  {(ev.data.tool_calls_count ?? 0) > 0 && (
                    <span className="text-xs opacity-50 mt-1 block">
                      {ev.data.tool_calls_count} tool call(s) used
                    </span>
                  )}
                </div>
              );
            case "error":
              return (
                <div key={i} className="text-xs p-2 rounded bg-error/10 text-error">
                  {ev.data.error}
                </div>
              );
            default:
              return null;
          }
        })}
      </div>
    </div>
  );
};
