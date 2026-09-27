import { FC } from "react";
import { useUserTranscript } from "../../hooks/useUserTranscript";

export const UserTranscriptDisplay: FC = () => {
  const { transcripts, partial, isSpeaking, currentUtterance, turnCandidate } =
    useUserTranscript();

  if (
    transcripts.length === 0 &&
    !partial &&
    !isSpeaking &&
    !currentUtterance &&
    !turnCandidate
  ) {
    return null;
  }

  return (
    <div className="mt-4 p-3 rounded-lg bg-base-200 border border-base-300">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-primary">
          User Transcript
        </span>
        {isSpeaking && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-error rounded-full animate-pulse" />
            <span className="text-xs text-error">Speaking</span>
          </span>
        )}
        {turnCandidate && (
          <span className="text-xs text-success font-medium">
            Turn ready
          </span>
        )}
      </div>
      <div className="space-y-1">
        {transcripts.map((t, i) => (
          <p key={`${t.segmentId ?? i}-${t.sequence ?? i}`} className="text-sm">
            {t.text}
            {t.provider && (
              <span className="ml-2 text-xs opacity-50">[{t.provider}]</span>
            )}
          </p>
        ))}
        {currentUtterance && (
          <p className="text-sm text-primary font-medium">
            {currentUtterance}
          </p>
        )}
        {partial && (
          <p className="text-sm opacity-60 italic">{partial.text}...</p>
        )}
      </div>
    </div>
  );
};
