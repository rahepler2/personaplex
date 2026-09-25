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
    <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-medium text-blue-700">
          User Transcript
        </span>
        {isSpeaking && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-xs text-red-600">Speaking</span>
          </span>
        )}
        {turnCandidate && (
          <span className="text-xs text-green-600 font-medium">
            Turn ready
          </span>
        )}
      </div>
      <div className="space-y-1">
        {transcripts.map((t, i) => (
          <p key={`${t.segmentId ?? i}-${t.sequence ?? i}`} className="text-sm text-gray-800">
            {t.text}
            {t.provider && (
              <span className="ml-2 text-xs text-gray-400">[{t.provider}]</span>
            )}
          </p>
        ))}
        {currentUtterance && (
          <p className="text-sm text-blue-700 font-medium">
            {currentUtterance}
          </p>
        )}
        {partial && (
          <p className="text-sm text-gray-500 italic">{partial.text}...</p>
        )}
      </div>
    </div>
  );
};
