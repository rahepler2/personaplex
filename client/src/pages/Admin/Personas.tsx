import { FC, useCallback, useEffect, useState } from "react";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

type Persona = {
  id: string;
  name: string;
  description: string;
  text_prompt: string;
  voice_prompt: string;
  text_temperature: number;
  text_topk: number;
  audio_temperature: number;
  audio_topk: number;
  created_at: number;
  updated_at: number;
};

const emptyPersona = (): Partial<Persona> => ({
  name: "",
  description: "",
  text_prompt: "",
  voice_prompt: "NATF0.pt",
  text_temperature: 0.7,
  text_topk: 25,
  audio_temperature: 0.8,
  audio_topk: 250,
});

const PersonaEditor: FC<{
  persona: Partial<Persona>;
  onChange: (p: Partial<Persona>) => void;
  onSave: () => void;
  onCancel: () => void;
  isNew: boolean;
  saving: boolean;
}> = ({ persona, onChange, onSave, onCancel, isNew, saving }) => {
  const set = (key: string, value: string | number) =>
    onChange({ ...persona, [key]: value });

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
      <h3 className="text-lg font-medium text-gray-900">
        {isNew ? "New Persona" : "Edit Persona"}
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
          <input
            type="text"
            value={persona.name || ""}
            onChange={(e) => set("name", e.target.value)}
            className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900]"
            placeholder="e.g. Customer Support Agent"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Voice</label>
          <select
            value={persona.voice_prompt || "NATF0.pt"}
            onChange={(e) => set("voice_prompt", e.target.value)}
            className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900]"
          >
            {VOICE_OPTIONS.map((v) => (
              <option key={v} value={v}>
                {v.replace(".pt", "").replace(/^NAT/, "NATURAL_").replace(/^VAR/, "VARIETY_")}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
        <input
          type="text"
          value={persona.description || ""}
          onChange={(e) => set("description", e.target.value)}
          className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900]"
          placeholder="Short description of this persona"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Text Prompt
          <span className="text-xs text-gray-400 ml-2">
            ({(persona.text_prompt || "").length}/2000)
          </span>
        </label>
        <textarea
          value={persona.text_prompt || ""}
          onChange={(e) => set("text_prompt", e.target.value)}
          className="w-full h-40 p-3 bg-white text-black border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900] font-mono text-sm"
          placeholder="Define the persona's behavior, role, knowledge, and personality..."
          maxLength={2000}
        />
      </div>

      <details className="border border-gray-200 rounded p-3">
        <summary className="text-sm font-medium text-gray-600 cursor-pointer">
          Advanced Parameters
        </summary>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Text Temperature</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              max="1.5"
              value={persona.text_temperature ?? 0.7}
              onChange={(e) => set("text_temperature", parseFloat(e.target.value))}
              className="w-full p-1.5 text-sm bg-white text-black border border-gray-300 rounded"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Text Top-K</label>
            <input
              type="number"
              min="1"
              max="500"
              value={persona.text_topk ?? 25}
              onChange={(e) => set("text_topk", parseInt(e.target.value))}
              className="w-full p-1.5 text-sm bg-white text-black border border-gray-300 rounded"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Audio Temperature</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              max="1.5"
              value={persona.audio_temperature ?? 0.8}
              onChange={(e) => set("audio_temperature", parseFloat(e.target.value))}
              className="w-full p-1.5 text-sm bg-white text-black border border-gray-300 rounded"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Audio Top-K</label>
            <input
              type="number"
              min="1"
              max="500"
              value={persona.audio_topk ?? 250}
              onChange={(e) => set("audio_topk", parseInt(e.target.value))}
              className="w-full p-1.5 text-sm bg-white text-black border border-gray-300 rounded"
            />
          </div>
        </div>
      </details>

      <div className="flex gap-2 pt-2">
        <button
          onClick={onSave}
          disabled={saving || !persona.name?.trim()}
          className="px-4 py-2 bg-[#76b900] text-white rounded hover:bg-[#5a8f00] disabled:opacity-50 text-sm font-medium"
        >
          {saving ? "Saving..." : isNew ? "Create Persona" : "Save Changes"}
        </button>
        <button
          onClick={onCancel}
          className="px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50 text-sm"
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export const Personas: FC = () => {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<Partial<Persona> | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [saving, setSaving] = useState(false);

  const fetchPersonas = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/personas");
      if (res.ok) {
        setPersonas(await res.json());
      }
    } catch (e) {
      console.error("Failed to load personas:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPersonas();
  }, [fetchPersonas]);

  const handleCreate = () => {
    setEditing(emptyPersona());
    setIsNew(true);
  };

  const handleEdit = (persona: Persona) => {
    setEditing({ ...persona });
    setIsNew(false);
  };

  const handleSave = async () => {
    if (!editing) return;
    setSaving(true);
    try {
      const url = isNew
        ? "/api/admin/personas"
        : `/api/admin/personas/${editing.id}`;
      const method = isNew ? "POST" : "PUT";
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editing),
      });
      if (res.ok) {
        setEditing(null);
        await fetchPersonas();
      }
    } catch (e) {
      console.error("Save failed:", e);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this persona?")) return;
    try {
      await fetch(`/api/admin/personas/${id}`, { method: "DELETE" });
      await fetchPersonas();
    } catch (e) {
      console.error("Delete failed:", e);
    }
  };

  if (loading) {
    return <div className="text-gray-500">Loading personas...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Personas</h2>
          <p className="text-sm text-gray-500">
            Create and manage conversation personas with custom prompts and voices.
          </p>
        </div>
        {!editing && (
          <button
            onClick={handleCreate}
            className="px-4 py-2 bg-[#76b900] text-white rounded hover:bg-[#5a8f00] text-sm font-medium"
          >
            + New Persona
          </button>
        )}
      </div>

      {editing && (
        <PersonaEditor
          persona={editing}
          onChange={setEditing}
          onSave={handleSave}
          onCancel={() => setEditing(null)}
          isNew={isNew}
          saving={saving}
        />
      )}

      <div className="space-y-3">
        {personas.map((persona) => (
          <div
            key={persona.id}
            className="bg-white rounded-lg border border-gray-200 p-4 flex justify-between items-start hover:border-gray-300 transition-colors"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="font-medium text-gray-900">{persona.name}</h3>
                <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full">
                  {persona.voice_prompt.replace(".pt", "")}
                </span>
              </div>
              {persona.description && (
                <p className="text-sm text-gray-500 mb-1">{persona.description}</p>
              )}
              <p className="text-xs text-gray-400 truncate max-w-xl">
                {persona.text_prompt}
              </p>
            </div>
            <div className="flex gap-2 ml-4 shrink-0">
              <button
                onClick={() => handleEdit(persona)}
                className="px-3 py-1.5 text-xs bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50"
              >
                Edit
              </button>
              <button
                onClick={() => handleDelete(persona.id)}
                className="px-3 py-1.5 text-xs bg-white text-red-600 border border-red-200 rounded hover:bg-red-50"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
        {personas.length === 0 && !editing && (
          <div className="text-center py-12 text-gray-400">
            No personas yet. Click "New Persona" to create one.
          </div>
        )}
      </div>
    </div>
  );
};
