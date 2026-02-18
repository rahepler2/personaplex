import { FC, useCallback, useEffect, useState } from "react";

type MCPTool = {
  name: string;
  description: string;
};

type MCPServer = {
  id: string;
  name: string;
  transport: "sse" | "stdio";
  url?: string;
  command?: string;
  args: string[];
  env: Record<string, string>;
  enabled: boolean;
  connected: boolean;
  tools: MCPTool[];
  connection_error?: string;
  created_at: number;
  updated_at: number;
};

const emptyServer = (): Partial<MCPServer> => ({
  name: "",
  transport: "sse",
  url: "",
  command: "",
  args: [],
  env: {},
  enabled: true,
});

const ServerEditor: FC<{
  server: Partial<MCPServer>;
  onChange: (s: Partial<MCPServer>) => void;
  onSave: () => void;
  onCancel: () => void;
  onTest: () => void;
  isNew: boolean;
  saving: boolean;
  testing: boolean;
  testResult: { success: boolean; tools: MCPTool[]; error?: string } | null;
}> = ({ server, onChange, onSave, onCancel, onTest, isNew, saving, testing, testResult }) => {
  const set = (key: string, value: unknown) =>
    onChange({ ...server, [key]: value });

  const [envText, setEnvText] = useState(
    Object.entries(server.env || {})
      .map(([k, v]) => `${k}=${v}`)
      .join("\n")
  );

  const [argsText, setArgsText] = useState(
    (server.args || []).join(" ")
  );

  const handleEnvChange = (text: string) => {
    setEnvText(text);
    const env: Record<string, string> = {};
    text.split("\n").forEach((line) => {
      const eqIdx = line.indexOf("=");
      if (eqIdx > 0) {
        env[line.slice(0, eqIdx).trim()] = line.slice(eqIdx + 1).trim();
      }
    });
    set("env", env);
  };

  const handleArgsChange = (text: string) => {
    setArgsText(text);
    set("args", text.split(/\s+/).filter(Boolean));
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
      <h3 className="text-lg font-medium text-gray-900">
        {isNew ? "Add MCP Server" : "Edit MCP Server"}
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Server Name</label>
          <input
            type="text"
            value={server.name || ""}
            onChange={(e) => set("name", e.target.value)}
            className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900]"
            placeholder="e.g. knowledge-base"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Transport</label>
          <select
            value={server.transport || "sse"}
            onChange={(e) => set("transport", e.target.value)}
            className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900]"
          >
            <option value="sse">SSE (HTTP)</option>
            <option value="stdio">Stdio (Local Process)</option>
          </select>
        </div>
      </div>

      {server.transport === "sse" ? (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Server URL</label>
          <input
            type="text"
            value={server.url || ""}
            onChange={(e) => set("url", e.target.value)}
            className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] font-mono text-sm"
            placeholder="http://localhost:8080/sse"
          />
          <p className="text-xs text-gray-400 mt-1">
            The SSE endpoint URL of the MCP server
          </p>
        </div>
      ) : (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Command</label>
            <input
              type="text"
              value={server.command || ""}
              onChange={(e) => set("command", e.target.value)}
              className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] font-mono text-sm"
              placeholder="e.g. npx, python, node"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Arguments</label>
            <input
              type="text"
              value={argsText}
              onChange={(e) => handleArgsChange(e.target.value)}
              className="w-full p-2 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] font-mono text-sm"
              placeholder="e.g. -y @modelcontextprotocol/server-filesystem /path"
            />
            <p className="text-xs text-gray-400 mt-1">Space-separated arguments</p>
          </div>
        </>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Environment Variables
          <span className="text-xs text-gray-400 ml-1">(one per line, KEY=VALUE)</span>
        </label>
        <textarea
          value={envText}
          onChange={(e) => handleEnvChange(e.target.value)}
          className="w-full h-20 p-2 bg-white text-black border border-gray-300 rounded font-mono text-xs focus:outline-none focus:ring-2 focus:ring-[#76b900]"
          placeholder="API_KEY=your-key-here"
        />
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="enabled"
          checked={server.enabled ?? true}
          onChange={(e) => set("enabled", e.target.checked)}
          className="rounded border-gray-300"
        />
        <label htmlFor="enabled" className="text-sm text-gray-700">
          Auto-connect on server start
        </label>
      </div>

      {testResult && (
        <div
          className={`p-3 rounded text-sm ${
            testResult.success
              ? "bg-green-50 border border-green-200 text-green-800"
              : "bg-red-50 border border-red-200 text-red-800"
          }`}
        >
          {testResult.success ? (
            <>
              <span className="font-medium">Connection successful!</span>
              {testResult.tools.length > 0 && (
                <div className="mt-2">
                  <span className="text-xs font-medium">Available tools:</span>
                  <ul className="mt-1 space-y-0.5">
                    {testResult.tools.map((t) => (
                      <li key={t.name} className="text-xs">
                        <span className="font-mono">{t.name}</span>
                        {t.description && (
                          <span className="text-green-600 ml-1">- {t.description}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <>
              <span className="font-medium">Connection failed: </span>
              {testResult.error}
            </>
          )}
        </div>
      )}

      <div className="flex gap-2 pt-2">
        <button
          onClick={onTest}
          disabled={testing || !server.name?.trim()}
          className="px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50 text-sm font-medium disabled:opacity-50"
        >
          {testing ? "Testing..." : "Test Connection"}
        </button>
        <button
          onClick={onSave}
          disabled={saving || !server.name?.trim()}
          className="px-4 py-2 bg-[#76b900] text-white rounded hover:bg-[#5a8f00] disabled:opacity-50 text-sm font-medium"
        >
          {saving ? "Saving..." : isNew ? "Add Server" : "Save Changes"}
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

export const MCPServers: FC = () => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<Partial<MCPServer> | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    tools: MCPTool[];
    error?: string;
  } | null>(null);

  const fetchServers = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/mcp-servers");
      if (res.ok) {
        setServers(await res.json());
      }
    } catch (e) {
      console.error("Failed to load MCP servers:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchServers();
  }, [fetchServers]);

  const handleAdd = () => {
    setEditing(emptyServer());
    setIsNew(true);
    setTestResult(null);
  };

  const handleEdit = (server: MCPServer) => {
    setEditing({ ...server });
    setIsNew(false);
    setTestResult(null);
  };

  const handleTest = async () => {
    if (!editing) return;
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch("/api/admin/mcp-servers/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editing),
      });
      if (res.ok) {
        setTestResult(await res.json());
      }
    } catch (e) {
      setTestResult({ success: false, tools: [], error: String(e) });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    if (!editing) return;
    setSaving(true);
    try {
      const url = isNew
        ? "/api/admin/mcp-servers"
        : `/api/admin/mcp-servers/${editing.id}`;
      const method = isNew ? "POST" : "PUT";
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editing),
      });
      if (res.ok) {
        setEditing(null);
        setTestResult(null);
        await fetchServers();
      }
    } catch (e) {
      console.error("Save failed:", e);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Remove this MCP server?")) return;
    try {
      await fetch(`/api/admin/mcp-servers/${id}`, { method: "DELETE" });
      await fetchServers();
    } catch (e) {
      console.error("Delete failed:", e);
    }
  };

  const handleConnect = async (id: string) => {
    try {
      await fetch(`/api/admin/mcp-servers/${id}/connect`, { method: "POST" });
      await fetchServers();
    } catch (e) {
      console.error("Connect failed:", e);
    }
  };

  const handleDisconnect = async (id: string) => {
    try {
      await fetch(`/api/admin/mcp-servers/${id}/disconnect`, { method: "POST" });
      await fetchServers();
    } catch (e) {
      console.error("Disconnect failed:", e);
    }
  };

  if (loading) {
    return <div className="text-gray-500">Loading MCP servers...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Tools (MCP Servers)</h2>
          <p className="text-sm text-gray-500">
            Connect to external MCP servers to give your personas access to tools and knowledge bases.
          </p>
        </div>
        {!editing && (
          <button
            onClick={handleAdd}
            className="px-4 py-2 bg-[#76b900] text-white rounded hover:bg-[#5a8f00] text-sm font-medium"
          >
            + Add Server
          </button>
        )}
      </div>

      {editing && (
        <ServerEditor
          server={editing}
          onChange={setEditing}
          onSave={handleSave}
          onCancel={() => {
            setEditing(null);
            setTestResult(null);
          }}
          onTest={handleTest}
          isNew={isNew}
          saving={saving}
          testing={testing}
          testResult={testResult}
        />
      )}

      <div className="space-y-3">
        {servers.map((server) => (
          <div
            key={server.id}
            className="bg-white rounded-lg border border-gray-200 p-4 hover:border-gray-300 transition-colors"
          >
            <div className="flex justify-between items-start">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <div
                    className={`w-2.5 h-2.5 rounded-full ${
                      server.connected ? "bg-[#76b900]" : "bg-gray-300"
                    }`}
                  />
                  <h3 className="font-medium text-gray-900">{server.name}</h3>
                  <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full uppercase">
                    {server.transport}
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      server.connected
                        ? "bg-green-100 text-green-700"
                        : "bg-gray-100 text-gray-500"
                    }`}
                  >
                    {server.connected ? "Connected" : "Disconnected"}
                  </span>
                </div>
                <p className="text-xs text-gray-400 font-mono truncate">
                  {server.transport === "sse"
                    ? server.url
                    : `${server.command} ${(server.args || []).join(" ")}`}
                </p>

                {server.tools && server.tools.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {server.tools.map((tool) => (
                      <span
                        key={tool.name}
                        className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full border border-blue-100"
                        title={tool.description}
                      >
                        {tool.name}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex gap-2 ml-4 shrink-0">
                {server.connected ? (
                  <button
                    onClick={() => handleDisconnect(server.id)}
                    className="px-3 py-1.5 text-xs bg-white text-orange-600 border border-orange-200 rounded hover:bg-orange-50"
                  >
                    Disconnect
                  </button>
                ) : (
                  <button
                    onClick={() => handleConnect(server.id)}
                    className="px-3 py-1.5 text-xs bg-white text-[#76b900] border border-[#76b900] rounded hover:bg-green-50"
                  >
                    Connect
                  </button>
                )}
                <button
                  onClick={() => handleEdit(server)}
                  className="px-3 py-1.5 text-xs bg-white text-gray-700 border border-gray-300 rounded hover:bg-gray-50"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleDelete(server.id)}
                  className="px-3 py-1.5 text-xs bg-white text-red-600 border border-red-200 rounded hover:bg-red-50"
                >
                  Remove
                </button>
              </div>
            </div>
          </div>
        ))}
        {servers.length === 0 && !editing && (
          <div className="text-center py-12 text-gray-400">
            <p className="mb-2">No MCP servers configured.</p>
            <p className="text-xs">
              Add an MCP server to give your personas access to external tools and knowledge.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
