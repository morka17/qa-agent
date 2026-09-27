import { useEffect, useState, useCallback } from "react";
import type { CSSProperties } from "react";
import { Layout } from "@/components/Layout";
import { StatusBadge } from "@/components/StatusBadge";
import { EmptyState } from "@/components/EmptyState";
import { listRuns, type RunSummary } from "@/api/client";

const ACTIVE_STATES = new Set(["ingesting", "planning", "executing", "verifying", "triaging", "reporting"]);

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(start: string, end: string | null): string {
  const startMs = new Date(start).getTime();
  const endMs = end ? new Date(end).getTime() : Date.now();
  const seconds = Math.round((endMs - startMs) / 1000);
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await listRuns({ limit: 100 });
      setRuns(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load runs.");
    }
  }, []);

  useEffect(() => {
    load();
    // Poll while any run is active, so a currently-executing run's status
    // visibly progresses without a manual refresh.
    const interval = setInterval(load, 4000);
    return () => clearInterval(interval);
  }, [load]);

  return (
    <Layout title="Runs">
      {error && (
        <div
          style={{
            marginBottom: "var(--space-4)",
            padding: "var(--space-3)",
            borderRadius: "var(--radius-sm)",
            background: "var(--signal-fail-bg)",
            color: "var(--signal-fail)",
            fontSize: 13,
          }}
        >
          Couldn't load runs: {error}
        </div>
      )}

      {runs === null && !error && <p style={{ color: "var(--text-muted)" }}>Loading runs…</p>}

      {runs !== null && runs.length === 0 && (
        <EmptyState
          title="No runs yet"
          description="Submit a story and trigger a run to see it appear here."
        />
      )}

      {runs !== null && runs.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 12.5 }}>
              <th style={headerCell}>Run</th>
              <th style={headerCell}>Target</th>
              <th style={headerCell}>State</th>
              <th style={headerCell}>Started</th>
              <th style={headerCell}>Duration</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const isActive = ACTIVE_STATES.has(run.state);
              return (
                <tr
                  key={run.id}
                  style={{
                    borderTop: "1px solid var(--border)",
                    borderLeft: isActive ? "2px solid var(--signal-active)" : "2px solid transparent",
                  }}
                >
                  <td style={{ ...cell, fontFamily: "var(--font-mono)", fontSize: 12.5 }}>
                    {run.id.slice(0, 8)}
                  </td>
                  <td style={{ ...cell, color: "var(--text-muted)" }}>{run.target_url ?? "—"}</td>
                  <td style={cell}>
                    <StatusBadge value={run.state} isActive={isActive} />
                  </td>
                  <td style={{ ...cell, color: "var(--text-muted)" }}>{formatTimestamp(run.created_at)}</td>
                  <td style={{ ...cell, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {formatDuration(run.created_at, run.finished_at)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Layout>
  );
}

const headerCell: CSSProperties = {
  padding: "var(--space-2) var(--space-3)",
  fontWeight: 500,
};

const cell: CSSProperties = {
  padding: "var(--space-3)",
  fontSize: 13.5,
};