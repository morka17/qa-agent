import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import { Layout } from "@/components/Layout";
import { StatusBadge } from "@/components/StatusBadge";
import { EmptyState } from "@/components/EmptyState";
import { listBugs, getBug, type BugSummary, type BugDetail } from "@/api/client";

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function BugsPage() {
  const [bugs, setBugs] = useState<BugSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<BugDetail | null>(null);

  useEffect(() => {
    listBugs({ limit: 100 })
      .then(setBugs)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load bugs."));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    getBug(selectedId).then(setDetail);
  }, [selectedId]);

  return (
    <Layout title="Bugs">
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
          Couldn't load bugs: {error}
        </div>
      )}

      {bugs === null && !error && <p style={{ color: "var(--text-muted)" }}>Loading bugs…</p>}

      {bugs !== null && bugs.length === 0 && (
        <EmptyState
          title="No bugs filed"
          description="When a run finds a real issue, it appears here with full repro evidence."
        />
      )}

      {bugs !== null && bugs.length > 0 && (
        <div style={{ display: "flex", gap: "var(--space-6)", alignItems: "flex-start" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", flex: 1 }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 12.5 }}>
                <th style={headerCell}>Title</th>
                <th style={headerCell}>Severity</th>
                <th style={headerCell}>Category</th>
                <th style={headerCell}>Seen</th>
                <th style={headerCell}>Filed</th>
              </tr>
            </thead>
            <tbody>
              {bugs.map((bug) => (
                <tr
                  key={bug.id}
                  onClick={() => setSelectedId(bug.id)}
                  style={{
                    borderTop: "1px solid var(--border)",
                    cursor: "pointer",
                    background: selectedId === bug.id ? "var(--surface-raised)" : "transparent",
                  }}
                >
                  <td style={{ ...cell, fontWeight: 500 }}>{bug.title}</td>
                  <td style={cell}>
                    <StatusBadge value={bug.severity} />
                  </td>
                  <td style={cell}>
                    <StatusBadge value={bug.category} />
                  </td>
                  <td style={{ ...cell, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    ×{bug.occurrence_count}
                  </td>
                  <td style={{ ...cell, color: "var(--text-muted)" }}>
                    {bug.tracker_ref ? (
                      bug.tracker_url ? (
                        <a href={bug.tracker_url} target="_blank" rel="noreferrer">
                          {bug.tracker_ref}
                        </a>
                      ) : (
                        bug.tracker_ref
                      )
                    ) : (
                      "Not filed"
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {detail && (
            <aside
              style={{
                width: 340,
                flexShrink: 0,
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                padding: "var(--space-5)",
                background: "var(--surface)",
              }}
            >
              <h2 style={{ fontSize: 15, marginBottom: "var(--space-3)" }}>{detail.title}</h2>
              <div style={{ display: "flex", gap: "var(--space-2)", marginBottom: "var(--space-4)" }}>
                <StatusBadge value={detail.severity} />
                <StatusBadge value={detail.category} />
              </div>

              <DetailSection label="Summary" text={detail.summary} />
              <DetailSection label="Expected" text={detail.expected_behavior} />
              <DetailSection label="Actual" text={detail.actual_behavior} />

              <p style={{ fontSize: 12, color: "var(--text-faint)", marginTop: "var(--space-5)" }}>
                Filed {detail.filed_at ? formatTimestamp(detail.filed_at) : "not yet"} · run{" "}
                <span className="mono">{detail.run_id.slice(0, 8)}</span>
              </p>
            </aside>
          )}
        </div>
      )}
    </Layout>
  );
}

function DetailSection({ label, text }: { label: string; text: string }) {
  return (
    <div style={{ marginBottom: "var(--space-4)" }}>
      <p style={{ fontSize: 12, fontWeight: 600, color: "var(--text-muted)", margin: "0 0 4px" }}>{label}</p>
      <p style={{ fontSize: 13.5, margin: 0 }}>{text}</p>
    </div>
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