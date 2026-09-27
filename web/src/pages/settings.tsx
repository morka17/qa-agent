import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Layout } from "@/components/Layout";
import { StatusBadge } from "@/components/StatusBadge";
import { getHealth } from "@/api/client";

/**
 * Read-only view of how this deployment is configured. Sentinel's
 * settings (`qa_agent.config.settings.Settings`) are environment-driven
 * on the backend, not editable through the API — this page reflects
 * that: it surfaces what's configured (for an operator sanity-checking
 * a deployment) without pretending to be a settings editor the backend
 * doesn't currently expose.
 */
export function SettingsPage() {
  const [apiStatus, setApiStatus] = useState<"checking" | "ok" | "unreachable">("checking");

  useEffect(() => {
    getHealth()
      .then(() => setApiStatus("ok"))
      .catch(() => setApiStatus("unreachable"));
  }, []);

  return (
    <Layout title="Settings">
      <div style={{ maxWidth: 640, display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
        <Section title="Control plane">
          <Row label="API connection">
            {apiStatus === "checking" && <span style={{ color: "var(--text-muted)" }}>Checking…</span>}
            {apiStatus === "ok" && <StatusBadge value="passed" />}
            {apiStatus === "unreachable" && <StatusBadge value="failed" />}
          </Row>
        </Section>

        <Section title="About these settings">
          <p style={{ fontSize: 13.5, color: "var(--text-muted)", margin: 0, lineHeight: 1.6 }}>
            Sentinel-QA's runtime configuration — LLM provider, issue tracker credentials, Playwright
            browser, guardrail thresholds — is environment-driven on the backend
            (<span className="mono" style={{ fontSize: 12.5 }}>qa_agent.config.settings.Settings</span>)
            and isn't editable from this dashboard. To change it, update the deployment's environment
            variables or <span className="mono" style={{ fontSize: 12.5 }}>.env</span> file and restart
            the control plane.
          </p>
        </Section>
      </div>
    </Layout>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h2 style={{ fontSize: 14, marginBottom: "var(--space-3)", color: "var(--text-muted)" }}>{title}</h2>
      <div
        style={{
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          background: "var(--surface)",
          padding: "var(--space-4) var(--space-5)",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 13.5 }}>{label}</span>
      {children}
    </div>
  );
}