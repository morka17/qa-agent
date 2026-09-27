import type { CSSProperties, ReactNode } from "react";

export interface FieldProps {
  label: string;
  hint?: string;
  children: ReactNode;
}

export function Field({ label, hint, children }: FieldProps) {
  return (
    <label style={{ display: "block", marginBottom: "var(--space-4)" }}>
      <span style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: "var(--space-2)" }}>
        {label}
      </span>
      {children}
      {hint && (
        <span style={{ display: "block", fontSize: 12, color: "var(--text-muted)", marginTop: "var(--space-1)" }}>
          {hint}
        </span>
      )}
    </label>
  );
}

export const inputStyle: CSSProperties = {
  width: "100%",
  background: "var(--surface)",
  border: "1px solid var(--border-strong)",
  borderRadius: "var(--radius-sm)",
  color: "var(--text)",
  padding: "var(--space-2) var(--space-3)",
  fontSize: 13.5,
  fontFamily: "var(--font-ui)",
};