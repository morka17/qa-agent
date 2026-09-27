import type { ReactNode } from "react";

export interface EmptyStateProps {
  title: string;
  description: string;
  action?: ReactNode;
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div
      style={{
        border: "1px dashed var(--border-strong)",
        borderRadius: "var(--radius-md)",
        padding: "var(--space-7) var(--space-5)",
        textAlign: "center",
        color: "var(--text-muted)",
      }}
    >
      <p style={{ color: "var(--text)", fontWeight: 600, margin: "0 0 var(--space-2)" }}>{title}</p>
      <p style={{ margin: "0 0 var(--space-4)", fontSize: 13.5 }}>{description}</p>
      {action}
    </div>
  );
}