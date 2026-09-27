import type { CSSProperties } from "react";

/**
 * Every status shown anywhere in the dashboard — run state, job status,
 * bug severity — renders through this one component, so "what does this
 * color mean" only has one answer across the whole app. Color is never
 * the only signal: each badge pairs a dot, a word, and (for active runs)
 * motion.
 */

type SignalKind = "pass" | "fail" | "watch" | "active" | "neutral";

const STATE_TO_SIGNAL: Record<string, SignalKind> = {
  passed: "pass",
  succeeded: "pass",
  failed: "fail",
  error: "fail",
  cancelled: "neutral",
  pending: "neutral",
  ingesting: "active",
  planning: "active",
  awaiting_approval: "watch",
  executing: "active",
  verifying: "active",
  triaging: "active",
  reporting: "active",
  running: "active",
  queued: "neutral",
  // failure categories (bugs)
  app_bug: "fail",
  flaky_test: "watch",
  selector_drift: "watch",
  environment_issue: "watch",
  plan_error: "neutral",
  unknown: "neutral",
  // severities
  critical: "fail",
  high: "watch",
  medium: "neutral",
  low: "neutral",
  info: "neutral",
};

function labelFor(value: string): string {
  return value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export interface StatusBadgeProps {
  value: string;
  /** Runs currently in-progress get the pulsing indicator — the dashboard's one deliberate motion. */
  isActive?: boolean;
}

export function StatusBadge({ value, isActive = false }: StatusBadgeProps) {
  const signal = STATE_TO_SIGNAL[value] ?? "neutral";
  const style: CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: "var(--space-2)",
    padding: "3px var(--space-2) 3px var(--space-1)",
    borderRadius: "var(--radius-sm)",
    fontSize: "12.5px",
    fontWeight: 500,
    color: `var(--signal-${signal})`,
    background: `var(--signal-${signal}-bg)`,
    lineHeight: 1,
  };

  return (
    <span style={style}>
      <Dot signal={signal} pulsing={isActive && signal === "active"} />
      {labelFor(value)}
    </span>
  );
}

function Dot({ signal, pulsing }: { signal: SignalKind; pulsing: boolean }) {
  return (
    <span
      aria-hidden="true"
      style={{
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: `var(--signal-${signal})`,
        display: "inline-block",
        animation: pulsing ? "sentinel-pulse 1.6s ease-in-out infinite" : undefined,
      }}
    />
  );
}

/** Injected once, globally, so every StatusBadge's pulsing dot shares one keyframe definition. */
export const statusBadgeKeyframes = `
@keyframes sentinel-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.7); }
}
`;