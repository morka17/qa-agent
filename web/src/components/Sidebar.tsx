import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/runs", label: "Runs" },
  { to: "/stories", label: "Stories" },
  { to: "/bugs", label: "Bugs" },
  { to: "/settings", label: "Settings" },
];

export function Sidebar() {
  return (
    <aside
      style={{
        width: "var(--sidebar-width)",
        flexShrink: 0,
        borderRight: "1px solid var(--border)",
        background: "var(--surface)",
        display: "flex",
        flexDirection: "column",
        padding: "var(--space-5) var(--space-4)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--space-2)",
          marginBottom: "var(--space-6)",
          paddingLeft: "var(--space-2)",
        }}
      >
        <SentinelMark />
        <span style={{ fontWeight: 600, fontSize: 15, letterSpacing: "-0.01em" }}>Sentinel-QA</span>
      </div>

      <nav style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            style={({ isActive }) => ({
              display: "block",
              padding: "var(--space-2) var(--space-3)",
              borderRadius: "var(--radius-sm)",
              color: isActive ? "var(--text)" : "var(--text-muted)",
              background: isActive ? "var(--surface-raised)" : "transparent",
              fontSize: 13.5,
              fontWeight: isActive ? 600 : 500,
              textDecoration: "none",
            })}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

/** A simple watchful-eye mark — the one graphic flourish, standing in for a logo. */
function SentinelMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M2 12C4.5 6 8 3 12 3s7.5 3 10 9c-2.5 6-6 9-10 9s-7.5-3-10-9Z"
        stroke="var(--signal-active)"
        strokeWidth="1.6"
      />
      <circle cx="12" cy="12" r="3.2" fill="var(--signal-active)" />
    </svg>
  );
}