import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";

export interface LayoutProps {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
}

export function Layout({ title, actions, children }: LayoutProps) {
  return (
    <div style={{ display: "flex", height: "100%" }}>
      <Sidebar />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "var(--space-5) var(--space-6)",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <h1 style={{ fontSize: 18 }}>{title}</h1>
          {actions}
        </header>
        <main style={{ flex: 1, overflow: "auto", padding: "var(--space-6)" }}>{children}</main>
      </div>
    </div>
  );
}