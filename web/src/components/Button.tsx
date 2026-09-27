import type { ButtonHTMLAttributes } from "react";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
}

export function Button({ variant = "secondary", style, ...rest }: ButtonProps) {
  const base = {
    fontSize: 13.5,
    fontWeight: 600,
    padding: "var(--space-2) var(--space-4)",
    borderRadius: "var(--radius-sm)",
    border: "1px solid var(--border-strong)",
    cursor: rest.disabled ? "not-allowed" : "pointer",
    opacity: rest.disabled ? 0.5 : 1,
    transition: "background 0.12s ease",
  };

  const variantStyle =
    variant === "primary"
      ? { background: "var(--signal-active)", color: "#06222c", border: "1px solid var(--signal-active)" }
      : { background: "var(--surface-raised)", color: "var(--text)" };

  return <button {...rest} style={{ ...base, ...variantStyle, ...style }} />;
}