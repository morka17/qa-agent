import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { Field, inputStyle } from "@/components/Field";
import { Button } from "@/components/Button";
import { submitStory, triggerRun, ApiError, type Priority } from "@/api/client";

const PRIORITIES: Priority[] = ["critical", "high", "medium", "low"];

export function StoriesPage() {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [narrative, setNarrative] = useState("");
  const [priority, setPriority] = useState<Priority>("medium");
  const [targetUrl, setTargetUrl] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [justTriggeredJobId, setJustTriggeredJobId] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setJustTriggeredJobId(null);

    try {
      // Validate/normalize the story first (POST /stories), then trigger
      // a run from the same payload - matches the API's separation
      // between "define a story" and "run a story".
      await submitStory({
        title,
        narrative,
        priority,
        labels: [],
        target_url: targetUrl || null,
      });

      const { job_id } = await triggerRun({
        title,
        narrative,
        priority,
        target_url: targetUrl || null,
      });

      setJustTriggeredJobId(job_id);
      setTitle("");
      setNarrative("");
      setTargetUrl("");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Something went wrong submitting the story.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Layout title="Stories">
      <div style={{ maxWidth: 560 }}>
        <p style={{ color: "var(--text-muted)", marginTop: 0, marginBottom: "var(--space-5)" }}>
          Describe a user story in plain language. Sentinel plans a test, runs it against your app, and
          files a bug if anything doesn't hold up.
        </p>

        {justTriggeredJobId && (
          <div
            style={{
              marginBottom: "var(--space-5)",
              padding: "var(--space-3)",
              borderRadius: "var(--radius-sm)",
              background: "var(--signal-pass-bg)",
              color: "var(--signal-pass)",
              fontSize: 13.5,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span>Run triggered.</span>
            <button
              onClick={() => navigate("/runs")}
              style={{
                background: "none",
                border: "none",
                color: "var(--signal-pass)",
                fontWeight: 600,
                cursor: "pointer",
                fontSize: 13.5,
                textDecoration: "underline",
              }}
            >
              View in Runs
            </button>
          </div>
        )}

        {error && (
          <div
            style={{
              marginBottom: "var(--space-5)",
              padding: "var(--space-3)",
              borderRadius: "var(--radius-sm)",
              background: "var(--signal-fail-bg)",
              color: "var(--signal-fail)",
              fontSize: 13.5,
            }}
          >
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <Field label="Title">
            <input
              style={inputStyle}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Add item to cart"
              required
            />
          </Field>

          <Field
            label="Story"
            hint='"As a logged-in customer, I want to add an item to my cart, so that I can buy it later." Given/When/Then acceptance criteria are also understood.'
          >
            <textarea
              style={{ ...inputStyle, minHeight: 120, resize: "vertical", fontFamily: "var(--font-ui)" }}
              value={narrative}
              onChange={(e) => setNarrative(e.target.value)}
              required
            />
          </Field>

          <Field label="Target URL" hint="The page or app this story should be tested against.">
            <input
              style={inputStyle}
              type="url"
              value={targetUrl}
              onChange={(e) => setTargetUrl(e.target.value)}
              placeholder="https://staging.example.com/cart"
            />
          </Field>

          <Field label="Priority">
            <select
              style={inputStyle}
              value={priority}
              onChange={(e) => setPriority(e.target.value as Priority)}
            >
              {PRIORITIES.map((p) => (
                <option key={p} value={p}>
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </option>
              ))}
            </select>
          </Field>

          <Button type="submit" variant="primary" disabled={submitting}>
            {submitting ? "Submitting…" : "Submit and run"}
          </Button>
        </form>
      </div>
    </Layout>
  );
}