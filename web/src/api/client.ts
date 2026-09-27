/**
 * Typed client for the Sentinel-QA control-plane API
 * (`qa_agent/api/app.py` and its routers). Every type here mirrors a
 * pydantic response model on the backend exactly — when a router's
 * response_model changes, this file is the one place on the frontend
 * that needs to change with it.
 *
 * Requests go through `/api/*`, proxied to the FastAPI backend by
 * `vite.config.ts` in development; in production this is served behind
 * the same origin or reverse-proxied identically.
 */

const BASE_URL = "/api";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API error ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON — fall back to statusText, set above
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

// --- Shared enums (mirror qa_agent.ingestion.schemas / triage / reporting) ---

export type Priority = "critical" | "high" | "medium" | "low";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type Severity = "critical" | "high" | "medium" | "low" | "info";
export type FailureCategory =
  | "app_bug"
  | "flaky_test"
  | "selector_drift"
  | "environment_issue"
  | "plan_error"
  | "unknown";

// --- Stories ---

export interface StorySubmission {
  title: string;
  narrative: string;
  priority: Priority;
  labels: string[];
  target_url: string | null;
}

export interface StoryResponse {
  id: string;
  external_id: string | null;
  source: string;
  title: string;
  narrative: string;
  priority: Priority;
  labels: string[];
  target_url: string | null;
}

export function submitStory(submission: StorySubmission): Promise<StoryResponse> {
  return request<StoryResponse>("/stories", {
    method: "POST",
    body: JSON.stringify(submission),
  });
}

// --- Runs ---

export interface RunTriggerRequest {
  title: string;
  narrative: string;
  priority: Priority;
  target_url: string | null;
}

export interface RunTriggerResponse {
  job_id: string;
  status: JobStatus;
}

export interface RunStatusResponse {
  job_id: string;
  status: JobStatus;
  run_id: string | null;
  final_state: string | null;
  bug_report_id: string | null;
  error: string | null;
}

export interface RunSummary {
  id: string;
  story_id: string | null;
  target_url: string | null;
  state: string;
  created_at: string;
  finished_at: string | null;
}

export function triggerRun(request_: RunTriggerRequest): Promise<RunTriggerResponse> {
  return request<RunTriggerResponse>("/runs", {
    method: "POST",
    body: JSON.stringify(request_),
  });
}

export function getRunStatus(jobId: string): Promise<RunStatusResponse> {
  return request<RunStatusResponse>(`/runs/${jobId}/status`);
}

export function listRuns(params?: { state?: string; limit?: number; offset?: number }): Promise<RunSummary[]> {
  const query = new URLSearchParams();
  if (params?.state) query.set("state", params.state);
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));
  const qs = query.toString();
  return request<RunSummary[]>(`/runs${qs ? `?${qs}` : ""}`);
}

export function getRun(runId: string): Promise<RunSummary> {
  return request<RunSummary>(`/runs/${runId}`);
}

// --- Bugs ---

export interface BugSummary {
  id: string;
  run_id: string;
  title: string;
  severity: Severity;
  category: FailureCategory;
  occurrence_count: number;
  tracker_name: string | null;
  tracker_ref: string | null;
  tracker_url: string | null;
  created_at: string;
}

export interface BugDetail extends BugSummary {
  summary: string;
  description: string;
  expected_behavior: string;
  actual_behavior: string;
  labels: string[];
  cluster_id: string | null;
  filed_at: string | null;
}

export function listBugs(params?: { severity?: string; category?: string; limit?: number; offset?: number }): Promise<BugSummary[]> {
  const query = new URLSearchParams();
  if (params?.severity) query.set("severity", params.severity);
  if (params?.category) query.set("category", params.category);
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));
  const qs = query.toString();
  return request<BugSummary[]>(`/bugs${qs ? `?${qs}` : ""}`);
}

export function getBug(bugId: string): Promise<BugDetail> {
  return request<BugDetail>(`/bugs/${bugId}`);
}

// --- Health ---

export function getHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("/health");
}