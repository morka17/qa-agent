import { Navigate, Route, Routes } from "react-router-dom";
import { RunsPage } from "@/pages/Runs";
import { StoriesPage } from "@/pages/Stories";
import { BugsPage } from "@/pages/Bugs";
import { SettingsPage } from "@/pages/Settings";
import { statusBadgeKeyframes } from "@/components/StatusBadge";

export function App() {
  return (
    <>
      {/* Injected once, globally, for the pulsing "active run" indicator every StatusBadge shares. */}
      <style>{statusBadgeKeyframes}</style>
      <Routes>
        <Route path="/" element={<Navigate to="/runs" replace />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route path="/stories" element={<StoriesPage />} />
        <Route path="/bugs" element={<BugsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/runs" replace />} />
      </Routes>
    </>
  );
}