import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "@/components/layout/Sidebar";
import Dashboard from "@/pages/Dashboard";
import AtRiskAccounts from "@/pages/AtRiskAccounts";
import Analytics from "@/pages/Analytics";
import Prescriptions from "@/pages/Prescriptions";
import Interventions from "@/pages/Interventions";

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/at-risk" element={<AtRiskAccounts />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/prescriptions" element={<Prescriptions />} />
          <Route path="/interventions" element={<Interventions />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
