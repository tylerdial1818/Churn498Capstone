import { useQuery } from "@tanstack/react-query";
import type { AxiosResponse } from "axios";
import api from "@/lib/api";
import type {
  KPIResponse,
  TrendPoint,
  RiskDistribution,
  ActiveInactiveDistribution,
  AgentInsight,
} from "@/types/api";

export function useKPIs() {
  return useQuery<KPIResponse>({
    queryKey: ["dashboard", "kpis"],
    queryFn: () => api.get("/dashboard/kpis").then((r: AxiosResponse) => r.data),
  });
}

export function useTrends() {
  return useQuery<TrendPoint[]>({
    queryKey: ["dashboard", "trends"],
    queryFn: () => api.get("/dashboard/trends").then((r: AxiosResponse) => r.data),
  });
}

export function useRiskDistribution() {
  return useQuery<RiskDistribution>({
    queryKey: ["dashboard", "risk-distribution"],
    queryFn: () => api.get("/dashboard/risk-distribution").then((r: AxiosResponse) => r.data),
  });
}

export function useActiveInactive() {
  return useQuery<ActiveInactiveDistribution>({
    queryKey: ["dashboard", "active-inactive"],
    queryFn: () => api.get("/dashboard/active-inactive").then((r: AxiosResponse) => r.data),
  });
}

export function useExecutiveSummary() {
  return useQuery<AgentInsight>({
    queryKey: ["dashboard", "executive-summary"],
    queryFn: () => api.get("/dashboard/executive-summary").then((r: AxiosResponse) => r.data),
  });
}
