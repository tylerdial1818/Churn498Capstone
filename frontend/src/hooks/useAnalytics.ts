import { useQuery } from "@tanstack/react-query";
import api from "@/lib/api";
import type {
  AnalyticsOverview,
  TrendPoint,
  SegmentBreakdown,
  ModelMetrics,
  DriftStatus,
  SHAPFeature,
} from "@/types/api";

export function useAnalyticsOverview() {
  return useQuery<AnalyticsOverview>({
    queryKey: ["analytics", "overview"],
    queryFn: () => api.get("/analytics/overview").then((r) => r.data),
  });
}

export function useChurnTrends(months = 12) {
  return useQuery<TrendPoint[]>({
    queryKey: ["analytics", "churn-trends", months],
    queryFn: () => api.get("/analytics/churn-trends", { params: { months } }).then((r) => r.data),
  });
}

export function useSegments() {
  return useQuery<SegmentBreakdown>({
    queryKey: ["analytics", "segments"],
    queryFn: () => api.get("/analytics/segments").then((r) => r.data),
  });
}

export function useModelPerformance() {
  return useQuery<ModelMetrics>({
    queryKey: ["analytics", "model-performance"],
    queryFn: () => api.get("/analytics/model-performance").then((r) => r.data),
  });
}

export function useDriftStatus() {
  return useQuery<DriftStatus>({
    queryKey: ["analytics", "drift"],
    queryFn: () => api.get("/analytics/drift").then((r) => r.data),
  });
}

export function useSHAPGlobal() {
  return useQuery<SHAPFeature[]>({
    queryKey: ["analytics", "shap-global"],
    queryFn: () => api.get("/analytics/shap-global").then((r) => r.data),
  });
}
