import { useQuery } from "@tanstack/react-query";
import type { AxiosResponse } from "axios";
import api from "@/lib/api";
import type { PrescriptionGroup } from "@/types/api";

export function usePrescriptions() {
  return useQuery<PrescriptionGroup[]>({
    queryKey: ["prescriptions"],
    queryFn: () => api.get("/prescriptions").then((r: AxiosResponse) => r.data),
  });
}

export function usePrescriptionByStrategy(strategy: string) {
  return useQuery<PrescriptionGroup>({
    queryKey: ["prescriptions", strategy],
    queryFn: () => api.get(`/prescriptions/${strategy}`).then((r: AxiosResponse) => r.data),
    enabled: !!strategy,
  });
}
