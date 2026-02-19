import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "@/lib/api";
import type { InterventionDraft } from "@/types/api";

export function useInterventions(status?: string) {
  return useQuery<InterventionDraft[]>({
    queryKey: ["interventions", { status }],
    queryFn: () =>
      api.get("/interventions", { params: status ? { status } : {} }).then((r) => r.data),
  });
}

export function useIntervention(id: string) {
  return useQuery<InterventionDraft>({
    queryKey: ["interventions", id],
    queryFn: () => api.get(`/interventions/${id}`).then((r) => r.data),
    enabled: !!id,
  });
}

export function useDraftIntervention() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { account_id: string; strategy: string }) =>
      api.post("/interventions/draft", body).then((r) => r.data as InterventionDraft),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["interventions"] }),
  });
}

export function useBatchDraft() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { account_ids: string[]; strategy: string }) =>
      api.post("/interventions/batch-draft", body).then((r) => r.data as InterventionDraft[]),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["interventions"] }),
  });
}

export function useUpdateInterventionStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      api.patch(`/interventions/${id}/status`, { status }).then((r) => r.data as InterventionDraft),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["interventions"] }),
  });
}

export function useUpdateInterventionContent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, subject, body }: { id: string; subject: string; body: string }) =>
      api.patch(`/interventions/${id}/content`, { subject, body }).then((r) => r.data as InterventionDraft),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["interventions"] }),
  });
}
