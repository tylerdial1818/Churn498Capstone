import { useQuery } from "@tanstack/react-query";
import api from "@/lib/api";
import type { PaginatedAccounts, AccountDetail, SHAPFeature } from "@/types/api";

interface AtRiskParams {
  risk_tier?: string;
  sort_by?: string;
  page?: number;
  per_page?: number;
}

export function useAtRiskAccounts(params: AtRiskParams = {}) {
  return useQuery<PaginatedAccounts>({
    queryKey: ["accounts", "at-risk", params],
    queryFn: () => api.get("/accounts/at-risk", { params }).then((r) => r.data),
  });
}

export function useAccountDetail(accountId: string) {
  return useQuery<AccountDetail>({
    queryKey: ["accounts", accountId],
    queryFn: () => api.get(`/accounts/${accountId}`).then((r) => r.data),
    enabled: !!accountId,
  });
}

export function useAccountSHAP(accountId: string) {
  return useQuery<SHAPFeature[]>({
    queryKey: ["accounts", accountId, "shap"],
    queryFn: () => api.get(`/accounts/${accountId}/shap`).then((r) => r.data),
    enabled: !!accountId,
  });
}
