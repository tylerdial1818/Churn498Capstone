import { useState } from "react";
import PageShell from "@/components/layout/PageShell";
import FilterBar from "@/components/data-display/FilterBar";
import DataTable from "@/components/data-display/DataTable";
import RiskBadge from "@/components/data-display/RiskBadge";
import LoadingSkeleton from "@/components/data-display/LoadingSkeleton";
import ChartCard from "@/components/data-display/ChartCard";
import SHAPWaterfall from "@/components/charts/SHAPWaterfall";
import { useAtRiskAccounts, useAccountDetail, useAccountSHAP } from "@/hooks/useAccounts";
import type { AccountAtRisk } from "@/types/api";
import type { Column } from "@/components/data-display/DataTable";

const RISK_FILTERS = [
  { label: "All", value: "" },
  { label: "High", value: "high" },
  { label: "Medium", value: "medium" },
];

const COLUMNS: Column<AccountAtRisk>[] = [
  {
    key: "account_id",
    header: "Account",
    render: (r) => (
      <div>
        <span className="font-medium text-text-primary">{r.account_id}</span>
        <p className="text-xs text-text-muted">{r.email}</p>
      </div>
    ),
  },
  {
    key: "churn_probability",
    header: "Churn Prob",
    render: (r) => (
      <span className="font-mono text-sm">{(r.churn_probability * 100).toFixed(1)}%</span>
    ),
  },
  {
    key: "risk_tier",
    header: "Risk",
    render: (r) => <RiskBadge tier={r.risk_tier} />,
  },
  {
    key: "plan_type",
    header: "Plan",
    render: (r) => <span className="text-text-secondary">{r.plan_type}</span>,
  },
  {
    key: "tenure_days",
    header: "Tenure",
    render: (r) => <span className="text-text-secondary">{r.tenure_days}d</span>,
  },
  {
    key: "top_drivers",
    header: "Top Drivers",
    render: (r) => (
      <div className="flex flex-wrap gap-1">
        {r.top_drivers.slice(0, 2).map((d) => (
          <span key={d} className="rounded bg-surface-hover px-2 py-0.5 text-xs text-text-secondary">
            {d.replace(/_/g, " ")}
          </span>
        ))}
      </div>
    ),
  },
];

export default function AtRiskAccounts() {
  const [riskFilter, setRiskFilter] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  const { data, isLoading } = useAtRiskAccounts({
    risk_tier: riskFilter || undefined,
    page,
    per_page: 15,
  });

  const detail = useAccountDetail(selectedId ?? "");
  const shap = useAccountSHAP(selectedId ?? "");

  return (
    <PageShell
      title="At-Risk Accounts"
      subtitle="Subscribers flagged by the churn model"
      actions={
        <FilterBar options={RISK_FILTERS} value={riskFilter} onChange={setRiskFilter} />
      }
    >
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Table */}
        <div className="lg:col-span-2">
          {isLoading ? (
            <LoadingSkeleton variant="row" count={8} />
          ) : data ? (
            <>
              <DataTable
                columns={COLUMNS}
                data={data.items}
                keyExtractor={(r) => r.account_id}
                onRowClick={(r) => setSelectedId(r.account_id)}
                emptyMessage="No at-risk accounts found"
              />
              {/* Pagination */}
              {data.total > data.per_page && (
                <div className="mt-4 flex items-center justify-between text-xs text-text-muted">
                  <span>
                    Showing {(page - 1) * data.per_page + 1}–
                    {Math.min(page * data.per_page, data.total)} of {data.total}
                  </span>
                  <div className="flex gap-2">
                    <button
                      disabled={page <= 1}
                      onClick={() => setPage((p) => p - 1)}
                      className="rounded-button border border-border px-3 py-1.5 text-text-secondary transition-colors hover:bg-surface-hover disabled:opacity-40"
                    >
                      Prev
                    </button>
                    <button
                      disabled={page * data.per_page >= data.total}
                      onClick={() => setPage((p) => p + 1)}
                      className="rounded-button border border-border px-3 py-1.5 text-text-secondary transition-colors hover:bg-surface-hover disabled:opacity-40"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : null}
        </div>

        {/* Detail Sidebar */}
        <div className="space-y-4">
          {selectedId ? (
            <>
              {detail.data && (
                <div className="rounded-card border border-border bg-surface p-card">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-medium text-text-primary">{detail.data.account_id}</h3>
                      <p className="text-xs text-text-muted">{detail.data.email}</p>
                    </div>
                    <RiskBadge tier={detail.data.risk_tier} />
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="text-text-muted">Churn Prob</span>
                      <p className="font-mono font-medium text-text-primary">
                        {(detail.data.churn_probability * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <span className="text-text-muted">Plan</span>
                      <p className="font-medium text-text-primary">{detail.data.plan_type}</p>
                    </div>
                    <div>
                      <span className="text-text-muted">Tenure</span>
                      <p className="font-medium text-text-primary">{detail.data.tenure_days} days</p>
                    </div>
                    <div>
                      <span className="text-text-muted">Last Stream</span>
                      <p className="font-medium text-text-primary">{detail.data.last_stream_days}d ago</p>
                    </div>
                    <div>
                      <span className="text-text-muted">Watch (30d)</span>
                      <p className="font-medium text-text-primary">{detail.data.watch_hours_30d}h</p>
                    </div>
                    <div>
                      <span className="text-text-muted">Sessions (30d)</span>
                      <p className="font-medium text-text-primary">{detail.data.sessions_30d}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* SHAP explanation */}
              {shap.data && (
                <ChartCard title="SHAP Explanation" subtitle="What's driving this risk score">
                  <SHAPWaterfall features={shap.data} maxFeatures={8} />
                </ChartCard>
              )}

              {/* Recent Payments */}
              {detail.data?.recent_payments && detail.data.recent_payments.length > 0 && (
                <div className="rounded-card border border-border bg-surface p-card">
                  <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
                    Recent Payments
                  </h4>
                  <div className="space-y-2">
                    {detail.data.recent_payments.map((p, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-text-secondary">{p.date}</span>
                        <span className="text-text-secondary">${p.amount}</span>
                        <span
                          className={
                            p.status === "paid"
                              ? "text-success"
                              : p.status === "failed"
                                ? "text-risk-high"
                                : "text-text-muted"
                          }
                        >
                          {p.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recent Tickets */}
              {detail.data?.recent_tickets && detail.data.recent_tickets.length > 0 && (
                <div className="rounded-card border border-border bg-surface p-card">
                  <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
                    Support Tickets
                  </h4>
                  <div className="space-y-2">
                    {detail.data.recent_tickets.map((t) => (
                      <div key={t.id} className="flex items-center justify-between text-xs">
                        <span className="truncate text-text-secondary">{t.subject}</span>
                        <span
                          className={
                            t.status === "open"
                              ? "text-warning"
                              : t.status === "resolved"
                                ? "text-success"
                                : "text-text-muted"
                          }
                        >
                          {t.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex h-64 items-center justify-center rounded-card border border-dashed border-border bg-surface">
              <p className="text-sm text-text-muted">Select an account to view details</p>
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
