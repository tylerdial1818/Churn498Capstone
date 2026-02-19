import { useState } from "react";
import { useNavigate } from "react-router-dom";
import PageShell from "@/components/layout/PageShell";
import KPICard from "@/components/data-display/KPICard";
import DataTable from "@/components/data-display/DataTable";
import RiskBadge from "@/components/data-display/RiskBadge";
import LoadingSkeleton from "@/components/data-display/LoadingSkeleton";
import EmptyState from "@/components/data-display/EmptyState";
import { usePrescriptions } from "@/hooks/usePrescriptions";
import { useBatchDraft } from "@/hooks/useInterventions";
import type { AccountAtRisk } from "@/types/api";
import type { Column } from "@/components/data-display/DataTable";

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
    render: (r) => <span className="font-mono text-sm">{(r.churn_probability * 100).toFixed(1)}%</span>,
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
];

export default function Prescriptions() {
  const navigate = useNavigate();
  const { data: groups, isLoading } = usePrescriptions();
  const batchDraft = useBatchDraft();
  const [expandedStrategy, setExpandedStrategy] = useState<string | null>(null);

  if (isLoading) {
    return (
      <PageShell title="Prescriptions" subtitle="AI-recommended retention strategies">
        <LoadingSkeleton variant="card" count={4} />
        <LoadingSkeleton variant="row" count={5} />
      </PageShell>
    );
  }

  if (!groups || groups.length === 0) {
    return (
      <PageShell title="Prescriptions" subtitle="AI-recommended retention strategies">
        <EmptyState
          title="No prescriptions available"
          description="Run the prescription pipeline to generate strategy recommendations."
        />
      </PageShell>
    );
  }

  return (
    <PageShell title="Prescriptions" subtitle="AI-recommended retention strategies">
      {/* Strategy Summary Cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {groups.map((g) => (
          <button
            key={g.strategy}
            onClick={() =>
              setExpandedStrategy(expandedStrategy === g.strategy ? null : g.strategy)
            }
            className={`rounded-card border p-card text-left transition-colors ${
              expandedStrategy === g.strategy
                ? "border-accent/50 bg-accent-muted"
                : "border-border bg-surface hover:border-border hover:bg-surface-hover"
            }`}
          >
            <p className="text-xs font-medium uppercase tracking-wider text-text-muted">
              {g.display_name}
            </p>
            <p className="mt-1 text-xl font-semibold text-text-primary">{g.account_count}</p>
            <p className="text-xs text-text-secondary">
              Est. MRR: ${g.estimated_mrr.toLocaleString()}
            </p>
          </button>
        ))}
      </div>

      {/* Expanded Strategy Detail */}
      {expandedStrategy && (() => {
        const group = groups.find((g) => g.strategy === expandedStrategy);
        if (!group) return null;
        return (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-text-primary">{group.display_name}</h3>
                <p className="text-sm text-text-secondary">
                  {group.account_count} accounts · ${group.estimated_mrr.toLocaleString()} MRR
                </p>
              </div>
              <button
                onClick={() => {
                  const ids = group.accounts.map((a) => a.account_id);
                  batchDraft.mutate(
                    { account_ids: ids, strategy: group.strategy },
                    {
                      onSuccess: () => navigate("/interventions"),
                    },
                  );
                }}
                disabled={batchDraft.isPending}
                className="rounded-button bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-50"
              >
                {batchDraft.isPending ? "Drafting..." : "Draft All Interventions"}
              </button>
            </div>
            <DataTable
              columns={COLUMNS}
              data={group.accounts}
              keyExtractor={(r) => r.account_id}
              emptyMessage="No accounts in this strategy"
            />
          </div>
        );
      })()}

      {/* Totals */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3">
        <KPICard
          label="Total Prescribed"
          value={groups.reduce((s, g) => s + g.account_count, 0).toLocaleString()}
        />
        <KPICard
          label="Total MRR at Risk"
          value={`$${groups.reduce((s, g) => s + g.estimated_mrr, 0).toLocaleString()}`}
          variant="accent"
        />
        <KPICard label="Strategies" value={groups.length} />
      </div>
    </PageShell>
  );
}
