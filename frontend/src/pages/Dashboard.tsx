import PageShell from "@/components/layout/PageShell";
import KPICard from "@/components/data-display/KPICard";
import ChartCard from "@/components/data-display/ChartCard";
import AgentInsightCard from "@/components/data-display/AgentInsightCard";
import LoadingSkeleton from "@/components/data-display/LoadingSkeleton";
import TrendLineChart from "@/components/charts/TrendLineChart";
import RiskDistributionChart from "@/components/charts/RiskDistributionChart";
import DonutChart from "@/components/charts/DonutChart";
import {
  useKPIs,
  useTrends,
  useRiskDistribution,
  useActiveInactive,
  useExecutiveSummary,
} from "@/hooks/useDashboard";

export default function Dashboard() {
  const kpis = useKPIs();
  const trends = useTrends();
  const riskDist = useRiskDistribution();
  const activeInactive = useActiveInactive();
  const summary = useExecutiveSummary();

  return (
    <PageShell title="Dashboard" subtitle="Customer lifecycle overview">
      {/* Executive Summary */}
      {summary.data && (
        <AgentInsightCard
          title={summary.data.title}
          content={summary.data.content}
          timestamp={summary.data.timestamp}
        />
      )}

      {/* KPI Row */}
      {kpis.isLoading ? (
        <LoadingSkeleton variant="card" count={4} />
      ) : kpis.data ? (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <KPICard
            label="Total Accounts"
            value={kpis.data.total_accounts.toLocaleString()}
            subtitle={`${kpis.data.active_subscribers.toLocaleString()} active`}
          />
          <KPICard
            label="Churn Rate (30d)"
            value={`${(kpis.data.churn_rate_30d * 100).toFixed(1)}%`}
            variant="danger"
          />
          <KPICard
            label="High Risk"
            value={kpis.data.high_risk_count.toLocaleString()}
            subtitle="accounts flagged"
            variant="danger"
          />
          <KPICard
            label="At-Risk MRR"
            value={`$${kpis.data.at_risk_mrr.toLocaleString()}`}
            subtitle={`CAC $${kpis.data.cac} · Save $${kpis.data.retention_cost_per_save}/acct`}
            variant="accent"
          />
        </div>
      ) : null}

      {/* Charts Row */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Churn Trend */}
        <ChartCard title="Churn Trend" subtitle="Monthly cancellations over time">
          {trends.isLoading ? (
            <div className="h-[280px] animate-pulse rounded bg-surface-hover" />
          ) : trends.data ? (
            <TrendLineChart data={trends.data} showPredicted />
          ) : null}
        </ChartCard>

        {/* Risk Distribution */}
        <ChartCard title="Risk Distribution" subtitle="Current subscriber risk tiers">
          {riskDist.isLoading ? (
            <div className="h-[280px] animate-pulse rounded bg-surface-hover" />
          ) : riskDist.data ? (
            <RiskDistributionChart
              low={riskDist.data.low}
              medium={riskDist.data.medium}
              high={riskDist.data.high}
            />
          ) : null}
        </ChartCard>
      </div>

      {/* Active / Inactive */}
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Subscriber Status" subtitle="Active vs inactive breakdown">
          {activeInactive.isLoading ? (
            <div className="h-[240px] animate-pulse rounded bg-surface-hover" />
          ) : activeInactive.data ? (
            <DonutChart
              data={[
                { name: "Active", value: activeInactive.data.active, color: "#22c55e" },
                { name: "Inactive", value: activeInactive.data.inactive, color: "#5c5d6e" },
                { name: "Recent Churn", value: activeInactive.data.recent_churn_30d, color: "#ef4444" },
              ]}
              innerLabel="Active"
              innerValue={activeInactive.data.active.toLocaleString()}
            />
          ) : null}
        </ChartCard>

        {/* Quick Stats */}
        {kpis.data && (
          <div className="grid grid-cols-2 gap-4">
            <KPICard
              label="Churned Accounts"
              value={kpis.data.churned_accounts.toLocaleString()}
              variant="danger"
            />
            <KPICard
              label="Active Subscribers"
              value={kpis.data.active_subscribers.toLocaleString()}
            />
            <KPICard
              label="Customer Acquisition Cost"
              value={`$${kpis.data.cac}`}
            />
            <KPICard
              label="Retention Cost/Save"
              value={`$${kpis.data.retention_cost_per_save}`}
              variant="accent"
            />
          </div>
        )}
      </div>
    </PageShell>
  );
}
