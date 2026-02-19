import PageShell from "@/components/layout/PageShell";
import KPICard from "@/components/data-display/KPICard";
import ChartCard from "@/components/data-display/ChartCard";
import LoadingSkeleton from "@/components/data-display/LoadingSkeleton";
import TrendLineChart from "@/components/charts/TrendLineChart";
import BarChart from "@/components/charts/BarChart";
import SHAPWaterfall from "@/components/charts/SHAPWaterfall";
import { clsx } from "clsx";
import {
  useChurnTrends,
  useSegments,
  useModelPerformance,
  useDriftStatus,
  useSHAPGlobal,
} from "@/hooks/useAnalytics";

export default function Analytics() {
  const trends = useChurnTrends(24);
  const segments = useSegments();
  const model = useModelPerformance();
  const drift = useDriftStatus();
  const shap = useSHAPGlobal();

  return (
    <PageShell title="Analytics" subtitle="Model performance and churn analysis">
      {/* Model Metrics */}
      {model.isLoading ? (
        <LoadingSkeleton variant="card" count={4} />
      ) : model.data ? (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <KPICard label="AUC-ROC" value={model.data.auc_roc.toFixed(3)} variant="accent" />
          <KPICard label="Precision" value={model.data.precision.toFixed(3)} />
          <KPICard label="Recall" value={model.data.recall.toFixed(3)} />
          <KPICard
            label="F1 Score"
            value={model.data.f1_score.toFixed(3)}
            subtitle={`Last trained: ${model.data.last_trained}`}
          />
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Churn Trends */}
        <ChartCard title="Churn Trends" subtitle="24-month cancellation trend">
          {trends.isLoading ? (
            <div className="h-[280px] animate-pulse rounded bg-surface-hover" />
          ) : trends.data ? (
            <TrendLineChart data={trends.data} showPredicted />
          ) : null}
        </ChartCard>

        {/* SHAP Feature Importance */}
        <ChartCard title="Global Feature Importance" subtitle="SHAP-based model explanations">
          {shap.isLoading ? (
            <div className="h-[280px] animate-pulse rounded bg-surface-hover" />
          ) : shap.data ? (
            <SHAPWaterfall features={shap.data} />
          ) : null}
        </ChartCard>
      </div>

      {/* Segments */}
      <div className="grid gap-4 lg:grid-cols-2">
        {segments.data && (
          <>
            <ChartCard title="Churn by Plan" subtitle="Risk breakdown by subscription tier">
              <BarChart
                data={segments.data.by_plan.map((s) => ({
                  label: s.plan,
                  value: s.churn_rate * 100,
                }))}
                height={240}
                color="#6366f1"
                valueFormatter={(v) => `${v.toFixed(1)}%`}
              />
            </ChartCard>
            <ChartCard title="Churn by Tenure" subtitle="Risk breakdown by account age">
              <BarChart
                data={segments.data.by_tenure.map((s) => ({
                  label: s.plan,
                  value: s.churn_rate * 100,
                }))}
                height={240}
                color="#818cf8"
                valueFormatter={(v) => `${v.toFixed(1)}%`}
              />
            </ChartCard>
          </>
        )}
      </div>

      {/* Drift Monitor */}
      {drift.data && (
        <ChartCard
          title="Data Drift Monitor"
          subtitle={`Last checked: ${drift.data.last_checked}`}
          actions={
            <span
              className={clsx(
                "rounded-badge px-2.5 py-0.5 text-xs font-medium capitalize",
                drift.data.overall_status === "ok" && "bg-success/15 text-success",
                drift.data.overall_status === "warning" && "bg-warning/15 text-warning",
                drift.data.overall_status === "drift" && "bg-risk-high/15 text-risk-high",
              )}
            >
              {drift.data.overall_status}
            </span>
          }
        >
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
            {drift.data.features.map((f) => (
              <div
                key={f.feature}
                className={clsx(
                  "rounded-lg border p-3 text-xs",
                  f.status === "ok" && "border-border bg-surface",
                  f.status === "warning" && "border-warning/30 bg-warning/5",
                  f.status === "drift" && "border-risk-high/30 bg-risk-high/5",
                )}
              >
                <p className="truncate font-medium text-text-secondary">
                  {f.feature.replace(/_/g, " ")}
                </p>
                <p className="mt-1 font-mono text-text-primary">PSI: {f.psi.toFixed(3)}</p>
                <span
                  className={clsx(
                    "capitalize",
                    f.status === "ok" && "text-success",
                    f.status === "warning" && "text-warning",
                    f.status === "drift" && "text-risk-high",
                  )}
                >
                  {f.status}
                </span>
              </div>
            ))}
          </div>
        </ChartCard>
      )}

      {/* Additional Model Stats */}
      {model.data && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <KPICard label="Log Loss" value={model.data.log_loss.toFixed(4)} />
          <KPICard label="Calibration Error" value={model.data.calibration_error.toFixed(4)} />
          <KPICard
            label="Training Samples"
            value={model.data.training_samples.toLocaleString()}
          />
          <KPICard label="Last Trained" value={model.data.last_trained} />
        </div>
      )}
    </PageShell>
  );
}
