import { clsx } from "clsx";

interface KPICardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: { value: number; label: string };
  variant?: "default" | "accent" | "danger";
}

export default function KPICard({ label, value, subtitle, trend, variant = "default" }: KPICardProps) {
  return (
    <div
      className={clsx(
        "rounded-card border border-border bg-surface p-card",
        variant === "accent" && "border-accent/30",
        variant === "danger" && "border-risk-high/30",
      )}
    >
      <p className="text-xs font-medium uppercase tracking-wider text-text-muted">{label}</p>
      <p
        className={clsx(
          "mt-1 text-2xl font-semibold",
          variant === "accent" && "text-accent",
          variant === "danger" && "text-risk-high",
          variant === "default" && "text-text-primary",
        )}
      >
        {value}
      </p>
      {subtitle && <p className="mt-0.5 text-xs text-text-secondary">{subtitle}</p>}
      {trend && (
        <div className="mt-2 flex items-center gap-1.5">
          <span
            className={clsx(
              "text-xs font-medium",
              trend.value > 0 ? "text-risk-high" : "text-success",
            )}
          >
            {trend.value > 0 ? "+" : ""}
            {trend.value}%
          </span>
          <span className="text-xs text-text-muted">{trend.label}</span>
        </div>
      )}
    </div>
  );
}
