import { clsx } from "clsx";

interface RiskBadgeProps {
  tier: "high" | "medium" | "low";
  className?: string;
}

const TIER_STYLES = {
  high: "bg-risk-high/15 text-risk-high",
  medium: "bg-risk-medium/15 text-risk-medium",
  low: "bg-risk-low/15 text-risk-low",
};

export default function RiskBadge({ tier, className }: RiskBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-badge px-2.5 py-0.5 text-xs font-medium capitalize",
        TIER_STYLES[tier],
        className,
      )}
    >
      {tier}
    </span>
  );
}
