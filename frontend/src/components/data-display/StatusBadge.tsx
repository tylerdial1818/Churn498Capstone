import { clsx } from "clsx";

interface StatusBadgeProps {
  status: "pending" | "approved" | "sent" | "rejected";
  className?: string;
}

const STATUS_STYLES = {
  pending: "bg-warning/15 text-warning",
  approved: "bg-info/15 text-info",
  sent: "bg-success/15 text-success",
  rejected: "bg-risk-high/15 text-risk-high",
};

export default function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-badge px-2.5 py-0.5 text-xs font-medium capitalize",
        STATUS_STYLES[status],
        className,
      )}
    >
      {status}
    </span>
  );
}
