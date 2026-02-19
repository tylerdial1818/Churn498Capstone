import { clsx } from "clsx";

interface LoadingSkeletonProps {
  variant?: "card" | "row" | "chart";
  count?: number;
}

export default function LoadingSkeleton({ variant = "card", count = 1 }: LoadingSkeletonProps) {
  const items = Array.from({ length: count }, (_, i) => i);

  if (variant === "row") {
    return (
      <div className="space-y-3">
        {items.map((i) => (
          <div key={i} className="flex gap-4 rounded-card border border-border bg-surface p-4">
            <div className="h-4 w-24 animate-pulse rounded bg-surface-hover" />
            <div className="h-4 w-32 animate-pulse rounded bg-surface-hover" />
            <div className="h-4 w-16 animate-pulse rounded bg-surface-hover" />
          </div>
        ))}
      </div>
    );
  }

  if (variant === "chart") {
    return (
      <div className="rounded-card border border-border bg-surface p-card">
        <div className="mb-4 h-4 w-32 animate-pulse rounded bg-surface-hover" />
        <div className="h-64 animate-pulse rounded bg-surface-hover" />
      </div>
    );
  }

  return (
    <div className={clsx("grid gap-4", count > 1 && "grid-cols-2 lg:grid-cols-4")}>
      {items.map((i) => (
        <div key={i} className="rounded-card border border-border bg-surface p-card">
          <div className="h-3 w-20 animate-pulse rounded bg-surface-hover" />
          <div className="mt-3 h-7 w-16 animate-pulse rounded bg-surface-hover" />
          <div className="mt-2 h-3 w-24 animate-pulse rounded bg-surface-hover" />
        </div>
      ))}
    </div>
  );
}
