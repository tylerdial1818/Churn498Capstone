import { clsx } from "clsx";

interface SHAPFeature {
  feature: string;
  importance: number;
  direction: "positive" | "negative";
}

interface SHAPWaterfallProps {
  features: SHAPFeature[];
  maxFeatures?: number;
}

export default function SHAPWaterfall({ features, maxFeatures = 10 }: SHAPWaterfallProps) {
  const displayed = features.slice(0, maxFeatures);
  const maxImportance = Math.max(...displayed.map((f) => f.importance));

  return (
    <div className="space-y-2">
      {displayed.map((f) => {
        const widthPct = (f.importance / maxImportance) * 100;
        return (
          <div key={f.feature} className="flex items-center gap-3">
            <span className="w-40 shrink-0 truncate text-right text-xs text-text-secondary">
              {f.feature.replace(/_/g, " ")}
            </span>
            <div className="relative h-5 flex-1 rounded bg-surface-hover">
              <div
                className={clsx(
                  "absolute inset-y-0 left-0 rounded transition-all",
                  f.direction === "positive" ? "bg-risk-high/60" : "bg-success/60",
                )}
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <span className="w-12 text-right text-xs font-mono text-text-secondary">
              {f.importance.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
