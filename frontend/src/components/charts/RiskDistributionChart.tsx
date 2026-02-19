import DonutChart from "./DonutChart";

interface RiskDistributionChartProps {
  low: number;
  medium: number;
  high: number;
}

export default function RiskDistributionChart({ low, medium, high }: RiskDistributionChartProps) {
  const total = low + medium + high;
  const data = [
    { name: "Low Risk", value: low, color: "#22c55e" },
    { name: "Medium Risk", value: medium, color: "#f59e0b" },
    { name: "High Risk", value: high, color: "#ef4444" },
  ];

  return (
    <div>
      <DonutChart
        data={data}
        innerLabel="Total Accounts"
        innerValue={total.toLocaleString()}
        height={220}
      />
      <div className="mt-3 flex justify-center gap-6">
        {data.map((d) => (
          <div key={d.name} className="flex items-center gap-2">
            <div className="h-2.5 w-2.5 rounded-full" style={{ background: d.color }} />
            <span className="text-xs text-text-secondary">{d.name}</span>
            <span className="text-xs font-medium text-text-primary">{d.value.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
